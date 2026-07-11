import os
import sys
import time
import json
from google import genai
from google.genai import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from predict_v2 import identify_representative_segments, analyze_audio
except ImportError:
    from ..predict_v2 import identify_representative_segments, analyze_audio

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


class NeuralAudioAgent:
    def __init__(self, api_key: str, mir_weight: float = 0.5, name: str = "Neural:AudioAnalyst"):
        self.name = name
        self.role = "Structural analysis and multimodal audio interpretation"
        self.client = genai.Client(api_key=api_key)
        self.mir_weight = max(0.0, min(1.0, mir_weight))

    def _trim_best_segment(self, file_path: str, best_segment: dict) -> str:
        if not PYDUB_AVAILABLE:
            print(f"[{self.name}]: Pydub nije instaliran. Šaljem ceo fajl na API...")
            return file_path
        try:
            print(f"[{self.name}]: Isecam reprezentativni segment for LLM...")
            song = AudioSegment.from_file(file_path)
            start_ms = int(best_segment['start'] * 1000)
            end_ms = int(best_segment['end'] * 1000)
            trimmed_song = song[start_ms:end_ms]
            trimmed_path = f"trimmed_temp_{os.path.basename(file_path)}"
            trimmed_song.export(trimmed_path, format="mp3")
            return trimmed_path
        except Exception as e:
            print(f"[{self.name}]: Greška pri sečenju fajla ({e}). Koristim originalni fajl.")
            return file_path

    def _get_deep_audio_insight(self, file_path: str, raw_metrics: dict) -> dict:
        """Šalje audio isečak na Gemini 2.0 sa ugrađenom autonomnom Retry logikom."""
        print(f"[{self.name}]: Pokrećem multimodalnu analizu produkcije (Gemini)...")

        json_schema = {
            "type": "OBJECT",
            "properties": {
                "production_quality_description": {"type": "STRING"},
                "vocal_presence_rating": {"type": "STRING", "description": "Opis vokala i miksa"},
                "potential_score": {"type": "NUMBER", "description": "Ocena od 0.0 do 1.0 komercijalnog potencijala"}
            },
            "required": ["production_quality_description", "vocal_presence_rating", "potential_score"]
        }

        max_attempts = 3
        initial_delay = 12.0  # Čekamo 12 sekundi ako dobijemo 429 Rate Limit

        for attempt in range(max_attempts):
            try:
                with open(file_path, "rb") as f:
                    audio_file = self.client.files.upload(
                        file=f,
                        config={'mime_type': 'audio/mpeg'}
                    )

                while audio_file.state.name == "PROCESSING":
                    time.sleep(1)
                    audio_file = self.client.files.get(name=audio_file.name)

                if audio_file.state.name == "FAILED":
                    raise Exception("Google AI nije uspeo da procesira audio fajl.")

                prompt = """
                Analyze this song segment. Evaluate the overall production quality and the vocal mix/presence.
                Provide your final evaluation through the requested schema. 
                The 'potential_score' must reflect how well this tracks aligns with modern commercial radio standards.
                """

                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[audio_file, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=json_schema,
                        temperature=0.2
                    )
                )

                # Čišćenje cloud resursa odmah nakon uspeha
                try:
                    self.client.files.delete(name=audio_file.name)
                    print(f"[{self.name}]: Privremeni fajl uspešno uklonjen sa Google servera.")
                except Exception as cloud_err:
                    print(f"[{self.name}]: Upozorenje - neuspešno brisanje sa klauda: {cloud_err}")

                if "trimmed_temp_" in file_path and os.path.exists(file_path):
                    os.remove(file_path)

                return json.loads(response.text)

            except Exception as e:
                # Provera da li je u pitanju privremeni Rate Limit (429)
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_attempts - 1:
                        print(f"⚠️ [{self.name}]: Detektovan 429 Rate Limit (Pokušaj {attempt + 1}/{max_attempts}).")
                        print(f"⏳ [Oporavak]: Pauziram izvršavanje na {initial_delay} sekundi pre ponovnog pokušaja...")
                        time.sleep(initial_delay)
                        continue  # Skače na sledeću iteraciju petlje (ponovni pokušaj)

                # Za sve ostale trajne greške (ili ako su ispucani svi pokušaji), idemo na lokalni fallback
                print(f"❌ [{self.name}]: KONAČNA GOOGLE GREŠKA NAKON RETRY-ja: {str(e)}")
                print(f"[{self.name}]: Prelazim na lokalni heuristički fallback.")
                if os.path.exists(file_path) and "trimmed_temp_" in file_path:
                    os.remove(file_path)
                return self._generate_heuristic_fallback(raw_metrics)

        # Sigurnosni povratak u slučaju nepredviđenog izlaza iz petlje
        return self._generate_heuristic_fallback(raw_metrics)

    def _generate_heuristic_fallback(self, raw_metrics: dict) -> dict:
        energy = raw_metrics.get('energy_%', 50) / 100
        return {
            "production_quality_description": "Fallback mode activated. Local analysis based on energy levels.",
            "vocal_presence_rating": "Undetected (Local Fallback)",
            "potential_score": round(energy, 2)
        }

    def process_track(self, file_path: str) -> dict:
        print(f"\n[{self.name}]: Započeta obrada pesme: {file_path}")
        raw_features = analyze_audio(file_path)
        segments = identify_representative_segments(file_path)
        best_segment = max(segments, key=lambda x: x['energy_score'])

        target_audio_path = self._trim_best_segment(file_path, best_segment)
        ai_insights = self._get_deep_audio_insight(target_audio_path, raw_features)

        mir_signal = raw_features.get('energy_%', 0) / 100
        ai_signal = ai_insights.get('potential_score', 0.5)
        activation_value = (mir_signal * self.mir_weight) + (ai_signal * (1.0 - self.mir_weight))

        report = {
            "agent_name": self.name,
            "config_weights": {
                "mir_weight": self.mir_weight,
                "ai_weight": round(1.0 - self.mir_weight, 2)
            },
            "activation_value": round(activation_value, 4),
            "status": "completed",
            "findings": {
                "deterministic_mir_metrics": {
                    "bpm": round(raw_features.get('bpm', 0), 1),
                    "key": f"{raw_features.get('detected_key_name', 'Unknown')} {raw_features.get('detected_mode_name', '')}",
                    "energy_level": f"{round(raw_features.get('energy_%', 0), 1)}%"
                },
                "neural_semantic_analysis": ai_insights,
                "structural_context": {
                    "total_segments_analyzed": len(segments),
                    "focal_segment": {
                        "timestamp": f"{best_segment['start']}s - {best_segment['end']}s",
                        "selection_reason": "Maximum Local Energy Potential"
                    }
                }
            }
        }
        return report


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
    if not GEMINI_API_KEY:
        print("❌ Greška: Iskopiraj GOOGLE_AI_API_KEY u .env")
    else:
        agent = NeuralAudioAgent(api_key=GEMINI_API_KEY, mir_weight=0.4)
        test_file = "/Users/admin/Downloads/National Anthem of Andorra.mp3"
        if os.path.exists(test_file):
            print(json.dumps(agent.process_track(test_file), indent=2))