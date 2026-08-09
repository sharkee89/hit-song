import os
import io
import time
import json
import torch
import torchaudio
from google import genai
from google.genai import types


class GPUAudioAgent:
    def __init__(self, api_key: str, mir_weight: float = 0.5, name: str = "GPU:AudioAnalyst", device: str = None):
        self.name = name
        self.role = "GPU-accelerated structural analysis and multimodal audio interpretation"
        self.client = genai.Client(api_key=api_key)
        self.mir_weight = max(0.0, min(1.0, mir_weight))

        # Automatska detekcija GPU-a
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[{self.name}]: Inicijalizovan na uređaju -> {self.device.upper()}")

    def _load_audio_to_vram(self, file_path: str):
        """
        Učitava MP3 sa diska direktno u RAM, a zatim šalje u GPU VRAM kao float32 Tensor.
        Vraća: waveform (Tensor u VRAM-u), sample_rate (int)
        """
        start_time = time.perf_counter()

        # Učitavanje i dekodiranje u float32 PCM tenzor
        waveform, sample_rate = torchaudio.load(file_path)

        # Prebacivanje na GPU u VRAM
        waveform_gpu = waveform.to(self.device)

        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"[{self.name}]: MP3 dekodiran i prebačen u VRAM za {elapsed:.2f} ms")

        return waveform_gpu, sample_rate

    def _trim_segment_in_vram(self, waveform_gpu: torch.Tensor, sample_rate: int, start_sec: float,
                              end_sec: float) -> bytes:
        """
        Apsolutno bez diska: Vrši sečenje (slicing) audio tenzora direktno u VRAM-u
        i konvertuje isečak u In-Memory BytesBuffer (WAV format) spreman za API.
        """
        start_time = time.perf_counter()

        start_frame = int(start_sec * sample_rate)
        end_frame = int(end_sec * sample_rate)

        # Sečenje u GPU VRAM-u (slice operacija u nanosekundama)
        trimmed_gpu = waveform_gpu[:, start_frame:end_frame]

        # Vraćamo samo isečeni deo na CPU radi serijalizacije u RAM bajtove
        trimmed_cpu = trimmed_gpu.cpu()

        # Pakovanje u In-Memory WAV bafer (zamenjuje Pydub i temporary fajlove na disku)
        buffer = io.BytesIO()
        torchaudio.save(buffer, trimmed_cpu, sample_rate, format="wav")
        buffer.seek(0)

        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"[{self.name}]: VRAM slicing i RAM baferovanje završeni za {elapsed:.2f} ms (0 bajtova upisano na disk)")

        return buffer.read()

    def _get_deep_audio_insight_from_buffer(self, audio_bytes: bytes, raw_metrics: dict) -> dict:
        """Šalje audio isečak iz RAM bajtova direktno na Gemini API."""
        print(f"[{self.name}]: Šaljem In-Memory audio bafer na Gemini AI...")

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
        initial_delay = 12.0

        for attempt in range(max_attempts):
            try:
                # Slanje sirovih bajtova iz RAM-a na Google Cloud bez privremenih fajlova!
                audio_file = self.client.files.upload(
                    file=io.BytesIO(audio_bytes),
                    config={
                        'mime_type': 'audio/wav',
                        'display_name': 'in_memory_segment.wav'
                    }
                )

                while audio_file.state.name == "PROCESSING":
                    time.sleep(1)
                    audio_file = self.client.files.get(name=audio_file.name)

                if audio_file.state.name == "FAILED":
                    raise Exception("Google AI nije uspeo da procesira in-memory audio bafer.")

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

                # Čišćenje cloud resursa nakon obrade
                try:
                    self.client.files.delete(name=audio_file.name)
                except Exception:
                    pass

                return json.loads(response.text)

            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_attempts - 1:
                        print(f"⚠️ [{self.name}]: 429 Rate Limit. Pauza {initial_delay}s...")
                        time.sleep(initial_delay)
                        continue

                print(f"❌ [{self.name}]: Greška pri Gemini analizi: {str(e)}")
                return self._generate_heuristic_fallback(raw_metrics)

        return self._generate_heuristic_fallback(raw_metrics)

    def _generate_heuristic_fallback(self, raw_metrics: dict) -> dict:
        energy = raw_metrics.get('energy_%', 50) / 100
        return {
            "production_quality_description": "GPU Fallback mode. Analysis derived from local VRAM metrics.",
            "vocal_presence_rating": "Undetected (Local Fallback)",
            "potential_score": round(energy, 2)
        }

    def process_track(self, file_path: str) -> dict:
        total_start = time.perf_counter()
        print(f"\n[{self.name}]: Započeta GPU obrada: {file_path}")

        # 1. Učitavanje direktno u VRAM
        waveform_gpu, sample_rate = self._load_audio_to_vram(file_path)

        # 2. Privremena mock/fallback struktura za MIR dok ne napišemo Triton Kernel u Fazi 2
        # (Privremeno koristimo brzi PyTorch GPU RMS proračun za energiju)
        rms_energy = torch.sqrt(torch.mean(waveform_gpu ** 2)).item()
        mock_raw_features = {
            'bpm': 120.0,
            'detected_key_name': 'C',
            'detected_mode_name': 'Major',
            'energy_%': min(100.0, rms_energy * 1000.0)
        }

        # Simulacija reprezentativnog segmenta (npr. 30s do 60s)
        best_segment = {'start': 30.0, 'end': 60.0, 'energy_score': mock_raw_features['energy_%']}

        # 3. Sečenje u VRAM-u i pakovanje u RAM bajtove
        audio_bytes = self._trim_segment_in_vram(
            waveform_gpu, sample_rate, best_segment['start'], best_segment['end']
        )

        # 4. Slanje na Gemini
        ai_insights = self._get_deep_audio_insight_from_buffer(audio_bytes, mock_raw_features)

        # 5. Izračunavanje konačne ocene
        mir_signal = mock_raw_features['energy_%'] / 100.0
        ai_signal = ai_insights.get('potential_score', 0.5)
        activation_value = (mir_signal * self.mir_weight) + (ai_signal * (1.0 - self.mir_weight))

        total_elapsed = (time.perf_counter() - total_start) * 1000

        return {
            "agent_name": self.name,
            "device": self.device,
            "total_execution_time_ms": round(total_elapsed, 2),
            "activation_value": round(activation_value, 4),
            "findings": {
                "deterministic_mir_metrics": mock_raw_features,
                "neural_semantic_analysis": ai_insights
            }
        }


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    GEMINI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
    if GEMINI_API_KEY:
        agent = GPUAudioAgent(api_key=GEMINI_API_KEY)
        test_file = "/Users/admin/Downloads/National Anthem of Andorra.mp3"
        if os.path.exists(test_file):
            print(json.dumps(agent.process_track(test_file), indent=2))