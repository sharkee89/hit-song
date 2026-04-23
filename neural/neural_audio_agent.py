import time
import re
from google import genai
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from predict_v2 import identify_representative_segments, analyze_audio
except ImportError:
    from ..predict_v2 import identify_representative_segments, analyze_audio


class NeuralAudioAgent:
    def __init__(self, api_key, name="Neural: AudioAnalyst"):
        self.name = name
        self.role = "Structural analysis and multimodal interpretation"
        self.client = genai.Client(api_key=api_key)

    def _get_deep_audio_insight(self, file_path, raw_metrics):
        print(f"[{self.name}]: Deep audio analysis & scoring...")
        try:
            with open(file_path, "rb") as f:
                audio_file = self.client.files.upload(
                    file=f,
                    config={'mime_type': 'audio/mpeg'}
                )

            time.sleep(2)

            prompt = """
            Analyze this song. Provide a brief description of production quality and vocal presence.
            Then, at the very end, provide a 'Potential Score' between 0.0 and 1.0 
            based on how modern and 'hit-ready' the production sounds.
            Format for score: [SCORE: 0.XX]
            """

            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[audio_file, prompt]
            )
            return response.text
        except Exception as e:
            if "429" in str(e):
                return self._generate_heuristic_analysis(raw_metrics)
            return f"Error: {str(e)}"

    def _extract_score_from_text(self, text):
        match = re.search(r"\[SCORE:\s*(\d+\.?\d*)\]", text)
        if match:
            return float(match.group(1))
        return 0.5

    def _generate_heuristic_analysis(self, raw_metrics):
        energy = raw_metrics.get('energy_%', 0)
        score = energy / 100
        return f"Local analysis: stable tempo. [SCORE: {score:.2f}]"

    def process_track(self, file_path):
        print(f"[{self.name}]: Initiating song analysis: {file_path}...")

        raw_features = analyze_audio(file_path)
        segments = identify_representative_segments(file_path)
        best_segment = max(segments, key=lambda x: x['energy_score'])

        ai_interpretation = self._get_deep_audio_insight(file_path, raw_features)

        mir_signal = raw_features.get('energy_%', 0) / 100
        ai_signal = self._extract_score_from_text(ai_interpretation)

        activation_value = (mir_signal + ai_signal) / 2

        report = {
            "agent_name": self.name,
            "activation_value": round(activation_value, 4),
            "status": "completed",
            "findings": {
                "global_metrics": {
                    "bpm": round(raw_features['bpm'], 1),
                    "key": f"{raw_features['detected_key_name']} {raw_features['detected_mode_name']}",
                    "energy_level": f"{round(raw_features['energy_%'], 1)}%"
                },
                "ai_semantic_analysis": ai_interpretation,
                "structure": {
                    "total_segments": len(segments),
                    "highlight_segment": {
                        "timestamp": f"{best_segment['start']}s - {best_segment['end']}s",
                        "reason": "Energy potential"
                    }
                }
            }
        }
        return report