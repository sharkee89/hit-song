import numpy as np
try:
    from predict_v2 import identify_representative_segments
except ImportError:
    from ..predict_v2 import identify_representative_segments


class NeuralViralAgent:
    def __init__(self, name="Neural: ViralScouter"):
        self.name = name

    def _calculate_viral_potential(self, segments):
        if not segments:
            return 0.1

        energies = [s['energy_score'] for s in segments]
        max_energy = max(energies) if energies else 0

        vocal_presence = [s['vocal_presence_index'] for s in segments]
        avg_vocal = np.mean(vocal_presence) if vocal_presence else 0

        has_strong_chorus = any(s['label_candidate'] == "High Energy/Chorus" for s in segments)
        chorus_bonus = 0.2 if has_strong_chorus else 0.0

        raw_signal = (max_energy * 0.4) + (avg_vocal * 0.4) + chorus_bonus

        return round(min(raw_signal, 1.0), 4)

    def process_track(self, file_path):
        print(f"[{self.name}]: Scanning audio segments for viral potential...")

        segments = identify_representative_segments(file_path)

        activation_value = self._calculate_viral_potential(segments)

        best_segment = max(segments, key=lambda x: x['energy_score']) if segments else None

        return {
            "agent_name": self.name,
            "activation_value": round(activation_value, 4),
            "status": "completed",
            "findings": {
                "viral_peak_timestamp": f"{best_segment['start']}s - {best_segment['end']}s" if best_segment else "N/A",
                "vocal_clarity_score": round(activation_value * 0.8, 2),
                "tiktok_viability": "High" if activation_value > 0.7 else "Medium" if activation_value > 0.4 else "Low"
            }
        }