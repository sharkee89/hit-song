import numpy as np
import joblib
try:
    from predict_v2 import identify_representative_segments, analyze_audio
except ImportError:
    from ..predict_v2 import analyze_audio

class NeuralTrendAgent:
    def __init__(self, name="Neural: TrendMatcher"):
        self.name = name
        try:
            self.trend_vector = joblib.load('trend_vector_v2.joblib')
            self.trend_metadata = joblib.load('trend_metadata_v2.joblib')
        except Exception as e:
            print(f"[{self.name}] Error loading trend data: {e}")
            self.trend_vector = None

    def _calculate_trend_alignment(self, audio_data):
        if self.trend_vector is None:
            return 0.5

        common_cols = [c for c in self.trend_vector.index if c in audio_data]
        audio_values = np.array([audio_data[c] for c in common_cols])
        trend_values = self.trend_vector[common_cols].values

        dist = np.linalg.norm(audio_values - trend_values)

        alignment_score = np.exp(-dist / 50)

        return round(float(alignment_score), 4)

    def _check_harmonic_alignment(self, audio_data):
        dom_key = self.trend_metadata.get('dominant_key')
        dom_mode = self.trend_metadata.get('dominant_mode')

        if (audio_data['detected_key_name'] == dom_key and
                audio_data['detected_mode_name'] == dom_mode):
            return 0.15
        return 0.0

    def process_track(self, file_path):
        print(f"[{self.name}]: Analysing matching with market trend...")

        audio_data = analyze_audio(file_path)

        base_signal = self._calculate_trend_alignment(audio_data)

        bonus = self._check_harmonic_alignment(audio_data)

        activation_value = min(base_signal + bonus, 1.0)

        return {
            "agent_name": self.name,
            "activation_value": round(activation_value, 4),
            "status": "completed",
            "findings": {
                "distance_from_trend": round(1 - base_signal, 4),
                "harmonic_bonus_applied": bonus > 0,
                "target_trend_scale": f"{self.trend_metadata.get('dominant_key')} {self.trend_metadata.get('dominant_mode')}"
            }
        }