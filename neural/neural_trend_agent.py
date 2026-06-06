import numpy as np
import joblib
import json
import os
import sys

# Osiguravamo putanje za uvoz tvojih custom modula
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from predict_v2 import identify_representative_segments, analyze_audio
except ImportError:
    # Fallback ako se skripta pokreće direktno iz foldera gde su agenti
    from predict_v2 import analyze_audio


class NeuralTrendAgent:
    def __init__(self, name="Neural: TrendMatcher"):
        self.name = name
        # Učitavanje unapred izračunatih vektora trendova sa tržišta
        try:
            # Ovi fajlovi bi trebalo da se nalaze u istom folderu ili da putanja bude definisana
            self.trend_vector = joblib.load('trend_vector_v2.joblib')
            self.trend_metadata = joblib.load('trend_metadata_v2.joblib')
        except Exception as e:
            print(f"[{self.name}] Error loading trend data: {e}")
            self.trend_vector = None
            self.trend_metadata = {}

    def _calculate_trend_alignment(self, audio_data):
        """Računa geometrijsku udaljenost (Euklidsku normu) pesme od idealnog trenda."""
        if self.trend_vector is None:
            return 0.5

        # Uzimamo samo kolone koje postoje u oba seta podataka (audio vs trend)
        common_cols = [c for c in self.trend_vector.index if c in audio_data]
        audio_values = np.array([audio_data[c] for c in common_cols])
        trend_values = self.trend_vector[common_cols].values

        # L2 norma (udaljenost u višedimenzionalnom prostoru)
        dist = np.linalg.norm(audio_values - trend_values)

        # Eksponencijalni pad skora - što je distanca manja, skor je bliži 1.0
        alignment_score = np.exp(-dist / 50)

        return round(float(alignment_score), 4)

    def _check_harmonic_alignment(self, audio_data):
        """Dodaje bonus ako se pesma poklapa sa najpopularnijim tonalitetom na tržištu."""
        dom_key = self.trend_metadata.get('dominant_key')
        dom_mode = self.trend_metadata.get('dominant_mode')

        if (audio_data.get('detected_key_name') == dom_key and
                audio_data.get('detected_mode_name') == dom_mode):
            return 0.15
        return 0.0

    def process_track(self, file_path):
        """Glavna metoda koja poredi pesmu sa tržišnim trendovima."""
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}

        print(f"[{self.name}]: Analysing matching with market trend...")

        # 1. Dobijanje audio karakteristika preko predict_v2
        audio_data = analyze_audio(file_path)

        # 2. Proračun bazične usklađenosti (Trend Alignment)
        base_signal = self._calculate_trend_alignment(audio_data)

        # 3. Provera harmonskog bonusa (da li je u popularnom tonalitetu)
        bonus = self._check_harmonic_alignment(audio_data)

        # Finalni activation_value (max 1.0)
        activation_value = min(base_signal + bonus, 1.0)

        return {
            "agent_name": self.name,
            "activation_value": round(activation_value, 4),
            "status": "completed",
            "findings": {
                "distance_from_trend": round(1 - base_signal, 4),
                "harmonic_bonus_applied": bonus > 0,
                "target_trend_scale": f"{self.trend_metadata.get('dominant_key', 'Unknown')} {self.trend_metadata.get('dominant_mode', 'Unknown')}"
            }
        }


# --- TEST BLOK ---
if __name__ == "__main__":
    # Inicijalizacija agenta
    agent = NeuralTrendAgent()

    # Putanja do tvog mp3 fajla za testiranje (new_zealand_anthem.mp3)
    test_file = "/Users/admin/Downloads/new_zealand_anthem.mp3"

    if os.path.exists(test_file):
        report = agent.process_track(test_file)
        print("\n--- TREND ANALYSIS REPORT ---")
        print(json.dumps(report, indent=2))
    else:
        # Prikazujemo dummy podatke ako fajl ne postoji, samo radi provere strukture
        print(f"⚠️ Test fajl '{test_file}' nije pronađen. Proveri putanju.")

        # Simulacija za prikaz rezultata bez fajla (opciono)
        dummy_audio = {
            'detected_key_name': 'C',
            'detected_mode_name': 'Major'
        }
        print("\nPrimer strukture izveštaja (Simulacija):")
        print(json.dumps(agent._check_harmonic_alignment(dummy_audio), indent=2))