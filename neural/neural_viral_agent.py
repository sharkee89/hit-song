import numpy as np
import json
import os
import sys

# Podešavanje putanja za uvoz modula iz roditeljskog direktorijuma
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from predict_v2 import identify_representative_segments
except ImportError:
    # Fallback za direktno pokretanje iz foldera sa agentima
    from predict_v2 import identify_representative_segments


class NeuralViralAgent:
    def __init__(self, name="NeuralViralAgent"):
        self.name = name

    def _calculate_viral_potential(self, segments):
        """
        Računa verovatnoću viralnosti na osnovu energije, vokala i strukture.
        """
        if not segments:
            return 0.1

        # Fokus na najenergičniji trenutak (npr. 'drop' ili vrhunac refrena)
        energies = [s['energy_score'] for s in segments]
        max_energy = max(energies) if energies else 0

        # Prisustvo vokala je ključno za TikTok/Reels formate (lip-sync, pozadina)
        vocal_presence = [s['vocal_presence_index'] for s in segments]
        avg_vocal = np.mean(vocal_presence) if vocal_presence else 0

        # Bonus za detektovan jasan refren ili deonice visokog intenziteta
        has_strong_chorus = any(s['label_candidate'] == "High Energy/Chorus" for s in segments)
        chorus_bonus = 0.2 if has_strong_chorus else 0.0

        raw_signal = (max_energy * 0.4) + (avg_vocal * 0.4) + chorus_bonus

        return round(min(raw_signal, 1.0), 4)

    def process_track(self, file_path):
        """
        Analizira audio fajl i vraća izveštaj o potencijalnoj viralnosti.
        """
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"Fajl nije pronađen: {file_path}"}

        print(f"[{self.name}]: Scanning audio segments for viral potential...")

        # Segmentacija pesme preko bazne logike
        segments = identify_representative_segments(file_path)

        # Proračun viralnog signala
        activation_value = self._calculate_viral_potential(segments)

        # Identifikacija najboljeg isečka (npr. onaj koji treba koristiti za TikTok)
        best_segment = max(segments, key=lambda x: x['energy_score']) if segments else None

        return {
            "agent_name": self.name,
            "activation_value": round(activation_value, 4),
            "status": "completed",
            "findings": {
                "viral_peak_timestamp": f"{best_segment['start']}s - {best_segment['end']}s" if best_segment else "N/A",
                "vocal_clarity_score": round(avg_vocal, 2) if 'avg_vocal' in locals() else 0.0,
                "tiktok_viability": "High" if activation_value > 0.7 else "Medium" if activation_value > 0.4 else "Low"
            }
        }


# --- TEST BLOK ---
if __name__ == "__main__":
    # Inicijalizacija agenta
    agent = NeuralViralAgent()

    # Putanja do tvog fajla (npr. new_zealand_anthem.mp3)
    test_file = "/Users/admin/Downloads/new_zealand_anthem.mp3"

    if os.path.exists(test_file):
        report = agent.process_track(test_file)
        print("\n--- VIRAL SCOUT REPORT ---")
        print(json.dumps(report, indent=2))
    else:
        print(f"⚠️ Test fajl '{test_file}' nije pronađen.")

        # Simulacija rezultata radi provere strukture koda
        print("\nPrimer strukture izveštaja (Simulacija):")
        dummy_segments = [
            {'energy_score': 0.85, 'vocal_presence_index': 0.7, 'label_candidate': 'High Energy/Chorus', 'start': 30,
             'end': 35},
            {'energy_score': 0.4, 'vocal_presence_index': 0.5, 'label_candidate': 'Verse', 'start': 0, 'end': 5}
        ]
        simulated_val = agent._calculate_viral_potential(dummy_segments)
        print(f"Simulirani Activation Value: {simulated_val}")