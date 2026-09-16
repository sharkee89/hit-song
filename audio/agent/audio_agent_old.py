import json
import librosa
import numpy as np
import os
import sys
import torch
import joblib
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audio.ai.ai_engine import AIEngine
from audio.features.chroma.chroma_utils import get_chroma_neural_network_data
from audio.features.rms.rms_utils import get_rms_neural_network_data
from audio.features.spectral_centroid.spectral_centroi_utils import get_spectral_neural_network_data


class AudioAgent:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            try:
                from audio.engine.gpu_engine import GPUAudioEngine
                from audio.kernels.rms_kernel import triton_rms
                self.engine = GPUAudioEngine()
            except ImportError as e:
                print(f"⚠️ CUDA is unavailable, error during import gpu engine ({e}). Switching to CPU.")
                from audio.engine.cpu_engine import CPUAudioEngine
                self.engine = CPUAudioEngine()
                self.device = "cpu"
        else:
            from audio.engine.cpu_engine import CPUAudioEngine
            self.engine = CPUAudioEngine()

        print(f"[AudioAgent] Device: {self.device}")

    def process_audio_file(self, file_path: str) -> str:
        return self.engine.process_audio_file(file_path)

    def get_ai_analysis(self, audio_data):
        ai_engine = AIEngine()
        return ai_engine.get_audio_detail_analysis(audio_data)

    def get_data(self, audio_data, audio_detail_analysis, nn_data):
        audio_data_dict = json.loads(audio_data)
        audio_data_dict["ai_analysis"] = audio_detail_analysis
        audio_data_dict["nn_data"] = nn_data
        return json.dumps(audio_data_dict, indent=4)

    def get_neural_network_input(self, audio_data: dict) -> dict:
        rms_nn_data = get_rms_neural_network_data(audio_data)
        spectral_nn_data = get_spectral_neural_network_data(audio_data)
        chroma_nn_data = get_chroma_neural_network_data(audio_data)

        return {
            "rms_nn_data": rms_nn_data,
            "spectral_nn_data": spectral_nn_data,
            **chroma_nn_data
        }

    def process(self, file_path: str) -> str:
        audio_data = self.process_audio_file(file_path)
        audio_detail_analysis = self.get_ai_analysis(audio_data)
        nn_data = self.get_neural_network_input(json.loads(audio_data))
        return self.get_data(audio_data, audio_detail_analysis, nn_data)

    def to_spotify_features(self, analysis_json_str: str) -> list:
        return self.engine.to_spotify_features(analysis_json_str)

    def to_spotify_features_from_librosa(self, file_path: str) -> list:
        """
        Ekstrahuje 12-dimenzionalni Spotify format prilagođen realnim vrednostima baze.
        """
        y, sr = librosa.load(file_path, sr=None)

        # 1. Trajanje u milisekundama
        duration_ms = (len(y) / sr) * 1000.0

        # 2. Tempo (BPM)
        tempo_arr, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_arr.item() if isinstance(tempo_arr, np.ndarray) else tempo_arr)
        # Ako je izrazito sporo, prepuštamo algoritmu, ali možemo korigovati ako je izvan standardnog pop opsega
        if 60 <= tempo < 90:
            tempo *= 2.0

        # 3. Loudness u dB (korigovano sa realnijim offsetom za masterovan audio)
        rms = librosa.feature.rms(y=y)[0]
        non_silent_rms = rms[rms > 0.01]  # Gledamo samo glasnije delove
        mean_rms = float(np.mean(non_silent_rms)) if len(non_silent_rms) > 0 else float(np.mean(rms))
        mean_rms = max(mean_rms, 1e-5)
        # Spotify loudness obično ide od -60 do 0, pop hitovi su oko -5 do -9 dB
        loudness = float(20.0 * np.log10(mean_rms) + 3.0)
        loudness = max(min(loudness, 0.0), -35.0)

        # 4. Energija bazirana na RMS-u i spektralnoj širini
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        energy = float(np.clip(np.mean(rms) * 12.0 + (np.mean(spec_rolloff) / sr) * 0.3, 0.0, 1.0))

        # 5. Chroma (Key i Mode)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key = float(np.argmax(chroma_mean))
        mode = 1.0 if chroma_mean[(int(key) + 4) % 12] > chroma_mean[(int(key) + 3) % 12] else 0.0

        # 6. Speechiness (govor je obično u užem opsegu srednjih frekvencija)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        speechiness = float(np.clip(np.mean(spec_cent) / 8000.0 * 0.5, 0.0, 0.33))  # Većina pesama je ispod 0.3

        # 7. Acousticness (realnija procena na osnovu visoke frekvencije i energije)
        # Moderne produkovane pesme imaju ekstremno nisku akustičnost
        acousticness = float(np.clip(1.0 - (energy * 1.5), 0.0001, 1.0))

        # 8. Instrumentalness (većina pesama sa vokalima je blizu 0)
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_ratio = np.sum(np.abs(y_harm)) / (np.sum(np.abs(y)) + 1e-6)
        instrumentalness = float(np.clip((harm_ratio - 0.7) * 3.0, 0.0, 1.0))  # Stroži prag da ne lažira instrumentale

        # 9. Liveness (procena preko Crest faktora)
        peak_rms = float(np.max(rms))
        crest_factor = peak_rms / (float(np.mean(rms)) + 1e-6)
        liveness = float(np.clip(crest_factor / 15.0, 0.05, 0.4))  # Većina studijskih pesama je ispod 0.3

        # 10. Danceability (stabilnija procena preko ritmičke strukture)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_std = np.std(onset_env)
        danceability = float(np.clip((np.mean(onset_env) / 4.0) + (onset_std / 5.0), 0.1, 0.95))

        # 11. Valence (pozitivnost)
        valence = float(np.clip((energy * 0.5) + (mode * 0.2) + (danceability * 0.3), 0.0, 1.0))

        return [
            danceability, energy, key, loudness, mode,
            speechiness, acousticness, instrumentalness, liveness,
            valence, tempo, duration_ms
        ]

    def compare_with_librosa(self, file_path: str, our_vector: list):
        y_librosa, sr = librosa.load(file_path, sr=None)

        tempo_librosa, _ = librosa.beat.beat_track(y=y_librosa, sr=sr)
        if isinstance(tempo_librosa, np.ndarray):
            tempo_librosa = tempo_librosa.item()

        rms_librosa = librosa.feature.rms(y=y_librosa)
        loudness_librosa = float(20.0 * np.log10(np.maximum(np.mean(rms_librosa), 1e-5)))

        centroid_librosa = np.mean(librosa.feature.spectral_centroid(y=y_librosa, sr=sr))
        duration_librosa = (len(y_librosa) / sr) * 1000.0

        print("--- POREĐENJE VREDNOSTI ---")
        print(f"Tempo:     Naš agent = {our_vector[10]} | Librosa = {tempo_librosa:.2f}")
        print(f"Loudness:  Naš agent = {our_vector[3]:.2f} | Librosa = {loudness_librosa:.2f}")
        print(f"Trajanje:  Naš agent = {our_vector[11]} | Librosa = {duration_librosa:.2f}")
        print(f"Centroid:  Librosa srednja vrednost = {centroid_librosa:.2f} Hz")


if __name__ == "__main__":
    load_dotenv()
    file_path = os.getenv("AUDIO_FILE_PATH", "")
    agent = AudioAgent()

    if file_path:
        # Možeš testirati i novu funkciju preko procesiranog JSON-a:
        raw_json_str = agent.process_audio_file(file_path)
        spotify_vector = agent.to_spotify_features(raw_json_str)

        # spotify_vector = agent.to_spotify_features_from_librosa(file_path)
        print("✅ Generisan 12-dimenzionalni vektor za model:", spotify_vector)

        # Putanje do joblib fajlova u folderu 'joblib' koji se nalazi u korijenu projekta
        joblib_dir = os.path.join(PROJECT_ROOT, "joblib")
        model_path = os.path.join(joblib_dir, "hit_random_forest.joblib")
        scaler_path = os.path.join(joblib_dir, "scaler.joblib")

        try:
            rf_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # Osnovni vektor iz dataseta (koji daje 99.33%)
            baseline_vector = [
                0.513,  # danceability
                0.731,  # energy
                1.0,  # key
                -5.941,  # loudness
                1.0,  # mode
                0.0598,  # speechiness
                0.00143,  # acousticness
                9.54e-05,  # instrumentalness
                0.0897,  # liveness
                0.334,  # valence
                171.0,  # tempo
                200040.0  # duration_ms
            ]

            # Hardkodovane vrednosti direktno sa Spotify API-ja za "Blinding Lights"
            api_vector = [
                0.747,  # danceability
                0.507,  # energy
                2.0,  # key
                -10.171,  # loudness
                1.0,  # mode
                0.0358,  # speechiness
                0.2,  # acousticness
                0.0608,  # instrumentalness
                0.117,  # liveness
                0.438,  # valence
                104.978,  # tempo
                210373.0  # duration_ms
            ]

            feature_names = [
                "danceability", "energy", "key", "loudness", "mode",
                "speechiness", "acousticness", "instrumentalness",
                "liveness", "valence", "tempo", "duration_ms"
            ]

            print("\n--- DIJAGNOSTIKA POJEDINAČNIH PARAMETARA ---")
            for i, name in enumerate(feature_names):
                test_vector = baseline_vector.copy()
                test_vector[i] = api_vector[i]

                x_scaled = scaler.transform([test_vector])
                X_input = np.hstack([x_scaled, x_scaled[:, :12]])
                probabilities = rf_model.predict_proba(X_input)[0]
                hit_prob = probabilities[1] * 100

                print(
                    f"{name:18} | Baseline: {baseline_vector[i]:<10} | API: {api_vector[i]:<10} | Nova verovatnoća hita: {hit_prob:.2f}%")

            # Finalna predikcija sa čistim API vektorom
            x_scaled = scaler.transform([api_vector])
            X_input = np.hstack([x_scaled, x_scaled[:, :12]])

            prediction = rf_model.predict(X_input)[0]
            probabilities = rf_model.predict_proba(X_input)[0]
            hit_prob = probabilities[1] * 100

            print(f"\n--- REZULTAT PREDIKCIJE (Sa API Vektorom) ---")
            print(f"Status: {'🔥 HIT' if prediction == 1 else '❌ NIJE HIT'}")
            print(f"Verovatnoća hita: {hit_prob:.2f}%")

        except FileNotFoundError as e:
            print(f"⚠️ Nisu pronađeni joblib fajlovi u folderu 'joblib': {e}")
    else:
        print("❌ AUDIO_FILE_PATH nije definisan u .env")