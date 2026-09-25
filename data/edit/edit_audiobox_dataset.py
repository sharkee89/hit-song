import os
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import librosa

# Forsiramo CUDA da bude vidljiva (ukoliko je zatreba za druge procese)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset" / "spotify_tracks_audiobox.parquet"
OUTPUT_DATASET_PATH = PROJECT_ROOT / "data" / "dataset" / "spotify_tracks_audiobox_librosa.parquet"
TEMP_AUDIO_PATH = PROJECT_ROOT / "data" / "test" / "temp_preview_librosa.mp3"


def download_preview(url: str) -> bool:
    """Skida 30-sekundni preview sa Spotify URL-a na lokalnu putanju."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            TEMP_AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(TEMP_AUDIO_PATH, "wb") as f:
                f.write(response.content)
            return True
    except Exception:
        pass
    return False


def extract_librosa_features(file_path: str):
    """Ekstrahuje napredne fizičke, harmonijske i spektralne karakteristike bezbedno."""
    features_dict = {}
    try:
        # Učitavamo 30 sekundi audio fajla
        y, sr = librosa.load(file_path, sr=22050, duration=30)
        if len(y) == 0:
            return None

        # 1. Osnovne fizičke karakteristike
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features_dict["librosa_tempo"] = float(np.atleast_1d(tempo)[0])
        except Exception:
            features_dict["librosa_tempo"] = np.nan

        try:
            features_dict["librosa_energy"] = float(np.mean(librosa.feature.rms(y=y)))
            features_dict["librosa_spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            features_dict["librosa_spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            features_dict["librosa_zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            # Dodatno: Spektralna širina i ravnost (odlično za model)
            features_dict["librosa_spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            features_dict["librosa_spectral_flatness"] = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        except Exception:
            pass

        # 2. MFCC koeficijenti (13 koeficijenata)
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features_dict[f"librosa_mfcc_{i + 1}"] = float(np.mean(mfccs[i]))
        except Exception:
            for i in range(13):
                features_dict[f"librosa_mfcc_{i + 1}"] = np.nan

        # 3. Chroma karakteristike (12 nota)
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            for i in range(12):
                features_dict[f"librosa_chroma_{i + 1}"] = float(np.mean(chroma[i]))
        except Exception:
            for i in range(12):
                features_dict[f"librosa_chroma_{i + 1}"] = np.nan

        return features_dict

    except Exception as e:
        print(f"Greška pri obradi fajla {file_path}: {e}")
        return None


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nije pronađen na putanji: {DATASET_PATH}")

    print(f"Učitavam dataset: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH).head(12000)

    # Generišemo listu svih kolona
    base_cols = [
        "librosa_tempo", "librosa_energy",
        "librosa_spectral_centroid", "librosa_spectral_rolloff", "librosa_zcr",
        "librosa_spectral_bandwidth", "librosa_spectral_flatness"
    ]
    mfcc_cols = [f"librosa_mfcc_{i + 1}" for i in range(13)]
    chroma_cols = [f"librosa_chroma_{i + 1}" for i in range(12)]

    librosa_cols = base_cols + mfcc_cols + chroma_cols

    # Inicijalizujemo kolone eksplicitno kao float64
    for col in librosa_cols:
        df[col] = pd.Series(dtype="float64")

    print(f"Ukupno pesama za Librosa obradu: {len(df)}")
    print(f"Ukupno audio kolona koje se dodaju: {len(librosa_cols)}")
    print("Pokrećem preuzimanje preview-a i naprednu ekstrakciju...")

    results = {col: [] for col in librosa_cols}

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        preview_url = row.get("preview_url", None)

        features = None
        if pd.notna(preview_url) and download_preview(preview_url):
            features = extract_librosa_features(str(TEMP_AUDIO_PATH.resolve()))

        # Upisujemo vrednosti za svaku kolonu
        for col in librosa_cols:
            val = features.get(col) if (features and col in features) else np.nan
            results[col].append(val)

    # Ubacujemo rezultate u dataframe
    for col in librosa_cols:
        df[col] = pd.Series(results[col], dtype="float64")

    OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_DATASET_PATH, index=False)

    if TEMP_AUDIO_PATH.exists():
        TEMP_AUDIO_PATH.unlink()

    print(f"\nUspešno završeno! Novi obogaćeni dataset je sačuvan na: {OUTPUT_DATASET_PATH}")
    print(f"Ukupno kolona u novom datasetu: {len(df.columns)}")

    # Provera da li su kolone tu
    sample_display_cols = ["name", "librosa_tempo", "librosa_mfcc_1", "librosa_mfcc_13", "librosa_chroma_1"]
    cols_to_show = [c for c in sample_display_cols if c in df.columns]
    print("\nProvera unetih vrednosti:")
    print(df[cols_to_show].head(5))
    print(df.columns.tolist())
    print(df[["name", "librosa_mfcc_1", "librosa_mfcc_13", "librosa_chroma_1"]].head(5))

if __name__ == "__main__":
    main()