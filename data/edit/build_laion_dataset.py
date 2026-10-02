import os
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import torch
import laion_clap

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# ... tvoji ostali importi idu ovde ...
import os
from pathlib import Path
import pandas as pd

# Forsiramo CUDA ukoliko je dostupna
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Izvorni fajl je sada onaj koji već ima Librosa podatke
DATASET_PATH = PROJECT_ROOT / "data" / "dataset" / "spotify_tracks_audiobox_librosa.parquet"
# Novi izlazni fajl sa dodatim CLAP podacima
OUTPUT_DATASET_PATH = PROJECT_ROOT / "data" / "dataset" / "spotify_tracks_audiobox_librosa_clap.parquet"
TEMP_AUDIO_PATH = PROJECT_ROOT / "data" / "test" / "temp_preview_clap.mp3"


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


def extract_clap_embedding(file_path: str, clap_model):
    """Ekstrahuje 512-dimenzionalni LAION-CLAP audio embedding."""
    try:
        # Uklonjen argument use_cuda
        audio_embed = clap_model.get_audio_embedding_from_filelist(x=[file_path])
        return audio_embed[0].astype(np.float32).tolist()
    except Exception as e:
        print(f"Greška pri CLAP ekstrakciji za {file_path}: {e}")
        return None


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Izvorni dataset sa Librosa podacima nije pronađen na putanji: {DATASET_PATH}")

    print(f"Učitavam postojeći dataset sa Librosa podacima: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)

    # Inicijalizacija LAION-CLAP modela
    print("Učitavam LAION-CLAP model...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False)
    clap_model.load_ckpt()

    # Dodajemo kolonu za CLAP embedding ako već ne postoji
    df["clap_embedding"] = None

    print(f"Ukupno pesama u datasetu: {len(df)}")
    print("Pokrećem preuzimanje preview-a i ekstrakciju SAMO CLAP podataka...")

    clap_results = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        preview_url = row.get("preview_url", None)
        clap_emb = None

        if pd.notna(preview_url) and download_preview(preview_url):
            temp_path_str = str(TEMP_AUDIO_PATH.resolve())
            clap_emb = extract_clap_embedding(temp_path_str, clap_model)

        clap_results.append(clap_emb if clap_emb is not None else [np.nan] * 512)

    # Ubacujemo CLAP rezultate u dataframe
    df["clap_embedding"] = clap_results

    OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_DATASET_PATH, index=False)

    if TEMP_AUDIO_PATH.exists():
        TEMP_AUDIO_PATH.unlink()

    print(f"\nUspešno završeno! Kompletan dataset (Librosa + CLAP) sačuvan na: {OUTPUT_DATASET_PATH}")
    print(f"Ukupno kolona u novom datasetu: {len(df.columns)}")

    # Provera rezultata
    sample_cols = [c for c in ["name", "librosa_tempo", "clap_embedding"] if c in df.columns]
    print("\nProvera unetih vrednosti:")
    print(df[sample_cols].head(3))


if __name__ == "__main__":
    main()