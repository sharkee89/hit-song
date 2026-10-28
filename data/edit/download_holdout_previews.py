"""
Preuzima 30s Spotify preview fajlove za demo pesme iz holdout skupa
(pesme koje XGBoost model nikada nije video tokom treninga).

Pokretanje (iz korena projekta, nakon python -m neural.mlp.train_xgboost):
    python data/edit/download_holdout_previews.py
"""
import re
from pathlib import Path

import pandas as pd
import requests

try:
    # Koristi OS skladište sertifikata (npr. kada antivirus presreće HTTPS)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HOLDOUT_PATH = PROJECT_ROOT / "data" / "models" / "holdout_tracks.csv"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "downloads" / "holdout_preview_files"

# Broj pesama po opsegu stvarne popularnosti (hitovi, srednje, slabije)
SAMPLES = [(60, 101, 8), (40, 60, 6), (0, 40, 6)]


def _safe(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", str(text).strip())


def main():
    df = pd.read_csv(HOLDOUT_PATH).dropna(subset=["preview_url"])
    df = df.drop_duplicates("artist_name")

    buckets = [df[(df["popularity"] >= low) & (df["popularity"] < high)] for low, high, _ in SAMPLES]
    sample = pd.concat(
        bucket.sample(min(n, len(bucket)), random_state=42) for bucket, (_, _, n) in zip(buckets, SAMPLES)
    )

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    success = 0
    for row in sample.itertuples():
        file_path = DOWNLOAD_DIR / f"{_safe(row.artist_name)}-{_safe(row.name)}-{int(row.popularity)}.mp3"
        try:
            response = requests.get(row.preview_url, timeout=15)
            response.raise_for_status()
            file_path.write_bytes(response.content)
            success += 1
            print(f"OK  {file_path.name}")
        except Exception as e:
            print(f"ERR {row.artist_name} - {row.name}: {e}")

    print(f"\nPreuzeto {success}/{len(sample)} fajlova u: {DOWNLOAD_DIR}")


if __name__ == "__main__":
    main()
