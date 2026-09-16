import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm

# Ensure root path resolution
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audio.agent.audio_agent import AudioAgent

# Prevent CPU thread thrashing across parallel workers
torch.set_num_threads(1)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent.parent.parent

PARQUET_PATH = (
    PROJECT_ROOT_DIR
    / "data"
    / "dataset"
    / "spotify_tracks_with_artists_aligned.parquet"
)
OUTPUT_DIR = PROJECT_ROOT_DIR / "data" / "processed" / "audio_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_WORKERS = 8
LIMIT_TRACKS = None

# Initialize single global AudioAgent instance
SHARED_AGENT = AudioAgent(device="cpu")


def process_single_track(track_info: tuple) -> bool:
    track_id, url = track_info
    save_path = OUTPUT_DIR / f"{track_id}.npy"

    if save_path.exists():
        return True

    try:
        response = requests.get(url, timeout=12)
        if response.status_code != 200:
            return False

        # Convert network stream into BytesIO buffer
        audio_bytes = io.BytesIO(response.content)

        # Call the unified AudioAgent method
        embedding = SHARED_AGENT.extract_features(audio_bytes)

        # Save extracted feature vector as numpy binary array
        np.save(save_path, np.array(embedding, dtype=np.float32))
        return True

    except Exception:
        return False


def main():
    if not PARQUET_PATH.exists():
        print(f"ERROR: File not found at {PARQUET_PATH}")
        return

    print("Loading aligned parquet dataset metadata...")
    df = pd.read_parquet(PARQUET_PATH)

    df_valid = df[
        df["preview_url"].notna() & df["preview_url"].str.startswith("http")
    ].copy()

    if LIMIT_TRACKS is not None:
        df_valid = df_valid.head(LIMIT_TRACKS)

    print(f"Total tracks queued for processing: {len(df_valid)}")

    tasks = [
        (str(row.get("id", f"track_{idx}")), row["preview_url"])
        for idx, row in df_valid.iterrows()
    ]

    successful = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_track, task) for task in tasks]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Batch Processing via AudioAgent",
        ):
            if future.result():
                successful += 1

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} tracks.")
    print(f"Embeddings saved at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()