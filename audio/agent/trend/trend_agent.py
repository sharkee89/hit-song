import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors


# Ensure root path resolution
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
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

EMBEDDINGS_DIR = (
    PROJECT_ROOT_DIR
    / "data"
    / "processed"
    / "trend_audio_embeddings"
)

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


MAX_WORKERS = 8
LIMIT_TRACKS = None


# Initialize single global AudioAgent instance
SHARED_AGENT = AudioAgent(device="cpu")


def process_single_track(track_info: tuple) -> bool:
    track_id, url = track_info

    save_path = EMBEDDINGS_DIR / f"{track_id}.npy"

    if save_path.exists():
        return True

    try:
        response = requests.get(url, timeout=12)

        if response.status_code != 200:
            return False

        audio_bytes = io.BytesIO(response.content)

        embedding = SHARED_AGENT.extract_features(audio_bytes)

        np.save(
            save_path,
            np.array(embedding, dtype=np.float32),
        )

        return True

    except Exception:
        return False


def extract_and_cache_embeddings(df_filtered: pd.DataFrame):
    print(
        f"Total trend tracks queued for feature extraction: "
        f"{len(df_filtered)}"
    )

    tasks = [
        (
            str(row.get("id", f"track_{idx}")),
            row["preview_url"],
        )
        for idx, row in df_filtered.iterrows()
        if (
            pd.notna(row.get("preview_url"))
            and str(row["preview_url"]).startswith("http")
        )
    ]

    successful = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_single_track, task)
            for task in tasks
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Extracting Trend Embeddings via AudioAgent",
        ):
            if future.result():
                successful += 1

    print(
        "\nEmbedding extraction complete! "
        f"Successfully processed: {successful} tracks."
    )


def train_and_predict_trend(
    df_filtered: pd.DataFrame,
    target_audio_path: str,
) -> dict:
    print("\nLoading cached embeddings for modeling...")

    X_list = []
    y_list = []
    valid_indices = []

    df_reset = df_filtered.reset_index(drop=True)

    for idx, row in df_reset.iterrows():
        track_id = str(row.get("id", f"track_{idx}"))

        emb_path = EMBEDDINGS_DIR / f"{track_id}.npy"

        if emb_path.exists():
            try:
                emb = np.load(emb_path)

                if emb.shape == (2048,):
                    X_list.append(emb)
                    y_list.append(float(row["trend_alignment"]))
                    valid_indices.append(idx)

            except Exception:
                continue

    if not X_list:
        print("❌ No valid embedding vectors found for training.")
        return {}

    X = np.array(X_list)
    y = np.array(y_list)

    print(
        f"Dataset matrix shape for ML model: "
        f"X={X.shape}, y={y.shape}"
    )

    # 1. KNN / Distance lookup
    print("Fitting Nearest Neighbors index...")

    nn_model = NearestNeighbors(
        n_neighbors=1,
        metric="cosine",
    )

    nn_model.fit(X)

    # 2. Random Forest Regressor
    print("Training Random Forest Regressor...")

    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(X, y)

    print(
        "\nProcessing target input audio via AudioAgent "
        f"sliding window from: {target_audio_path}"
    )

    if not os.path.exists(target_audio_path):
        print(
            f"❌ Target audio file not found at: "
            f"{target_audio_path}"
        )
        return {}

    target_embeddings = SHARED_AGENT.extract_sliding_features(
        target_audio_path
    )

    best_distance = float("inf")
    best_closest_idx = None
    best_predicted_trend = None

    SIMILARITY_THRESHOLD = 0.15

    for target_emb in target_embeddings:
        target_x = np.array(
            target_emb,
            dtype=np.float32,
        ).reshape(1, -1)

        distances, indices = nn_model.kneighbors(target_x)

        min_dist = distances[0][0]

        if min_dist < best_distance:
            best_distance = min_dist
            best_closest_idx = indices[0][0]
            best_predicted_trend = float(
                rf_model.predict(target_x)[0]
            )

    if best_distance <= SIMILARITY_THRESHOLD:
        matched_row = df_reset.iloc[best_closest_idx]

        matched_trend = float(
            matched_row["trend_alignment"]
        )

        matched_name = matched_row.get(
            "name",
            "Unknown Track",
        )

        matched_artist = matched_row.get(
            "artist_name",
            "Unknown Artist",
        )

        final_trend = matched_trend

        print(
            "\n🎯 Exact or highly similar segment found "
            "in trend dataset!"
        )

        print(
            f"Matched Track: "
            f"{matched_artist} - {matched_name}"
        )

        print(
            f"Best Segment Cosine Distance: "
            f"{best_distance:.4f}"
        )

        print(
            f"Assigned trend_alignment: "
            f"{final_trend}"
        )

    else:
        final_trend = (
            max(0.5, min(1.0, best_predicted_trend))
            if best_predicted_trend >= 0.5
            else max(0.0, best_predicted_trend)
        )

        print(
            "\n🤖 No direct close neighbor found across "
            "sliding windows. Estimated via "
            "Random Forest Regressor:"
        )

        print(
            f"Best Nearest Cosine Distance: "
            f"{best_distance:.4f} "
            "(Above threshold)"
        )

        print(
            f"Predicted trend_alignment: "
            f"{final_trend:.4f}"
        )

    # Kreiranje kompatibilnog trend feature tenzora
    # za MLP fuziju:
    # [trend_alignment, cosine_distance]
    trend_feature_tensor = torch.tensor(
        [
            float(final_trend),
            float(best_distance),
        ],
        dtype=torch.float32,
    )

    return {
        "trend_alignment": float(final_trend),
        "cosine_distance": float(best_distance),
        "trend_feature_tensor": trend_feature_tensor,
    }


def main():
    load_dotenv()

    target_audio_path = os.getenv(
        "AUDIO_FILE_PATH",
        "",
    )

    if not PARQUET_PATH.exists():
        print(
            f"ERROR: File not found at {PARQUET_PATH}"
        )
        return

    print("Loading aligned parquet dataset...")

    df = pd.read_parquet(PARQUET_PATH)

    df_filtered = df[
        df["trend_alignment"] >= 0.5
    ].copy()

    print(
        "Filtered tracks with trend_alignment >= 0.5: "
        f"{len(df_filtered)}"
    )

    extract_and_cache_embeddings(df_filtered)

    if target_audio_path:
        trend_outputs = train_and_predict_trend(
            df_filtered,
            target_audio_path,
        )

        print(
            "\n✅ Trend analysis outputs generated successfully:"
        )

        print(trend_outputs)

    else:
        print(
            "❌ AUDIO_FILE_PATH isn't defined in .env "
            "for target inference."
        )


if __name__ == "__main__":
    main()