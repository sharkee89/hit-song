import os
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
from audiobox_aesthetics.infer import initialize_predictor

# Forsiramo CUDA da bude vidljiva procesu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "spotify_tracks_all.parquet"
OUTPUT_DATASET_PATH = PROJECT_ROOT / "dataset" / "spotify_tracks_audiobox.parquet"
TEMP_AUDIO_PATH = PROJECT_ROOT / "test" / "temp_preview.mp3"


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


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nije pronađen na putanji: {DATASET_PATH}")

    print(f"Učitavam dataset: {DATASET_PATH}")
    # Testiramo na prvih 100 pesama
    df = pd.read_parquet(DATASET_PATH)

    target_columns = [
        "id", "name", "popularity", "preview_url",
        "artist_id", "artist_name", "artist_popularity",
        "followers", "artist_genres", "trend_alignment"
    ]

    existing_cols = [col for col in target_columns if col in df.columns]
    df_subset = df[existing_cols].copy()

    # Inicijalizujemo kolone eksplicitno kao float (prazne vrednosti će biti NaN, ali tip je sa decimalama)
    for col in ["PQ", "PC", "CE", "CU"]:
        df_subset[col] = pd.Series(dtype="float64")

    print("Inicijalizujem Audiobox model direktno u memoriji (CUDA)...")
    predictor = initialize_predictor()

    print(f"Ukupno pesama za obradu: {len(df_subset)}")
    print("Pokrećem preuzimanje i direktnu Python analizu...")

    results_pq = []
    results_pc = []
    results_ce = []
    results_cu = []

    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
        preview_url = row.get("preview_url", None)

        pq, pc, ce, cu = None, None, None, None

        if pd.notna(preview_url) and download_preview(preview_url):
            try:
                preds = predictor.forward([{"path": str(TEMP_AUDIO_PATH.resolve())}])
                if preds and len(preds) > 0:
                    res = preds[0]
                    # Osiguravamo da su vrednosti eksplicitno float
                    pq = float(res.get("PQ")) if res.get("PQ") is not None else None
                    pc = float(res.get("PC")) if res.get("PC") is not None else None
                    ce = float(res.get("CE")) if res.get("CE") is not None else None
                    cu = float(res.get("CU")) if res.get("CU") is not None else None
            except Exception:
                pass

        results_pq.append(pq)
        results_pc.append(pc)
        results_ce.append(ce)
        results_cu.append(cu)

    # Upisujemo rezultate u dataframe sa očuvanim decimalama
    df_subset["PQ"] = pd.Series(results_pq, dtype="float64")
    df_subset["PC"] = pd.Series(results_pc, dtype="float64")
    df_subset["CE"] = pd.Series(results_ce, dtype="float64")
    df_subset["CU"] = pd.Series(results_cu, dtype="float64")

    OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_subset.to_parquet(OUTPUT_DATASET_PATH, index=False)

    if TEMP_AUDIO_PATH.exists():
        TEMP_AUDIO_PATH.unlink()

    print(f"\nUspešno završeno! Sačuvano na: {OUTPUT_DATASET_PATH}")

    print("\nProvera unetih vrednosti (prvih 5 pesama):")
    print(df_subset[["name", "PQ", "PC", "CE", "CU"]].head(5))


if __name__ == "__main__":
    main()