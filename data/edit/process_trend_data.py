import pandas as pd
from pathlib import Path

# Definiši putanje
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent.parent  # Prilagodi po potrebi
PARQUET_PATH = PROJECT_ROOT_DIR / "data" / "dataset" / "spotify_tracks_with_talent_enriched.parquet"


def finalize_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Finalizacija Trend karakteristika (2 dimenzije)...")

    # Proveri da li trend_alignment postoji
    if 'trend_alignment' not in df.columns:
        raise ValueError("Kolona 'trend_alignment' nije pronađena u datasetu!")

    # 1. Osiguraj da su vrednosti u trend_alignment u opsegu 0.0 - 1.0 (popuni nule gde fali)
    df['trend_alignment'] = df['trend_alignment'].fillna(0.0)

    # 2. Kreiraj drugu dimenziju: binarni indikator prisustva u trendu (1.0 ako je > 0, inače 0.0)
    df['trend_is_active'] = (df['trend_alignment'] > 0.0).astype(float)

    return df


def main():
    if not PARQUET_PATH.exists():
        print(f"Fajl nije pronađen na putanji: {PARQUET_PATH}")
        return

    df = pd.read_parquet(PARQUET_PATH)
    df = finalize_trend_features(df)

    # Snimi finalni master dataset za treniranje
    output_parquet = PROJECT_ROOT_DIR / "data" / "dataset" / "spotify_tracks_all.parquet"
    df.to_parquet(output_parquet)
    print(f"Successfully saved all dataset with complete trtend data: {output_parquet}")


if __name__ == "__main__":
    main()