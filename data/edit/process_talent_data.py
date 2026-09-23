import pandas as pd
import numpy as np
from pathlib import Path

# Definiši putanje
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent.parent
PARQUET_PATH = PROJECT_ROOT_DIR / "data" / "dataset" / "spotify_tracks_with_artists_aligned.parquet"


def prepare_talent_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Obrada Talent karakteristika za izvođače...")

    # 1. Normalizacija popularnosti (0-100 -> 0.0-1.0)
    df['talent_norm_popularity'] = df['artist_popularity'].fillna(0) / 100.0

    # 2. Logaritamska skala za pratioce (da se izbegnu gigantski brojevi)
    # Pretpostavka da je 'followers' u milionima ili apsolutnom broju; log1p lepo pegla ekstreme
    df['talent_log_followers'] = np.log1p(df['followers'].fillna(0))
    # Skaliranje log pratilaca na otprilike 0-1 opseg (uz pretpostavku da max followers retko prelazi 100M / log(1e8) ≈ 18.4)
    df['talent_norm_followers'] = df['talent_log_followers'] / 18.4
    df['talent_norm_followers'] = df['talent_norm_followers'].clip(0.0, 1.0)

    # 3. Relativni koeficijent (popularnost u odnosu na log pratilaca - pokazuje "hype" faktor)
    df['talent_hype_ratio'] = (df['talent_norm_popularity'] / (df['talent_norm_followers'] + 1e-5)).clip(0.0, 2.0) / 2.0

    # 4. Binarni ili indikator prisustva validnog izvođača
    df['talent_has_artist_data'] = df['artist_id'].notna().astype(float)

    return df


def main():
    if not PARQUET_PATH.exists():
        print(f"Fajl nije pronađen na putanji: {PARQUET_PATH}")
        return

    df = pd.read_parquet(PARQUET_PATH)
    df = prepare_talent_features(df)

    # Možemo da sačuvamo obogaćeni dataframe ili da ga prosledimo dalje
    output_parquet = PROJECT_ROOT_DIR / "data" / "dataset" / "spotify_tracks_with_talent_enriched.parquet"
    df.to_parquet(output_parquet)
    print(f"Uspešno sačuvan obogaćeni dataset sa talent karakteristikama na: {output_parquet}")


if __name__ == "__main__":
    main()