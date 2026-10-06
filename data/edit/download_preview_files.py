import os
from pathlib import Path
import pandas as pd
import requests

# Putanje bazirane na strukturi projekta
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Izlazimo iz trenutnog foldera i ulazimo u dataset folder
DATASET_PATH = PROJECT_ROOT / "data" / "dataset" / "spotify_tracks_audiobox_librosa.parquet"
# Download folder van trenutnog radnog direktorijuma ili prilagođen po želji
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "downloads" / "preview_files"


def download_preview_files():
    print("Učitavam dataset...")
    if not DATASET_PATH.exists():
        print(f"Greška: Dataset nije pronađen na putanji {DATASET_PATH}")
        return

    # Učitavanje Parquet fajla
    df = pd.read_parquet(DATASET_PATH)

    # Provera neophodnih kolona
    required_cols = ["artist_name", "name", "popularity", "preview_url"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Greška: Dataset ne sadrži obaveznu kolonu '{col}'.")
            return

    # Čistimo redove gde nema preview_url-a ili imena izvođača
    df = df.dropna(subset=["preview_url", "artist_name", "popularity"])

    # Osiguravamo se da uzimamo RAZLIČITE izvođače (drop duplicates po artist_name)
    df_unique_artists = df.drop_duplicates(subset=["artist_name"]).copy()

    # Sortiramo po popularnosti opadajuće
    df_sorted = df_unique_artists.sort_values(by="popularity", ascending=False).reset_index(drop=True)

    total_artists = len(df_sorted)
    if total_artists < 51:
        print(f"Upozorenje: Dataset ima samo {total_artists} jedinstvenih izvođača. Uzorkovaću srazmerno.")

    # Delimo na 3 grupe po 17 (ili koliko je dostupno)
    third = 17

    # 1. Visoka popularnost (Top)
    high_pop = df_sorted.head(third)

    # 2. Niska popularnost (Bottom)
    low_pop = df_sorted.tail(third)

    # 3. Srednja popularnost (Middle)
    mid_start = max(0, (total_artists // 2) - (third // 2))
    mid_pop = df_sorted.iloc[mid_start : mid_start + third]

    # Spajamo u konačan skup od 51 pesme (po 17 iz svake kategorije)
    sample_df = pd.concat([high_pop, mid_pop, low_pop]).reset_index(drop=True)

    print(f"Izabrano je {len(sample_df)} pesama od različitih izvođača za preuzimanje.")

    # Kreiramo download folder ako ne postoji
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for idx, row in sample_df.iterrows():
        artist = str(row["artist_name"]).strip().replace("/", "_").replace("\\", "_")
        name = str(row["name"]).strip().replace("/", "_").replace("\\", "_")
        popularity = int(row["popularity"])
        preview_url = row["preview_url"]

        # Format naziva fajla: artist_name-name-popularity.mp3
        file_name = f"{artist}-{name}-{popularity}.mp3"
        file_path = DOWNLOAD_DIR / file_name

        print(f"[{idx+1}/{len(sample_df)}] Preuzimam: {artist} - {name} (Pop: {popularity})...")

        try:
            response = requests.get(preview_url, timeout=15)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                success_count += 1
            else:
                print(f"  -> Greška pri skidanju (Status kod: {response.status_code})")
        except Exception as e:
            print(f"  -> Izuzetak prilikom preuzimanja: {e}")

    print("\n" + "=" * 40)
    print(f"Preuzimanje završeno! Uspešno skinuto: {success_count}/{len(sample_df)} fajlova.")
    print(f"Fajl je sačuvani u folderu: {DOWNLOAD_DIR}")
    print("=" * 40)


if __name__ == "__main__":
    download_preview_files()