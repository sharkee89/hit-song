import ast
import os
from pathlib import Path
import pandas as pd

# ==========================================
# 1. PRONALAŽENJE CSV FAJLOVA (Relativne putanje)
# ==========================================
CURRENT_DIR = Path(__file__).resolve().parent

TRACKS_CSV_PATH = (
    CURRENT_DIR.parent
    / "dataset"
    / "SpotGenTrack"
    / "Data_Sources"
    / "spotify_tracks_clean.csv"
)

ARTISTS_CSV_PATH = (
    CURRENT_DIR.parent
    / "dataset"
    / "SpotGenTrack"
    / "Data_Sources"
    / "spotify_artists.csv"
)

OUTPUT_PATH = CURRENT_DIR / "spotify_tracks_with_artists.csv"

print(f"Učitavam pesme sa: {TRACKS_CSV_PATH}")
print(f"Učitavam izvođače sa: {ARTISTS_CSV_PATH}")

if not TRACKS_CSV_PATH.exists():
    raise FileNotFoundError(f"GREŠKA: Nije pronađen tracks CSV na: {TRACKS_CSV_PATH}")

if not ARTISTS_CSV_PATH.exists():
    raise FileNotFoundError(f"GREŠKA: Nije pronađen artists CSV na: {ARTISTS_CSV_PATH}")

# ==========================================
# 2. UČITAVANJE DATASET-OVA
# ==========================================
df_tracks = pd.read_csv(TRACKS_CSV_PATH)
df_artists = pd.read_csv(ARTISTS_CSV_PATH)

print(f"Učitano pesama: {len(df_tracks)}")
print(f"Učitano izvođača: {len(df_artists)}")

# Priprema kolona izvođača za spajanje (preimenovanje radi izbegavanja kolizije imena)
artists_clean = df_artists[
    ["id", "name", "artist_popularity", "followers", "genres"]
].copy()

artists_clean.rename(
    columns={
        "id": "artist_id",
        "name": "artist_name",
        "genres": "artist_genres",
    },
    inplace=True,
)

# Ukoliko ima duplikata izvođača u spotify_artists.csv, zadržavamo jedinstvene po artist_id
artists_clean.drop_duplicates(subset=["artist_id"], inplace=True)

# ==========================================
# 3. PARSIRANJE 'artists_id' LISTE I EXPLODE
# ==========================================
def parse_artist_ids(val):
    """Sigurno pretvara string listu "['id1', 'id2']" u pravu Python listu."""
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []

print("\nParsiranje artist_id listi i ekspandovanje pesama...")
df_tracks["artist_id_list"] = df_tracks["artists_id"].apply(parse_artist_ids)

# Svaki izvođač iz liste dobija sopstveni red za istu pesmu
df_expanded = df_tracks.explode("artist_id_list").rename(
    columns={"artist_id_list": "artist_id"}
)

# ==========================================
# 4. SPAJANJE (MERGE) PESAMA I IZVOĐAČA
# ==========================================
print("Spajanje pesama sa podacima o izvođačima...")
df_final = pd.merge(
    df_expanded,
    artists_clean,
    on="artist_id",
    how="left",
)

# ==========================================
# 5. ČUVANJE NOVOG DATASETA
# ==========================================
df_final.to_csv(OUTPUT_PATH, index=False)

print(f"\n Obrada uspešno završena!")
print(f" Ukupan broj redova u novom datasetu: {len(df_final)}")
print(f" Sačuvano na: {OUTPUT_PATH}")