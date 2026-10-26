from pathlib import Path

import numpy as np
import pandas as pd

# Upišite naziv pesme. Ako ostane prazno ili pesma ne postoji,
# skripta predlaže 15 poznatih pesama srednje popularnosti i uzima prvu.
SONG = "Dive"
ARTIST = "Ed Sheeran"   # ostavite prazno ako izvođač nije bitan

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset" / "spotify_tracks_audiobox_librosa.parquet"

pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

df = pd.read_parquet(DATA_PATH)

name_col = "name"
if SONG:
    names = df[name_col].astype(str).str.strip().str.lower()
    match = df[names == SONG.strip().lower()]
    if match.empty:
        match = df[names.str.contains(SONG.strip().lower(), regex=False, na=False)]
    if ARTIST:
        match = match[match["artist_name"].astype(str).str.contains(ARTIST, case=False, regex=False, na=False)]
else:
    match = df.iloc[0:0]

if match.empty:
    if SONG:
        print(f"'{SONG}' nije pronađena.")
    print("Predlozi (poznati izvođači, popularnost 50–75):\n")
    match = (
        df[df["popularity"].between(50, 75)]
        .sort_values("artist_popularity", ascending=False)
        .drop_duplicates("artist_name")
        .head(15)
    )

print(match[["name", "artist_name", "popularity", "artist_popularity"]].to_string(index=False))

row = match.iloc[0]

groups = {
    "Production agent": ["PQ", "PC", "CE", "CU"],
    "Talent agent": ["artist_popularity", "followers"],
    "Audio agent": [c for c in df.columns if c.startswith("librosa_")],
}

out = []
for agent, cols in groups.items():
    for c in cols:
        v = row[c]
        out.append({"agent": agent, "feature": c, "value": v})
        if c == "followers":
            out.append({
                "agent": agent,
                "feature": "log1p(followers)  [model input]",
                "value": np.log1p(v),
            })

table = pd.DataFrame(out)

print(f"\nPesma: {row['name']} - {row['artist_name']} | popularity (target): {row['popularity']}")
print(f"Broj feature-a: {sum(len(c) for c in groups.values())}\n")
print(table.to_string(index=False))

out_path = Path(__file__).resolve().parent / "appendix_features.csv"
table.to_csv(out_path, index=False)
print(f"\nSačuvano: {out_path}")