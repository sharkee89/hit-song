import ast
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. PRONALAŽENJE FAJLOVA I PUTANJA
# ==========================================
CURRENT_DIR = Path(__file__).resolve().parent

TRACKS_RAW_PATH = (
    CURRENT_DIR.parent
    / "dataset"
    / "SpotGenTrack"
    / "Data_Sources"
    / "spotify_tracks.csv"
)

ARTISTS_CSV_PATH = (
    CURRENT_DIR.parent
    / "dataset"
    / "SpotGenTrack"
    / "Data_Sources"
    / "spotify_artists.csv"
)

OUTPUT_PATH = CURRENT_DIR / "spotify_tracks_with_artists.parquet"

print(f"Tražim sirove pesme na: {TRACKS_RAW_PATH}")
print(f"Učitavam izvođače sa: {ARTISTS_CSV_PATH}")

if not TRACKS_RAW_PATH.exists():
    raise FileNotFoundError(f"GREŠKA: Nije pronađen tracks CSV na: {TRACKS_RAW_PATH}")

if not ARTISTS_CSV_PATH.exists():
    raise FileNotFoundError(f"GREŠKA: Nije pronađen artists CSV na: {ARTISTS_CSV_PATH}")

# ==========================================
# 2. UČITAVANJE I ČIŠĆENJE PESAMA (ML filtriranje)
# ==========================================
df = pd.read_csv(TRACKS_RAW_PATH)
print(f"Učitano sirovih traka: {len(df)}")

# Zadržavamo samo redove sa validnim preview_url-om
df_valid = df[
    df["preview_url"].notna() & df["preview_url"].str.startswith("http")
].copy()

feature_cols = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

X = df_valid[feature_cols].dropna()
df_valid = df_valid.loc[X.index]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n1. Pokretanje Isolation Forest-a (uklanjanje ekstrema)...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df_valid["is_anomaly"] = iso_forest.fit_predict(X_scaled)

df_normal = df_valid[df_valid["is_anomaly"] == 1].copy()
X_normal_scaled = scaler.transform(df_normal[feature_cols])

print("2. Pokretanje K-Means klasterovanja...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_normal["cluster"] = kmeans.fit_predict(X_normal_scaled)

cluster_stats = df_normal.groupby("cluster")[
    ["speechiness", "danceability", "energy", "acousticness"]
].mean()

bad_cluster = cluster_stats["speechiness"].idxmax()
print("\nStatistika po klasterima:")
print(cluster_stats)
print(f"\n Identifikovan ne-muzički klaster za izbacivanje: Klaster #{bad_cluster}")

# Dobijamo očišćen dataframe pesama
df_clean_tracks = df_normal[df_normal["cluster"] != bad_cluster].copy()
df_clean_tracks = df_clean_tracks.drop(columns=["is_anomaly", "cluster"])

# Generisanje 2D grafika klastera
print("\n3. Generisanje 2D grafika klastera...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normal_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df_normal["cluster"],
    cmap="tab10",
    alpha=0.5,
    s=10,
)
plt.colorbar(scatter, label="Klaster ID")
plt.title(f"PCA Klasteri (Klaster #{bad_cluster} označava govor/priče/anomalije)")
plt.xlabel("PCA Komponenta 1")
plt.ylabel("PCA Komponenta 2")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(CURRENT_DIR / "clusters_visualization.png", dpi=300)
print(f" Grafik sačuvan kao: {CURRENT_DIR / 'clusters_visualization.png'}")

print(f"\n Čišćenje završeno! Zadržano čisto muzičkih pesama: {len(df_clean_tracks)} od {len(df)}")

# ==========================================
# 3. UČITAVANJE I PRIPREMA IZVOĐAČA
# ==========================================
df_artists = pd.read_csv(ARTISTS_CSV_PATH)
print(f"Učitano izvođača: {len(df_artists)}")

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

artists_clean.drop_duplicates(subset=["artist_id"], inplace=True)

# ==========================================
# 4. PARSIRANJE 'artists_id' I EXPLODE
# ==========================================
def parse_artist_ids(val):
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
df_clean_tracks["artist_id_list"] = df_clean_tracks["artists_id"].apply(parse_artist_ids)

df_expanded = df_clean_tracks.explode("artist_id_list").rename(
    columns={"artist_id_list": "artist_id"}
)

# ==========================================
# 5. SPAJANJE (MERGE) OČIŠĆENIH PESAMA I IZVOĐAČA
# ==========================================
print("Spajanje očišćenih pesama sa podacima o izvođačima...")
df_final = pd.merge(
    df_expanded,
    artists_clean,
    on="artist_id",
    how="left",
)

# ==========================================
# 6. ČUVANJE KONAČNOG DATASETA U PARQUET FORMATU
# ==========================================
df_final.to_parquet(OUTPUT_PATH, index=False)

print(f"\n Obrada i spajanje uspešno završeni!")
print(f" Ukupan broj redova u novom datasetu: {len(df_final)}")
print(f" Sačuvano na: {OUTPUT_PATH}")