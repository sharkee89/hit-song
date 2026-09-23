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
# 1. PRONALAŽENJE CSV FAJLA IZ DRUGOG FOLDERA
# ==========================================
CURRENT_DIR = Path(__file__).resolve().parent

# Izlazimo iz trenutnog foldera (parent) i ulazimo u ciljanu strukturu putanje
CSV_PATH = (
    CURRENT_DIR.parent
    / "dataset"
    / "SpotGenTrack"
    / "Data_Sources"
    / "spotify_tracks.csv"
)
OUTPUT_PATH = CURRENT_DIR / "spotify_tracks_clean.csv"

print(f"Tražim CSV na putanji: {CSV_PATH}")

if not CSV_PATH.exists():
    raise FileNotFoundError(f"GREŠKA: Fajl nije pronađen na lokaciji: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"Učitano traka: {len(df)}")

# Zadržavamo samo redove sa validnim preview_url-om
df_valid = df[
    df["preview_url"].notna() & df["preview_url"].str.startswith("http")
].copy()

# ==========================================
# 2. IZDVAJANJE I SKALIRANJE OSOBINA
# ==========================================
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

# ==========================================
# 3. ISOLATION FOREST (Uklanjanje ekstrema)
# ==========================================
print("\n1. Pokretanje Isolation Forest-a...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df_valid["is_anomaly"] = iso_forest.fit_predict(X_scaled)

df_normal = df_valid[df_valid["is_anomaly"] == 1].copy()
X_normal_scaled = scaler.transform(df_normal[feature_cols])

# ==========================================
# 4. K-MEANS KLASTEROVANJE
# ==========================================
print("2. Pokretanje K-Means klasterovanja...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_normal["cluster"] = kmeans.fit_predict(X_normal_scaled)

# Pronalaženje ne-muzičkog klastera (najveći speechiness / najniži danceability)
cluster_stats = df_normal.groupby("cluster")[
    ["speechiness", "danceability", "energy", "acousticness"]
].mean()

bad_cluster = cluster_stats["speechiness"].idxmax()
print("\nStatistika po klasterima:")
print(cluster_stats)
print(f"\n Identifikovan ne-muzički klaster za izbacivanje: Klaster #{bad_cluster}")

# Odvajanje čistih muzičkih pesama
df_final = df_normal[df_normal["cluster"] != bad_cluster].copy()

# ==========================================
# 5. VIZUELIZACIJA (PCA 2D Grafik)
# ==========================================
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

# ==========================================
# 6. ČUVANJE FINALNOG CSV-A
# ==========================================
df_final = df_final.drop(columns=["is_anomaly", "cluster"])
df_final.to_csv(OUTPUT_PATH, index=False)

print(f"\n Čišćenje završeno!")
print(f" Zadržano čisto muzičkih pesama: {len(df_final)} od {len(df)}")
print(f" Sačuvano na: {OUTPUT_PATH}")