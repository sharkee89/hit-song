from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = (
    PROJECT_ROOT / "data" / "dataset" / "spotify_tracks_audiobox_librosa.parquet"
)


def main():
  if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset nije pronađen na putanji: {DATASET_PATH}")

  print(f"Učitavam dataset za analiziranje parova: {DATASET_PATH}")
  df = pd.read_parquet(DATASET_PATH)

  # 1. Definišemo features (PQ, PC, CE, CU + sve librosa kolone)
  metadata_features = ["PQ", "PC", "CE", "CU"]
  librosa_features = [col for col in df.columns if col.startswith("librosa_")]
  feature_cols = metadata_features + librosa_features

  # Proveravamo koje kolone stvarno postoje u datasetu
  valid_features = [col for col in feature_cols if col in df.columns]
  print(f"Broj karakteristika korišćenih za poređenje: {len(valid_features)}")

  # 2. Čistimo dataset od redova koji imaju NaN u ovim kolonama
  cols_needed = valid_features + ["name", "artist_name", "popularity", "id"]
  df_clean = df.dropna(subset=cols_needed).copy()
  print(f"Broj validnih pesama za pretragu: {len(df_clean)}")

  if len(df_clean) < 10:
    print("Nema dovoljno validnih pesama u datasetu.")
    return

  # 3. Izdvajamo vrednosti i skaliramo ih (da svaka karakteristika ima istu težinu)
  X = df_clean[valid_features].values.astype(np.float32)
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # 4. Koristimo NearestNeighbors da nađemo najbliže susede u prostoru audio obeležja
  # Tražimo 5 suseda (prvi je uvek sama pesma sa distancom 0)
  print(
      "Računam najsličnije pesme po audio karakteristikama i metapodacima..."
  )
  nn = NearestNeighbors(n_neighbors=5, algorithm="auto", metric="euclidean")
  nn.fit(X_scaled)
  distances, indices = nn.kneighbors(X_scaled)

  found_pairs = []
  seen_indices = set()

  # 5. Pretražujemo parove koji imaju visoku razliku u popularnosti
  for i in range(len(df_clean)):
    if i in seen_indices:
      continue

    current_song = df_clean.iloc[i]

    # Gledamo susede ove pesme (krećemo od indexa 1 jer je 0 ona sama)
    for neighbor_rank in range(1, len(indices[i])):
      j = indices[i][neighbor_rank]

      if j in seen_indices or i == j:
        continue

      neighbor_song = df_clean.iloc[j]

      pop_diff = abs(current_song["popularity"] - neighbor_song["popularity"])

      # Uslov za interesantan par: da im je razlika u popularnosti velika (npr. >= 40 poena)
      if pop_diff >= 40:
        found_pairs.append({
            "song_1": current_song["name"],
            "artist_1": current_song["artist_name"],
            "pop_1": current_song["popularity"],
            "song_2": neighbor_song["name"],
            "artist_2": neighbor_song["artist_name"],
            "pop_2": neighbor_song["popularity"],
            "pop_diff": pop_diff,
            "distance": distances[i][neighbor_rank],
        })

        seen_indices.add(i)
        seen_indices.add(j)
        break

    # Trebaju nam tačno 3 para
    if len(found_pairs) >= 3:
      break

  # 6. Prikaz i logovanje rezultata
  print("\n" + "=" * 70)
  print(" REZULTATI: 3 PARA PESAMA SA SLIČNIM AUDIO PROFILOM, A RAZLIČITOM POPULARNOŠĆU")
  print("=" * 70)

  if not found_pairs:
     print("Nisu pronađeni parovi sa zadatim kriterijumom razlike u popularnosti. Pokušaj da smanjiš prag razlike.")
  else:
      for idx, pair in enumerate(found_pairs, 1):
          print(f"\n[PAR {idx}] - Audio distanca: {pair['distance']:.4f} | Razlika u popularnosti: {pair['pop_diff']} poena")
          print(f"  -> Pesma A: '{pair['song_1']}' by {pair['artist_1']} (Popularnost: {pair['pop_1']})")
          print(f"  -> Pesma B: '{pair['song_2']}' by {pair['artist_2']} (Popularnost: {pair['pop_2']})")
          print("-" * 70)

if __name__ == "__main__":
  main()