from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = (
    PROJECT_ROOT / "data" / "dataset" / "spotify_tracks_audiobox_librosa.parquet"
)


def evaluate_feature_subset(X_train, X_test, y_train, y_test, name):
  """Pomoćna funkcija za treniranje i evaluaciju modela na zadatom skupu obeležja."""
  print(f"\n--- Treniram model za grupu: [{name}] ---")

  scaler = MinMaxScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  model = XGBRegressor(
      n_estimators=150,
      learning_rate=0.05,
      max_depth=6,
      random_state=42,
      n_jobs=-1,
      device="cuda",
  )
  model.fit(X_train_scaled, y_train)

  y_pred = model.predict(X_test_scaled)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  r2 = r2_score(y_test, y_pred)

  print(f"-> Završeno [{name}] | RMSE: {rmse:.4f} | R²: {r2:.4f}")
  return {"RMSE": rmse, "R2": r2}


def main():
  if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset nije pronađen na putanji: {DATASET_PATH}")

  print(f"Učitavam dataset: {DATASET_PATH}")
  df = pd.read_parquet(DATASET_PATH)

  # 1. Definišemo grupe obeležja
  group_meta = ["PQ", "PC", "CE", "CU"]
  group_artist = ["artist_popularity", "followers"]
  group_librosa = [
      col for col in df.columns if col.startswith("librosa_")
  ]

  # Sve zajedno kolone
  all_features = group_meta + group_artist + group_librosa
  target_col = "popularity"

  # Čistimo dataset od NaN vrednosti u svim potencijalnim kolonama
  df_clean = df.dropna(subset=all_features + [target_col]).copy()
  print(f"Broj validnih pesama nakon čišćenja: {len(df_clean)}")

  # Pripremamo sirove podatke
  X_full = df_clean[all_features].values.astype(np.float32)
  y = df_clean[target_col].values.astype(np.float32)

  # Log transformacija za kolonu "followers" unutar celog skupa
  followers_global_idx = all_features.index("followers")
  X_full[:, followers_global_idx] = np.log1p(X_full[:, followers_global_idx])

  # Delimo podatke na trening i test uz očuvanje istih indeksa
  indices = np.arange(len(df_clean))
  train_idx, test_idx = train_test_split(
      indices, test_size=0.2, random_state=42
  )

  # Rečnik za mapiranje naziva kolona na njihove indekse u X_full
  col_to_idx = {col: i for i, col in enumerate(all_features)}

  def get_subset_data(cols):
    idxs = [col_to_idx[c] for c in cols]
    return X_full[train_idx][:, idxs], X_full[test_idx][:, idxs]

  results = {}

  # ==========================================
  # 2. POJEDINAČNE GRUPE (Samo one)
  # ==========================================
  X_train_m, X_test_m = get_subset_data(group_meta)
  results["1. Samo Metapodaci (PQ,PC,CE,CU)"] = evaluate_feature_subset(
      X_train_m, X_test_m, y[train_idx], y[test_idx], "Meta"
  )

  X_train_a, X_test_a = get_subset_data(group_artist)
  results["2. Samo Artist Info (Pop + Followers)"] = evaluate_feature_subset(
      X_train_a, X_test_a, y[train_idx], y[test_idx], "Artist"
  )

  X_train_l, X_test_l = get_subset_data(group_librosa)
  results["3. Samo Librosa Audio Zvuk"] = evaluate_feature_subset(
      X_train_l, X_test_l, y[train_idx], y[test_idx], "Librosa"
  )

  # ==========================================
  # 3. KOMBINACIJE DVE GRUPE (Spajamo svaki sa svakim)
  # ==========================================
  X_train_ma, X_test_ma = get_subset_data(group_meta + group_artist)
  results["4. Meta + Artist Info"] = evaluate_feature_subset(
      X_train_ma, X_test_ma, y[train_idx], y[test_idx], "Meta + Artist"
  )

  X_train_ml, X_test_ml = get_subset_data(group_meta + group_librosa)
  results["5. Meta + Librosa Audio"] = evaluate_feature_subset(
      X_train_ml, X_test_ml, y[train_idx], y[test_idx], "Meta + Librosa"
  )

  X_train_al, X_test_al = get_subset_data(group_artist + group_librosa)
  results["6. Artist Info + Librosa Audio"] = evaluate_feature_subset(
      X_train_al, X_test_al, y[train_idx], y[test_idx], "Artist + Librosa"
  )

  # ==========================================
  # 4. SVE ZAJEDNO (Svi podaci)
  # ==========================================
  X_train_all, X_test_all = get_subset_data(all_features)
  results["7. SVE ZAJEDNO (Meta + Artist + Librosa)"] = evaluate_feature_subset(
      X_train_all, X_test_all, y[train_idx], y[test_idx], "Sve zajedno"
  )

  # ==========================================
  # ZAKLJUČNA POREDBENA TABELA
  # ==========================================
  print("\n" + "=" * 75)
  print(
      f"{'KOMBINACIJA OBELEŽJA (FEATURE GROUPS)':<42} | {'RMSE':<10} |"
      f" {'R² SCORE':<10}"
  )
  print("=" * 75)
  for combo_name, metrics in results.items():
    print(
        f"{combo_name:<42} | {metrics['RMSE']:<10.4f} |"
        f" {metrics['R2']:<10.4f}"
    )
  print("=" * 75)


if __name__ == "__main__":
  main()