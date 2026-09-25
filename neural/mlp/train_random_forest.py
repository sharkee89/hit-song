import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset" / "spotify_tracks_audiobox.parquet"
MODEL_DIR = PROJECT_ROOT / "data" / "models"


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nije pronađen na putanji: {DATASET_PATH}")

    print(f"Učitavam obogaćeni dataset: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)

    # Definišemo ulazne karakteristike (features) i ciljnu vrednost (target)
    feature_cols = ["PQ", "PC", "CE", "CU", "artist_popularity", "followers"]
    target_col = "popularity"

    # Čistimo dataset od redova koji imaju NaN vrednosti u ovim kolonama
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    print(f"Broj validnih pesama za trening nakon čišćenja: {len(df_clean)}")

    # Izdvajamo X i y
    X = df_clean[feature_cols].values.astype(np.float32)
    y = df_clean[target_col].values.astype(np.float32)

    # NAUČNO UNAPREĐENJE: Logaritamska transformacija za kolonu "followers" (indeks 5)
    X[:, 5] = np.log1p(X[:, 5])

    # Delimo podatke na trening (80%) i test (20%) skup
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Skaliramo ulazne podatke pomoću MinMaxScaler-a na opseg [0, 1]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Treniram Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    # Evaluacija modela na test skupu
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Rezultati evaluacije Random Forest modela (MinMax + Log Followers) ---")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"R² Score (Koeficijent determinacije): {r2:.4f}")

    # Čuvanje modela i skalera za inference
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_DIR / "random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(MODEL_DIR / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nRandom Forest model i skaler su uspešno sačuvani u: {MODEL_DIR}")

    # Primer predikcije za nekoliko pesama iz test skupa
    print("\nPrimer predikcija vs Stvarna popularnost:")
    for i in range(5):
        print(f"Predviđeno: {y_pred[i]:.2f} | Stvarno: {y_test[i]:.2f}")


if __name__ == "__main__":
    main()