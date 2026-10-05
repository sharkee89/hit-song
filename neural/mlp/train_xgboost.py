from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = (
        PROJECT_ROOT / "data" / "dataset" / "spotify_tracks_audiobox_librosa.parquet"
)
MODEL_DIR = PROJECT_ROOT / "data" / "models"


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nije pronađen na putanji: {DATASET_PATH}")

    print(f"Učitavam dataset sa Librosa obeležjima: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)

    # 1. Osnovne karakteristike i metapodaci
    base_features = ["PQ", "PC", "CE", "CU", "artist_popularity", "followers"]

    # 2. Dinamički hvatamo sve librosa kolone iz dataseta
    librosa_features = [col for col in df.columns if col.startswith("librosa_")]

    feature_cols = base_features + librosa_features
    target_col = "popularity"

    print(f"Ukupno ulaznih obeležja (features): {len(feature_cols)}")

    # Čistimo dataset od redova koji imaju NaN vrednosti
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    print(f"Broj validnih pesama za trening nakon čišćenja: {len(df_clean)}")

    if len(df_clean) < 100:
        print("Upozorenje: Mali broj pesama nakon čišćenja. Proveri da li su podaci popunjeni.")

    # Izdvajamo X i y
    X = df_clean[feature_cols].values.astype(np.float32)
    y = df_clean[target_col].values.astype(np.float32)

    # Logaritamska transformacija za kolonu "followers"
    followers_idx = feature_cols.index("followers")
    X[:, followers_idx] = np.log1p(X[:, followers_idx])

    # Delimo podatke na trening (80%) i test (20%) skup
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Skaliramo ulazne podatke pomoću MinMaxScaler-a na opseg [0, 1]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # OPTIMIZACIJA HIPERPARAMETARA (Grid Search)
    # ==========================================
    print("\nPokrećem pretragu optimalnih hiperparametara za XGBoost...")

    base_model = XGBRegressor(
        random_state=42,
        n_jobs=-1,
        device="cuda"
    )

    # Definišemo prostor pretrage za fino podešavanje
    param_grid = {
        'n_estimators': [150, 250],
        'learning_rate': [0.03, 0.05, 0.1],
        'max_depth': [5, 6, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='r2',
        cv=3,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"\nNajbolji parametri pronađeni: {grid_search.best_params_}")

    # Evaluacija optimizovanog modela na test skupu
    y_pred = best_model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("REZULTATI OPTIMIZOVANOG XGBOOST MODELA")
    print("=" * 50)
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"R² Score (Koeficijent determinacije): {r2:.4f}")
    print("=" * 50)

    # ==========================================
    # ANALIZA VAŽNOSTI OBELEŽJA (Feature Importance)
    # ==========================================
    print("\nTop 10 najvažnijih obeležja za predikciju popularnosti:")
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print(feature_importance_df.head(10).to_string(index=False))

    # Čuvanje modela, skalera i liste feature-a za produkciju
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_DIR / "xgboost_audio_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open(MODEL_DIR / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(MODEL_DIR / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    print(f"\nOptimizovani model i prateći fajlovi su uspešno sačuvani u: {MODEL_DIR}")


if __name__ == "__main__":
    main()