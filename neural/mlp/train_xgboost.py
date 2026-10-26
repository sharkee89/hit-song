from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
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

    # 1. Osnovne karakteristike, metapodaci i kolona za grupisanje (izvođač)
    base_features = ["PQ", "PC", "CE", "CU", "artist_popularity", "followers"]

    # Automatski prepoznajemo da li se kolona zove 'artist_name' ili 'artist_id'
    artist_col = "artist_name" if "artist_name" in df.columns else "artist_id"

    # 2. Dinamički hvatamo sve librosa kolone iz dataseta
    librosa_features = [col for col in df.columns if col.startswith("librosa_")]

    feature_cols = base_features + librosa_features
    target_col = "popularity"

    print(f"Ukupno ulaznih obeležja (features): {len(feature_cols)}")

    # Čistimo dataset od redova koji imaju NaN vrednosti (uključujući i kolonu za grupu)
    cols_to_check = feature_cols + [target_col, artist_col]
    df_clean = df.dropna(subset=cols_to_check).copy()
    print(f"Broj validnih pesama za trening nakon čišćenja: {len(df_clean)}")

    if len(df_clean) < 100:
        print("Upozorenje: Mali broj pesama nakon čišćenja. Proveri da li su podaci popunjeni.")

    # Izdvajamo X, y i grupe (izvođače)
    X = df_clean[feature_cols].values.astype(np.float32)
    y = df_clean[target_col].values.astype(np.float32)
    groups = df_clean[artist_col].values

    # Logaritamska transformacija za kolonu "followers"
    followers_idx = feature_cols.index("followers")
    X[:, followers_idx] = np.log1p(X[:, followers_idx])

    # Deljenje podataka pomoću GroupShuffleSplit-a (sprečava Data Leakage po izvođačima)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Trening skup: {len(X_train)} pesama | Test skup: {len(X_test)} pesama (razdvojeno po izvođačima)")

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

    # Definišemo prostor pretrage sa ograničenjima da sprečimo overfitting
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
    # STATISTIKA PREDIKCIJA (Opseg i kalibracija)
    # ==========================================
    print("\n" + "=" * 50)
    print("ANALIZA OPSEGA PREDIKCIJA MODELA")
    print("=" * 50)
    print(
        f"Stvarne vrednosti (y_test) -> Min: {y_test.min():.2f} | Max: {y_test.max():.2f} | Mean: {y_test.mean():.2f}")
    print(
        f"Predviđene vrednosti (y_pred) -> Min: {y_pred.min():.2f} | Max: {y_pred.max():.2f} | Mean: {y_pred.mean():.2f}")

    # Prikaz prvih nekoliko poređenja stvarnih i predviđenih vrednosti
    comparison_df = pd.DataFrame({'Stvarno': y_test[:5], 'Predviđeno': y_pred[:5]})
    print("\nPrimeri prvih 5 predviđanja:")
    print(comparison_df.to_string(index=False))
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