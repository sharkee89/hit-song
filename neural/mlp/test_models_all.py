import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Mašinsko učenje modeli
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# PyTorch za MLP i Transformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset" / "spotify_tracks_audiobox.parquet"


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nije pronađen na putanji: {DATASET_PATH}")

    print(f"Učitavam obogaćeni dataset: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)

    feature_cols = ["PQ", "PC", "CE", "CU", "artist_popularity", "followers"]
    target_col = "popularity"

    # Čistimo dataset od NaN vrednosti
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    print(f"Broj validnih pesama za trening: {len(df_clean)}")

    X = df_clean[feature_cols].values.astype(np.float32)
    y = df_clean[target_col].values.astype(np.float32)

    # Log transformacija za pratioce
    X[:, 5] = np.log1p(X[:, 5])

    # Delimo podatke (isti split za sve modele)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Skaliranje na [0, 1]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # ==========================================
    # 1. RANDOM FOREST
    # ==========================================
    print("\n[1/4] Treniram Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    results["Random Forest"] = {"RMSE": rf_rmse, "R2": rf_r2}
    print(f"-> Random Forest | RMSE: {rf_rmse:.4f} | R2: {rf_r2:.4f}")

    # ==========================================
    # 2. XGBOOST
    # ==========================================
    print("\n[2/4] Treniram XGBoost Regressor...")
    xgb_model = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1,
                             device="cuda")
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)

    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_r2 = r2_score(y_test, xgb_pred)
    results["XGBoost"] = {"RMSE": xgb_rmse, "R2": xgb_r2}
    print(f"-> XGBoost | RMSE: {xgb_rmse:.4f} | R2: {xgb_r2:.4f}")

    # Priprema za PyTorch modele (MLP i Tabular Transformer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

    # ==========================================
    # 3. MLP (Naša neuronska mreža)
    # ==========================================
    print("\n[3/4] Treniram MLP Neural Network...")

    class HitPredictorMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )

        def forward(self, x): return self.network(x)

    mlp_model = HitPredictorMLP(input_dim=len(feature_cols)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.0005)

    mlp_model.train()
    for epoch in range(50):  # 50 epoha za poređenje
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(mlp_model(inputs), targets)
            loss.backward()
            optimizer.step()

    mlp_model.eval()
    with torch.no_grad():
        mlp_pred = mlp_model(X_test_t.to(device)).cpu().numpy().flatten()

    mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))
    mlp_r2 = r2_score(y_test, mlp_pred)
    results["MLP (PyTorch)"] = {"RMSE": mlp_rmse, "R2": mlp_r2}
    print(f"-> MLP | RMSE: {mlp_rmse:.4f} | R2: {mlp_r2:.4f}")

    # ==========================================
    # 4. TABULAR TRANSFORMER (Self-Attention za tabele)
    # ==========================================
    print("\n[4/4] Treniram Tabular Transformer (Attention-based)...")

    class TabularTransformer(nn.Module):
        def __init__(self, input_dim, d_model=64):
            super().__init__()
            self.embed = nn.Linear(input_dim, d_model)
            self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            # x shape: (batch_size, input_dim) -> tretiramo obeležja kao sekvencu dužine 1 ili ih širimo
            # Za tabele: mapiramo u d_model, pa primenimo self-attention nad feature dimenzijom
            h = self.embed(x).unsqueeze(1)  # (batch, 1, d_model)
            attn_out, _ = self.attention(h, h, h)
            h = attn_out.squeeze(1)
            return self.fc(h)

    trans_model = TabularTransformer(input_dim=len(feature_cols)).to(device)
    optimizer_trans = optim.Adam(trans_model.parameters(), lr=0.0005)

    trans_model.train()
    for epoch in range(50):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_trans.zero_grad()
            loss = criterion(trans_model(inputs), targets)
            loss.backward()
            optimizer_trans.step()

    trans_model.eval()
    with torch.no_grad():
        trans_pred = trans_model(X_test_t.to(device)).cpu().numpy().flatten()

    trans_rmse = np.sqrt(mean_squared_error(y_test, trans_pred))
    trans_r2 = r2_score(y_test, trans_pred)
    results["Tabular Transformer"] = {"RMSE": trans_rmse, "R2": trans_r2}
    print(f"-> Transformer | RMSE: {trans_rmse:.4f} | R2: {trans_r2:.4f}")

    # ==========================================
    # ZAKLJUČNA POREDBENA TABELA
    # ==========================================
    print("\n" + "=" * 50)
    print(f"{'MODEL':<25} | {'RMSE':<10} | {'R² SCORE':<10}")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} | {metrics['RMSE']:<10.4f} | {metrics['R2']:<10.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()