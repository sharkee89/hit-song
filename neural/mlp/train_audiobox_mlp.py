import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
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

    # Definišemo ulazne karakteristike (features) i ciljnu vrednost (target)
    feature_cols = ["PQ", "PC", "CE", "CU", "artist_popularity", "followers"]
    target_col = "popularity"

    # Čistimo dataset od redova koji imaju NaN vrednosti u ovim kolonama
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    print(f"Broj validnih pesama za trening nakon čišćenja: {len(df_clean)}")

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values.astype(np.float32)

    # Delimo podatke na trening (80%) i test (20%) skup
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Skaliramo ulazne podatke (veoma važno za neuronske mreže!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Pretvaramo u PyTorch tenseore
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Definšemo MLP arhitekturu
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

        def forward(self, x):
            return self.network(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treniram mrežu na uređaju: {device}")

    model = HitPredictorMLP(input_dim=len(feature_cols)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Petlja za treniranje
    epochs = 50
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoha [{epoch+1}/{epochs}], Loss (MSE): {epoch_loss:.4f}")

    # Evaluacija modela na test skupu
    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_t.to(device)
        y_pred_dev = model(X_test_dev)
        y_pred = y_pred_dev.cpu().numpy().flatten()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Rezultati evaluacije MLP modela ---")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"R² Score (Koeficijent determinacije): {r2:.4f}")

    # Primer predikcije za nekoliko pesama iz test skupa
    print("\nPrimer predikcija vs Stvarna popularnost:")
    for i in range(5):
        print(f"Predviđeno: {y_pred[i]:.2f} | Stvarno: {y_test[i]:.2f}")

if __name__ == "__main__":
    main()