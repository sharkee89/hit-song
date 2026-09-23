import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


# 1. Definisanje Custom Dataset klase
class SpotifyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, embeddings_dir: Path):
        self.df = df.reset_index(drop=True)
        self.embeddings_dir = embeddings_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Pretpostavka je da se .npy fajlovi zovu po ID-ju pesme ili indeksu (npr. track_id.npy)
        # Prilagodi naziv kolone ukoliko se ID pesme zove drugačije u parquet-u (npr. 'id' ili 'track_id')
        track_id = row['track_id'] if 'track_id' in row else row.name
        npy_path = self.embeddings_dir / f"{track_id}.npy"

        # Učitavanje audio embedding-a (2048 dimenzija)
        if npy_path.exists():
            audio_emb = np.load(npy_path).astype(np.float32)
        else:
            # Fallback ako fajl slučajno fali (nule)
            audio_emb = np.zeros(2048, dtype=np.float32)

        # Skupljanje tabličnih karakteristika (Talent: 4, Trend: 2)
        talent_features = np.array([
            row.get('talent_norm_popularity', 0.0),
            row.get('talent_norm_followers', 0.0),
            row.get('talent_hype_ratio', 0.0),
            row.get('talent_has_artist_data', 0.0)
        ], dtype=np.float32)

        trend_features = np.array([
            row.get('trend_alignment', 0.0),
            row.get('trend_is_active', 0.0)
        ], dtype=np.float32)

        # Spajanje u jedinstveni ulazni vektor od 2054 dimenzije
        x = np.concatenate([audio_emb, talent_features, trend_features])

        # Ciljna promenljiva (Label) - npr. normalizovana popularnost pesme ili hit score iz dataseta
        # Prilagodi naziv kolone u parquet-u koja predstavlja tvoj cilj (target)
        target = float(row.get('popularity', 0.0)) / 100.0

        return torch.tensor(x, dtype=torch.float32), torch.tensor([target], dtype=torch.float32)


# 2. Definisanje MLP Arhitekture
class HitPredictorMLP(nn.Module):
    def __init__(self, input_dim=2054):
        super(HitPredictorMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


# 3. Glavna skripta za pokretanje treninga
def main():
    # Putanje
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT_DIR = CURRENT_DIR.parent.parent  # Prilagodi po potrebi strukturi projekta
    PARQUET_PATH = PROJECT_ROOT_DIR / "data" / "dataset" / "spotify_tracks_all.parquet"
    EMBEDDINGS_DIR = PROJECT_ROOT_DIR / "data" / "processed" / "audio_embeddings"

    print("Učitavanje master dataset-a...")
    df = pd.read_parquet(PARQUET_PATH)

    # Deljenje na trening i validaciju (80% - 20%)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = SpotifyDataset(train_df, EMBEDDINGS_DIR)
    val_dataset = SpotifyDataset(val_df, EMBEDDINGS_DIR)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Konfiguracija uređaja (GPU ako je dostupan)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Koristi se uređaj: {device}")

    model = HitPredictorMLP(input_dim=2054).to(device)
    criterion = nn.MSELoss()  # Srednja kvadratna greška za regresiju skora
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    print("Početak treniranja MLP mreže...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        # Validacija
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoha [{epoch + 1}/{epochs}] | Trening Loss: {train_loss:.4f} | Validacioni Loss: {val_loss:.4f}")

    # Čuvanje istreniranog modela
    model_output_path = PROJECT_ROOT_DIR / "models" / "hit_predictor_mlp.pth"
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_path)
    print(f"Model je uspešno sačuvan na: {model_output_path}")


if __name__ == "__main__":
    main()