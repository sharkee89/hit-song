import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


# 1. Definisanje Custom Dataset klase za učitavanje postojećih .npy embedding-a
class AudioEmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, embeddings_dir: Path):
        self.df = df.reset_index(drop=True)
        self.embeddings_dir = embeddings_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Pronalaženje track_id-ja iz parquet tabele
        track_id = str(row.get('id', row.get('track_id', row.name)))
        npy_path = self.embeddings_dir / f"{track_id}.npy"

        # Učitavanje 2048-dimenzionalnog audio embedding-a
        if npy_path.exists():
            audio_emb = np.load(npy_path).astype(np.float32)
        else:
            # Fallback ako fajl ne postoji (nule)
            audio_emb = np.zeros(2048, dtype=np.float32)

        # Ciljna vrednost (Target) - normalizovana popularnost pesme (0.0 - 1.0)
        target = float(row.get('popularity', 0.0)) / 100.0

        return torch.tensor(audio_emb, dtype=torch.float32), torch.tensor([target], dtype=torch.float32)


# 2. Definisanje Audio Agent MLP Arhitekture (Prima 2048 dimenzija, vraća skalar 0.0-1.0)
class AudioAgentPredictor(nn.Module):
    def __init__(self, input_dim=2048):
        super(AudioAgentPredictor, self).__init__()
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
            nn.Sigmoid()  # Osigurava da je izlazni skalar strogo u opsegu [0.0, 1.0]
        )

    def forward(self, x):
        return self.network(x)


# 3. Glavna skripta za treniranje
def main():
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT_DIR = CURRENT_DIR.parent.parent  # Prilagodi putanju po potrebi
    PARQUET_PATH = PROJECT_ROOT_DIR / "data" / "dataset" / "spotify_tracks_with_artists_aligned.parquet"
    EMBEDDINGS_DIR = PROJECT_ROOT_DIR / "data" / "processed" / "audio_embeddings"
    MODEL_OUTPUT_PATH = PROJECT_ROOT_DIR / "models" / "audio_agent_predictor.pth"

    if not PARQUET_PATH.exists():
        print(f"Greška: Parquet fajl nije pronađen na putanji {PARQUET_PATH}")
        return

    print("Učitavanje parquet metapodataka...")
    df = pd.read_parquet(PARQUET_PATH)

    # Filtriramo samo one redove koji imaju validan ID
    df_valid = df.dropna(subset=['popularity']).copy()

    # Deljenje na trening i validaciju (80% - 20%)
    train_df, val_df = train_test_split(df_valid, test_size=0.2, random_state=42)

    train_dataset = AudioEmbeddingDataset(train_df, EMBEDDINGS_DIR)
    val_dataset = AudioEmbeddingDataset(val_df, EMBEDDINGS_DIR)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treniranje se izvršava na uređaju: {device}")

    model = AudioAgentPredictor(input_dim=2048).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 15
    print("Započinjanje treniranja Audio Agent mreže...")

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

    # Čuvanje modela
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    print(f"Audio Agent model je uspešno sačuvan na: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()