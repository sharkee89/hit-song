import os
from pathlib import Path
import io
import requests
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. PODEŠAVANJE PUTANJA I UČITAVANJE DATASETA
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset" / "spotify_tracks_audiobox_librosa.parquet"

print(f"Učitavam dataset sa: {DATASET_PATH}")
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset nije pronađen na putanji: {DATASET_PATH}")

df = pd.read_parquet(DATASET_PATH)

# Filtriramo samo redove koji imaju validan preview_url i target (popularity)
df_clean = df.dropna(subset=["preview_url", "popularity"]).copy()

# Ograničićemo na npr. 2000 pesama za prvi test da se skripta ne izvršava satima
df_clean = df_clean.head(2000)
print(f"Broj pesama izabranih za audio test: {len(df_clean)}")


# ==========================================
# 2. AUDIO DATASET (Preuzimanje i Mel-Spektrogram)
# ==========================================
class SpotifyAudioDataset(Dataset):
    def __init__(self, dataframe, target_col="popularity", sr=22050, duration=30):
        self.df = dataframe.reset_index(drop=True)
        self.sr = sr
        self.target_col = target_col
        self.samples = sr * duration

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        url = row["preview_url"]
        target = float(row[self.target_col])

        print(f"Skidam audio [{idx+1}/{len(self.df)}]: {url}")

        # Preuzimanje audio fajla u memoriju
        audio_tensor = self._download_and_process_audio(url)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return audio_tensor, target_tensor

    def _download_and_process_audio(self, url):
        fixed_time_frames = 300  # Fiksna širina za sve spektrograme
        default_spectrogram = torch.zeros((1, 128, fixed_time_frames), dtype=torch.float32)
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return default_spectrogram

            audio_bytes = io.BytesIO(response.content)
            y, _ = librosa.load(audio_bytes, sr=self.sr, duration=30)

            if len(y) < self.samples:
                y = np.pad(y, (0, self.samples - len(y)))
            else:
                y = y[:self.samples]

            melspec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128, n_fft=2048, hop_length=512)
            melspec_db = librosa.power_to_db(melspec, ref=np.max)

            # Normalizacija
            melspec_db = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min() + 1e-8)

            # Fiksiranje dimenzije po vremenskoj osi (drugа osa, npr. širina)
            current_frames = melspec_db.shape[1]
            if current_frames < fixed_time_frames:
                # Padding nulama ako je kraće
                melspec_db = np.pad(melspec_db, ((0, 0), (0, fixed_time_frames - current_frames)), mode='constant')
            else:
                # Skraćivanje ako je duže
                melspec_db = melspec_db[:, :fixed_time_frames]

            return torch.tensor(melspec_db, dtype=torch.float32).unsqueeze(0)

        except Exception:
            return default_spectrogram


# Podela na trening i test set
train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)

train_dataset = SpotifyAudioDataset(train_df)
test_dataset = SpotifyAudioDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Koristim uređaj: {device}")


# ==========================================
# 3. DEFINICIJA MODELA: CNN vs AUDIO TRANSFORMER
# ==========================================

# A. CNN Model (Za obradu slika spektrograma)
class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 64 x 65
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32 x 32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Svedemo na fiksnu dimenziju 64 x 4 x 4
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(1)


# B. Audio Transformer Model (Pažnja preko vremenskih/frekventnih isečaka)
class AudioTransformer(nn.Module):
    def __init__(self, input_dim=128, d_model=128, num_heads=4, num_layers=2):
        super().__init__()
        # Projiciramo frekvencijske dimenzije (128 mels) u d_model
        self.proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x oblik: (Batch, 1, 128, Time) -> Transponujemo u (Batch, Time, 128)
        x = x.squeeze(1).permute(0, 2, 1)
        x = self.proj(x)
        x = self.transformer(x)
        # Uzimamo prosek preko vremenske ose (Global Average Pooling)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(1)


# ==========================================
# 4. TRENING I EVALUACIJA (TRAINING LOOP)
# ==========================================
def train_and_evaluate(model, model_name, epochs=5):
    print(f"\n--- Treniram {model_name} ---")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoha {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

    # Evaluacija
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    print(f"Rezultat za {model_name} -> RMSE: {rmse:.4f} | R²: {r2:.4f}")
    return rmse, r2


if __name__ == "__main__":
    # Pokrećemo CNN
    cnn_model = AudioCNN()
    cnn_rmse, cnn_r2 = train_and_evaluate(cnn_model, "Audio CNN", epochs=5)

    # Pokrećemo Audio Transformer
    trans_model = AudioTransformer()
    trans_rmse, trans_r2 = train_and_evaluate(trans_model, "Audio Transformer", epochs=5)

    print("\n" + "=" * 40)
    print(f"{'MODEL':<20} | {'RMSE':<10} | {'R² SCORE':<10}")
    print("=" * 40)
    print(f"{'Audio CNN':<20} | {cnn_rmse:<10.4f} | {cnn_r2:<10.4f}")
    print(f"{'Audio Transformer':<20} | {trans_rmse:<10.4f} | {trans_r2:<10.4f}")
    print("=" * 40)