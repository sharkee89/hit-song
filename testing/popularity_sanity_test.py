import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Pretpostavka putanja
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent  # Prilagodi po potrebi
PARQUET_PATH = PROJECT_ROOT_DIR / "data" / "dataset" / "spotify_tracks_with_artists_aligned.parquet"
EMBEDDINGS_DIR = PROJECT_ROOT_DIR / "data" / "processed" / "audio_embeddings"
MODEL_PATH = PROJECT_ROOT_DIR / "models" / "audio_agent_predictor.pth"

# 1. Učitavanje modela (isti onaj tvoj AudioAgentPredictor)
from testing.test_audio_agent import AudioAgentPredictor  # ili unesi klasu ovde

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioAgentPredictor(input_dim=2048)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

# 2. Učitavanje Parquet tabele
df = pd.read_parquet(PARQUET_PATH)

# 3. Filtriranje dve grupe: HIT-ovi (popularity >= 80) i NE-HIT-ovi (popularity <= 10)
hits_df = df[df['popularity'] >= 80].sample(n=min(1000, len(df[df['popularity'] >= 80])), random_state=42)
non_hits_df = df[df['popularity'] <= 10].sample(n=min(1000, len(df[df['popularity'] <= 10])), random_state=42)


def evaluate_group(subset_df):
    scores = []
    for _, row in subset_df.iterrows():
        track_id = str(row.get('id', row.get('track_id')))
        npy_path = EMBEDDINGS_DIR / f"{track_id}.npy"

        if npy_path.exists():
            emb = np.load(npy_path).astype(np.float32)
            tensor_x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(tensor_x).item()
                scores.append(pred)

    return np.mean(scores) if scores else 0.0


print("Evaluacija u toku...")
mean_hit_score = evaluate_group(hits_df)
mean_non_hit_score = evaluate_group(non_hits_df)

print(f"\n--- REZULTATI SANITY CHECK-A NA POPULACIJI ---> ")
print(f"Prosečan predviđeni skalar za prave HIT-ove (pop >= 80): {mean_hit_score:.4f}")
print(f"Prosečan predviđeni skalar za NE-HIT-ove (pop <= 10): {mean_non_hit_score:.4f}")