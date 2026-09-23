import os
import io
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import librosa
import soundfile as sf
from dotenv import load_dotenv

# Učitavanje .env fajla
load_dotenv()

# Podešavanje putanja
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from audio.agent.audio_agent import AudioAgent


# 1. Definisanje dedikovane MLP arhitekture za Audio Agent (Prima 2048 dimenzija -> Vraća 1 skalar)
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
            nn.Sigmoid()  # Osigurava opseg [0.0, 1.0]
        )

    def forward(self, x):
        return self.network(x)


SHARED_AGENT = AudioAgent()
MODEL_PATH = PROJECT_ROOT_DIR / "models" / "audio_agent_predictor.pth"
EMBEDDINGS_DIR = PROJECT_ROOT_DIR / "data" / "processed" / "audio_embeddings"
PARQUET_PATH = PROJECT_ROOT_DIR / "data" / "dataset" / "spotify_tracks_all.parquet"


def extract_and_save_peak_window(audio_source: str, duration: float = 30.0,
                                 output_filename: str = "extracted_30s_window.wav"):
    """Pronalazi 30-sekundni prozor sa najvećom RMS energijom i čuva ga na disk."""
    try:
        # Učitavanje audio fajla preko librosa-e
        y, sr = librosa.load(audio_source, sr=22050)
        total_duration = len(y) / sr

        target_samples = int(duration * sr)

        if len(y) <= target_samples:
            # Ako je pesma kraća od 30 sekundi, čuvamo je celu
            best_segment = y
            print(f"Audio is shorter than {duration}s. Saving full audio.")
        else:
            # RMS prozoring (korak od 1 sekunde)
            hop_length = sr
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]

            max_rms = -1
            best_start_sample = 0

            # Pretraga prozora od 30 sekundi sa maksimalnom energijom
            for i in range(len(rms)):
                start_sample = i * hop_length
                end_sample = start_sample + target_samples
                if end_sample > len(y):
                    break

                window_rms = np.mean(rms[i:i + int(duration)])
                if window_rms > max_rms:
                    max_rms = window_rms
                    best_start_sample = start_sample

            best_segment = y[best_start_sample:best_start_sample + target_samples]
            start_sec = best_start_sample / sr
            print(f"🎵 Peak RMS window found from {start_sec:.2f}s to {start_sec + duration:.2f}s")

        # Čuvanje izdvojenog segmenta
        output_path = PROJECT_ROOT_DIR / output_filename
        sf.write(output_path, best_segment, sr)
        print(f"💾 Successfully saved extracted 30s window to: {output_path}")

    except Exception as e:
        print(f"⚠️ Warning: Could not save peak RMS window segment: {e}")


def find_most_similar_tracks(query_embedding: np.ndarray, top_k: int = 3):
    """Upoređuje query embedding sa svim sačuvanim .npy fajlovima i vraća najsličnije."""
    if not EMBEDDINGS_DIR.exists():
        print(f"⚠️ Embeddings directory not found at {EMBEDDINGS_DIR}. Skipping similarity search.")
        return []

    similarities = []

    # Učitavanje metapodataka iz Parquet-a ako postoji da bismo videli nazive pesama
    df = None
    if PARQUET_PATH.exists():
        try:
            df = pd.read_parquet(PARQUET_PATH)
        except Exception:
            pass

    print(f"Scanning precomputed embeddings for similarity check...")
    for npy_file in EMBEDDINGS_DIR.glob("*.npy"):
        track_id = npy_file.stem
        try:
            saved_emb = np.load(npy_file).reshape(1, -1)
            sim = cosine_similarity(query_embedding, saved_emb)[0][0]
            similarities.append((track_id, sim))
        except Exception:
            continue

    if not similarities:
        return []

    # Sortiranje po najvećoj sličnosti
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarities[:top_k]

    # Mapiranje track_id-ova na nazive pesama iz dataframe-a
    results = []
    for track_id, sim in top_matches:
        track_name = "Unknown Track"
        artist_name = "Unknown Artist"
        if df is not None:
            match_row = None
            if 'id' in df.columns:
                matched = df[df['id'] == track_id]
                if not matched.empty:
                    match_row = matched.iloc[0]
            elif track_id in df.index:
                match_row = df.loc[track_id]

            if match_row is not None:
                track_name = match_row.get('name', match_row.get('name', 'Unknown'))
                artist_name = match_row.get('artist_name', match_row.get('artists', 'Unknown'))

        results.append({
            "track_id": track_id,
            "track_name": track_name,
            "artist_name": artist_name,
            "similarity": float(sim)
        })

    return results


def test_audio_agent_inference(audio_source: str):
    print(f"1. Executing fetching audio data: {audio_source}")

    try:
        if audio_source.startswith("http"):
            import requests
            response = requests.get(audio_source, timeout=15)
            response.raise_for_status()
            audio_bytes = io.BytesIO(response.content)
        else:
            with open(audio_source, "rb") as f:
                audio_bytes = io.BytesIO(f.read())

        # Čuvanje 30-sekundnog segmenta koji agent koristi
        if not audio_source.startswith("http"):
            extract_and_save_peak_window(audio_source)

        # Ekstrakcija 2048-dimenzionalnog embedding-a preko AudioAgent-a
        embedding = SHARED_AGENT.extract_features(audio_bytes)
        audio_emb = np.array(embedding, dtype=np.float32)
        print(f"Successfully fetched audio embedding dimensions: {audio_emb.shape}")

    except Exception as e:
        print(f"ERROR during fetching audio file: {e}")
        return

    print("2. Loading trained audio agent model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AudioAgentPredictor(input_dim=2048)
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found on path {MODEL_PATH}. Train audio model first!")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Priprema tenzora za model (dodajemo batch dimenziju -> [1, 2048])
    x_tensor = torch.tensor(audio_emb, dtype=torch.float32).unsqueeze(0).to(device)

    print("3. Executing audio agent prediction...")
    with torch.no_grad():
        output = model(x_tensor)
        audio_score = output.item()

    # 4. Provera sličnosti sa ostalim pesmama u bazi (Sanity check)
    query_2d = audio_emb.reshape(1, -1)
    similar_tracks = find_most_similar_tracks(query_2d, top_k=3)

    print(f"\n--- TEST AUDIO AGENT ---")
    print(f"Input file: {audio_source}")
    print(f"Audio Hit Scalar (0.0 - 1.0): {audio_score:.4f}")
    print(f"Hit potential prediction:       {audio_score * 100:.2f} / 100")

    if similar_tracks:
        print(f"\n--- EMBEDDING SANITY CHECK (Closest matches in dataset) ---")
        for i, match in enumerate(similar_tracks, 1):
            print(f"{i}. '{match['track_name']}' by {match['artist_name']} (Similarity: {match['similarity']:.4f})")

    print(f"------------------------------------")


def main():
    audio_source = os.getenv("AUDIO_FILE_PATH", "")

    if not audio_source:
        print("ERROR: Not defined AUDIO_FILE_PATH in .env file!")
        return

    test_audio_agent_inference(audio_source)


if __name__ == "__main__":
    main()