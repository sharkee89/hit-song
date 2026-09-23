import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Učitavanje .env fajla (gde je definisan AUDIO_FILE_PATH)
load_dotenv()

# Podešavanje putanja
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# Uvozimo tvoj postojeći AudioAgent iz projekta da koristimo istu metodu ekstrakcije
from audio.agent.audio_agent import AudioAgent

EMBEDDINGS_DIR = DATA_DIR / "data" / "processed" / "audio_embeddings"
PARQUET_PATH = PROJECT_ROOT_DIR / "data" / "dataset" / "spotify_tracks_all.parquet"


def extract_query_embedding(audio_path: str, agent: AudioAgent) -> np.ndarray:
    """Ekstrahuje embedding za ulazni fajl koristeći isti AudioAgent iz sistema."""
    print(f"Obrada ulaznog audio fajla: {audio_path}")

    # AudioAgent očekuje fajl ili BytesIO bafer. Prosleđujemo putanju direktno.
    # Ako AudioAgent u tvojoj implementaciji očekuje BytesIO, možemo i ovako:
    with open(audio_path, "rb") as f:
        import io
        audio_bytes = io.BytesIO(f.read())
        embedding = agent.extract_features(audio_bytes)

    return np.array(embedding, dtype=np.float32)


def find_matching_track(query_embedding: np.ndarray, top_k: int = 5):
    """Poredi upit sa svim .npy fajlovima u bazi pomoću kosinusne sličnosti."""
    if not EMBEDDINGS_DIR.exists():
        print(f"Greška: Direktorijum sa embedding-ima ne postoji: {EMBEDDINGS_DIR}")
        return []

    df = None
    if PARQUET_PATH.exists():
        try:
            df = pd.read_parquet(PARQUET_PATH)
            # Ispisujemo kolone da vidimo tačan naziv ako zatreba
            print(f"Učitana Parquet tabela. Dostupne kolone: {list(df.columns)}")
        except Exception as e:
            print(f"Upozorenje: Nije moguće učitati parquet tabelu: {e}")

    print("Skeniram bazu sačuvanih .npy embedding-a...")
    similarities = []
    query_2d = query_embedding.reshape(1, -1)

    for npy_file in EMBEDDINGS_DIR.glob("*.npy"):
        track_id = npy_file.stem
        try:
            saved_emb = np.load(npy_file).astype(np.float32)
            if saved_emb.shape != query_embedding.shape:
                continue
            saved_emb_2d = saved_emb.reshape(1, -1)
            sim = cosine_similarity(query_2d, saved_emb_2d)[0][0]
            similarities.append((track_id, sim))
        except Exception:
            continue

    if not similarities:
        return []

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarities[:top_k]

    results = []
    for track_id, sim in top_matches:
        track_name = "Nepoznata pesma"
        artist_name = "Nepoznat izvođač"

        if df is not None:
            matched_row = None

            # 1. Provera da li je track_id u indeksu tabele
            if track_id in df.index.astype(str):
                matched_row = df.loc[df.index.astype(str) == track_id].iloc[0]
            else:
                # 2. Provera po kolonama ako indeks ne odgovara
                for col in df.columns:
                    if 'id' in col.lower():
                        subset = df[df[col].astype(str) == track_id]
                        if not subset.empty:
                            matched_row = subset.iloc[0]
                            break

            if matched_row is not None:
                # Izvlačenje naziva i izvođača bez obzira na naziv kolone
                for col_name in ['name', 'track_name', 'title']:
                    if col_name in matched_row:
                        track_name = matched_row[col_name]
                        break
                for col_name in ['artist_name', 'artists', 'artist']:
                    if col_name in matched_row:
                        artist_name = matched_row[col_name]
                        break

        results.append({
            "track_id": track_id,
            "track_name": track_name,
            "artist_name": artist_name,
            "similarity": float(sim)
        })

    return results


def main():
    audio_source = os.getenv("AUDIO_FILE_PATH", "")

    if not audio_source or not os.path.exists(audio_source):
        print(f"GRESKA: AUDIO_FILE_PATH nije definisan u .env-u ili fajl ne postoji: {audio_source}")
        return

    # Inicijalizacija tvog AudioAgent-a
    agent = AudioAgent()

    # 1. Ekstrakcija embedding-a za dati fajl
    query_emb = extract_query_embedding(audio_source, agent)
    print(f"Uspešno izvučen embedding. Dimenzije: {query_emb.shape}")

    # 2. Pretraga najsličnijih u bazi
    matches = find_matching_track(query_emb, top_k=5)

    print(f"\n--- REZULTATI PREPOZNAVANJA AUDIO FAJLA ---")
    print(f"Ulazni fajl: {audio_source}")
    if matches:
        for i, match in enumerate(matches, 1):
            print(
                f"{i}. '{match['track_name']}' - {match['artist_name']} (Sličnost: {match['similarity']:.4f}) [ID: {match['track_id']}]")
    else:
        print("Nema pronađenih podudaranja u bazi.")
    print(f"--------------------------------------------")


if __name__ == "__main__":
    main()