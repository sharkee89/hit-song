import io
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import torch
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Učitavanje .env fajla
load_dotenv()

# Ensure root path resolution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from neural.mlp.train_mlp import HitPredictorMLP
from audio.agent.audio_agent import AudioAgent

# Inicijalizacija globalnog AudioAgent-a
SHARED_AGENT = AudioAgent()

PARQUET_PATH = PROJECT_ROOT_DIR / "data" / "dataset" / "spotify_tracks_all.parquet"
EMBEDDINGS_DIR = PROJECT_ROOT_DIR / "data" / "processed" / "audio_embeddings"
MODEL_PATH = PROJECT_ROOT_DIR / "models" / "hit_predictor_mlp.pth"


def extract_features_for_new_track(audio_source: str, track_id: str = "new_inference_track") -> Path:
    """Ekstrahuje audio embedding preko AudioAgent-a i čuva ga kao .npy fajl."""
    save_path = EMBEDDINGS_DIR / f"{track_id}.npy"
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        return save_path

    try:
        # Provera da li je URL ili lokalna putanja
        if audio_source.startswith("http"):
            response = requests.get(audio_source, timeout=15)
            response.raise_for_status()
            audio_bytes = io.BytesIO(response.content)
        else:
            with open(audio_source, "rb") as f:
                audio_bytes = io.BytesIO(f.read())

        # Ekstrakcija karakteristika preko AudioAgent-a
        embedding = SHARED_AGENT.extract_features(audio_bytes)
        np.save(save_path, np.array(embedding, dtype=np.float32))
        return save_path
    except Exception as e:
        print(f"Greška pri ekstrakciji audio embedding-a: {e}. Koriste se nule.")
        zero_emb = np.zeros(2048, dtype=np.float32)
        np.save(save_path, zero_emb)
        return save_path


def get_talent_features(artist_name: str, df: pd.DataFrame) -> list:
    """Traži izvođača u datasetu. Ako ga nađe, uzima njegove podatke, inače vraća nule."""
    if not artist_name or df.empty:
        return [0.0, 0.0, 0.0, 0.0]

    # Pretraga case-insensitive po imenu izvođača (pretpostavka da je kolona 'artist_name' ili 'artist')
    artist_col = 'artist_name' if 'artist_name' in df.columns else ('artist' if 'artist' in df.columns else None)

    if not artist_col:
        return [0.0, 0.0, 0.0, 0.0]

    matched = df[df[artist_col].str.lower() == artist_name.lower()]

    if not matched.empty:
        row = matched.iloc[0]
        return [
            float(row.get('talent_norm_popularity', 0.0)),
            float(row.get('talent_norm_followers', 0.0)),
            float(row.get('talent_hype_ratio', 0.0)),
            float(row.get('talent_has_artist_data', 1.0))
        ]
    else:
        print(f"Izvođač '{artist_name}'' nije pronađen u bazi. Cold start (nule).")
        return [0.0, 0.0, 0.0, 0.0]


def get_trend_features_by_similarity(new_embedding_path: Path, df: pd.DataFrame) -> list:
    """Nalazi najsličniju pesmu po kosinusnoj distanci embedding-a i uzima njen trend."""
    try:
        new_emb = np.load(new_embedding_path).reshape(1, -1)
    except Exception:
        return [0.0, 0.0]

    best_similarity = -1.0
    best_trend_alignment = 0.0
    best_trend_is_active = 0.0

    # Optimizovano poređenje sa pesmama iz dataseta koje imaju embedding fajlove
    for idx, row in df.iterrows():
        t_id = row.get('track_id', row.name)
        emb_p = EMBEDDINGS_DIR / f"{t_id}.npy"
        if emb_p.exists():
            try:
                db_emb = np.load(emb_p).reshape(1, -1)
                sim = cosine_similarity(new_emb, db_emb)[0][0]
                if sim > best_similarity:
                    best_similarity = sim
                    best_trend_alignment = float(row.get('trend_alignment', 0.0))
                    best_trend_is_active = float(row.get('trend_is_active', 0.0))
            except Exception:
                continue

    return [best_trend_alignment, best_trend_is_active]


def predict_track_score(
        audio_emb_path: str,
        talent_features: list,
        trend_features: list,
        model_path: str
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HitPredictorMLP(input_dim=2054)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model nije pronađen na putanji: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    emb_path = Path(audio_emb_path)
    if emb_path.exists():
        audio_emb = np.load(emb_path).astype(np.float32)
    else:
        print(f"Warning: Audio embedding file {emb_path} not found. Using zeros.")
        audio_emb = np.zeros(2048, dtype=np.float32)

    talent_arr = np.array(talent_features, dtype=np.float32)
    trend_arr = np.array(trend_features, dtype=np.float32)

    x = np.concatenate([audio_emb, talent_arr, trend_arr])
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        mlp_output = model(x_tensor)
        print(f"DEBUG - Sirovi izlaz (logit) pre Sigmoida: {mlp_output.item():.4f}")
        mlp_score = torch.sigmoid(mlp_output).item()

    return mlp_score


def calculate_viral_score(track_meta=None) -> float:
    return 0.5


def score_fusion(mlp_score: float, viral_score: float, w1: float = 0.7, w2: float = 0.3) -> float:
    final_score = (w1 * mlp_score) + (w2 * viral_score)
    return final_score


def main():
    # Učitavanje parametara iz .env fajla
    artist_name = os.getenv("ARTIST_NAME", "Unknown Artist")
    audio_source = os.getenv("AUDIO_FILE_PATH", "")

    if not audio_source:
        print("GREŠKA: Nije definisan INFER_AUDIO_URL u .env fajlu!")
        return

    print(f"Učitavanje dataset-a za pretragu izvođača i trend sličnosti...")
    if not PARQUET_PATH.exists():
        print(f"GREŠKA: Parquet fajl nije pronađen na {PARQUET_PATH}")
        return
    df = pd.read_parquet(PARQUET_PATH)

    print(f"Ekstrakcija audio karakteristika za novu pesmu...")
    audio_emb_path = extract_features_for_new_track(audio_source, track_id="target_new_track")

    print(f"Pretraga podataka za izvođača: '{artist_name}'...")
    talent_features = get_talent_features(artist_name, df)

    print(f"Pronalaženje trend karakteristika preko vektorske sličnosti audio embedding-a...")
    trend_features = get_trend_features_by_similarity(audio_emb_path, df)

    print("Pokretanje predikcije modela...")
    mlp_score = predict_track_score(
        audio_emb_path=str(audio_emb_path),
        talent_features=talent_features,
        trend_features=trend_features,
        model_path=MODEL_PATH
    )

    viral_score = calculate_viral_score()
    final_hit_score = score_fusion(mlp_score, viral_score, w1=0.7, w2=0.3)

    print(f"\n--- REZULTATI ANALIZE NOVE PESME ---")
    print(f"Izvođač: {artist_name}")
    print(f"Talent vektor:  {talent_features}")
    print(f"Trend vektor:   {trend_features} (nađen preko sličnosti)")
    print(f"----------------------------------------")
    print(f"MLP Hit Score (Audio + Talent + Trend): {mlp_score * 100:.2f} / 100")
    print(f"Viralni Heuristički Score:              {viral_score * 100:.2f} / 100")
    print(f"----------------------------------------")
    print(f"KONAČNI INTEGRISANI HIT SCORE:          {final_hit_score * 100:.2f} / 100")


if __name__ == "__main__":
    main()