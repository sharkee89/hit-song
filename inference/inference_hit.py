import json
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from audio.ai.deepseek import Deepseek
from audio.ai.gemini import Gemini
from audio.ai.gpt import Gpt
from audio.utils.audio_utils import get_librosa_data
from audio.agent.talent.talent_agent import TalentAgent

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "hit-song" / "data" / "models"


def predict_xgboost_popularity(librosa_features: dict, artist_name: str) -> dict:
    """
    Učitava sačuvani XGBoost model, skaler i listu obeležja,
    povlači podatke o izvođaču preko TalentAgent-a, priprema podatke
    i vraća predviđenu popularnost.
    """
    model_path = MODEL_DIR / "xgboost_audio_model.pkl"
    print(model_path)
    scaler_path = MODEL_DIR / "feature_scaler.pkl"
    cols_path = MODEL_DIR / "feature_cols.pkl"

    if not model_path.exists() or not scaler_path.exists() or not cols_path.exists():
        return {"error": "XGBoost modeli ili prateći fajlovi nisu pronađeni na disku."}

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(cols_path, "rb") as f:
        feature_cols = pickle.load(f)

    # Inicijalizujemo TalentAgent i dobijamo podatke o izvođaču
    talent_agent = TalentAgent()
    talent_data = talent_agent.fetch_talent_data(artist_name)

    # Kreiramo rečnik sa podrazumevanim vrednostima
    row_data = {col: 0.0 for col in feature_cols}

    # Popunjavamo librosa karakteristike
    for k, v in librosa_features.items():
        if k in row_data:
            row_data[k] = float(v) if v is not None else 0.0

    # Popunjavamo artist metrike iz TalentAgent-a ako je izvođač pronađen u datasetu
    if talent_data.get("found", False):
        spotify_info = talent_data.get("spotify_data", {})
        artist_pop = spotify_info.get("artist_popularity", 10.0)
        followers = spotify_info.get("followers", 10000)
        print(f"pronađen u datasetu: Popularnost={artist_pop}, Pratioci={followers}")
    else:
        # Fallback vrednosti ako izvođač ne postoji u bazi
        artist_pop = 10.0
        followers = 10000
        print(f"Izvođač '{artist_name}' nije pronađen u datasetu. Koristim podrazumevane vrednosti.")

    if "artist_popularity" in row_data:
        row_data["artist_popularity"] = float(artist_pop)
    if "followers" in row_data:
        row_data["followers"] = float(followers)

    # Konverzija u niz redosledom kojim je model treniran
    x_input = np.array([[row_data[col] for col in feature_cols]], dtype=np.float32)

    # Logaritamska transformacija za "followers"
    if "followers" in feature_cols:
        followers_idx = feature_cols.index("followers")
        x_input[:, followers_idx] = np.log1p(x_input[:, followers_idx])

    # Skaliranje
    x_scaled = scaler.transform(x_input)

    # Predikcija
    predicted_popularity = float(model.predict(x_scaled)[0])

    return {
        "predicted_popularity": round(predicted_popularity, 2),
        "artist_name": artist_name,
        "talent_data_used": talent_data
    }


def get_ai_analysis(features_str: str, audio_file_path: str = None, artist_name: str = None, talent_data: dict = None) -> dict:
    genre_benchmarks = """
    Genre Reference Baselines (Average Audio Features):
    - EDM / Festival Dance: Tempo 124-130 BPM, Energy 0.70-0.90, Spectral Centroid ~2500+ Hz, High Spectral Flatness
    - House / Deep House: Tempo 118-125 BPM, Energy 0.60-0.80, Spectral Centroid ~2200-2600 Hz, Steady 4/4 Kick
    - Techno / Tech-House: Tempo 125-135 BPM, Energy 0.75-0.95, High ZCR, Repetitive Transient Profile
    - Reggaeton / Latin Pop: Tempo 90-105 BPM (Crossover/Remixes ~125-130 BPM), Energy 0.30-0.50, High ZCR/Percussion
    - Afrobeats / Afro-Pop: Tempo 100-125 BPM, Energy 0.35-0.55, Warm Low-Mid Profile, Polyrhythmic Transient Structure
    - Hip-Hop / Trap: Tempo 130-150 BPM (half-time feel), Energy 0.40-0.65, Heavy Sub-Bass (Low MFCC1), Strong Percussive Transients
    - R&B / Contemporary Soul: Tempo 70-100 BPM, Energy 0.25-0.45, High Tonality (Low Flatness), Warm Vocal-Forward Centroid
    - Pop / Dance-Pop: Tempo 115-128 BPM, Energy 0.50-0.75, Bright Centroid (~2000-2400 Hz), High Commercial Polish
    - Synth-Pop / New Wave: Tempo 110-126 BPM, Energy 0.55-0.75, Prominent Mid-High Synthesizers
    - Rock / Alternative Rock: Tempo 110-140 BPM, Energy 0.60-0.85, Broad Spectral Bandwidth, Organic Distortion (High Flatness)
    - Metal / Hard Rock: Tempo 120-180 BPM, Energy 0.80-0.98, High Compression, Dense Frequency Spectrum
    - Indie / Acoustic / Folk: Tempo 90-120 BPM, Energy 0.15-0.35, Lower Spectral Flatness, Organic Dynamics
    - Ambient / Chillout / Lo-Fi: Tempo 60-90 BPM, Energy 0.05-0.25, Low Centroid, Soft High-Frequency Rolloff
    - Jazz / Blues: Tempo 70-130 BPM, Energy 0.20-0.50, High Dynamic Range, Natural Acoustic Timbre
    """

    prompt = (
        f"You are a top music producer, analyst, and A&R expert for Spotify. "
        f"Listen to the attached audio file and analyze its commercial hit potential and market readiness. "
        f"You can also reference the extracted audio features and genre baselines for technical context:\n\n"
        f"Extracted Audio Features:\n{features_str}\n\n"
        f"{genre_benchmarks}\n\n"
        f"this is the artist {artist_name}"
        f"this is data found for artist {talent_data}"
        f"Give detail analysis how artist and its genre is tied to the new song"
        f"Give detail analysis of production and audio quality of a file"
        f"Evaluate objectively: Do not penalize a track solely for moderate energy or specific spectral values if it sounds great and fits its target market.\n"
        f"Provide a detailed, professional assessment covering:\n"
        f"   - Commercial and streaming potential.\n"
        f"   - Most accurate genre and playlist fit.\n"
        f"   - Market placement and target audience.\n"
        f"   - Quality of audio.\n"
        f"   - Actionable production or mixing advice.\n"
        f"   - \n"
        f"   - All output format in a json response that is having following properties: \n"
        f"     prediction grade.\n"
        f"     production quality,\n"
        f"     audio quality,\n"
        f"     genre placement,\n"
        f"     possible_improvements,\n"
        f"     analysis,\n"
        f"     \n"
        f"     For every json property give two fields, numeric value that is a value in number format from 0 to 100 and analysis that is descriptive. Value for possible improvements should be higher if demand for improvements are lower and vice versa.\n"
    )

    # 1. Poziv Gemini modela
    gemini_analysis = Gemini().get_analysis(prompt, audio_file_path=audio_file_path)

    # USLOV: Ako Gemini padne/vrati grešku, prekidamo izvršavanje da ne trošimo tokene na GPT i DeepSeek
    if gemini_analysis.startswith("Error communicating"):
        return {
            "error": "Gemini API call failed. Aborting subsequent AI agent calls to save tokens.",
            "gemini": gemini_analysis,
            "gpt": None,
            "deep_seek": None
        }

    # 2. Poziv GPT modela (izvršava se samo ako je Gemini uspešan)
    gpt_analysis = Gpt().get_analysis(prompt, audio_file_path=audio_file_path)

    # Opciono: Možeš dodati proveru i za GPT ukoliko želiš, mada je Gemini prvi u lancu.
    if gpt_analysis.startswith("Error communicating"):
        return {
            "error": "GPT API call failed.",
            "gemini": gemini_analysis,
            "gpt": gpt_analysis,
            "deep_seek": None
        }

    # 3. Poziv DeepSeek meta-arbitraže
    deep_seek_analysis = Deepseek().get_analysis(gemini_analysis, gpt_analysis)

    return {
        "deep_seek": deep_seek_analysis,
        "gemini": gemini_analysis,
        "gpt": gpt_analysis
    }


def main():
    print("Executing inference pipeline...")

    audio_file_path = os.getenv("AUDIO_FILE_PATH")
    artist_name = os.getenv("ARTIST_NAME", "Unknown Artist")

    if not audio_file_path or not os.path.exists(audio_file_path):
        print(f"Upozorenje: AUDIO_FILE_PATH nije pronađen ili fajl ne postoji: {audio_file_path}")

    # 1. Ekstrakcija librosa karakteristika za novu pesmu
    audio_data = get_librosa_data()

    # 2. XGBoost predikcija sa stvarnim podacima iz TalentAgent-a
    xgboost_prediction = predict_xgboost_popularity(audio_data, artist_name=artist_name)
    artist_name_used = xgboost_prediction.get("artist_name", artist_name)
    talent_data_used = xgboost_prediction.get("talent_data_used", {})

    # 3. LLM AI analize
    ai_analysis = get_ai_analysis(str(audio_data), audio_file_path=audio_file_path, artist_name=artist_name_used, talent_data=talent_data_used)

    # Kombinovani podaci
    data = {
        "xgboost_prediction": xgboost_prediction,
        "ai_analysis": ai_analysis
    }

    print(json.dumps(data, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()