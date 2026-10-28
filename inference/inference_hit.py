import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from audio.ai.deepseek import Deepseek
from audio.ai.gemini import Gemini
from audio.ai.gpt import Gpt
from audio.utils.audio_utils import extract_peak_window, get_audiobox_data, get_librosa_data
from audio.agent.talent.talent_agent import TalentAgent
from inference.hit_model import (
    META_PATH, MODEL_DIR, MODEL_PATH, artist_catalog, build_feature_row, hit_probability,
    load_hit_model, parse_genres, popularity_percentile,
)

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def _talent_agent() -> TalentAgent:
    return TalentAgent()


def predict_xgboost_popularity(librosa_features: dict, artist_name: str, audiobox_features: dict = None) -> dict:
    """
    Učitava XGBoost hit model (inference/hit_model.py), povlači podatke o izvođaču
    preko TalentAgent-a, formira obeležja identično treningu i vraća predviđenu popularnost.
    """
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return {"error": "XGBoost model nije pronađen. Pokreni: python -m neural.mlp.train_xgboost"}

    model, meta = load_hit_model()
    feature_cols = meta["feature_cols"]

    talent_data = _talent_agent().fetch_talent_data(artist_name)
    artist_found = bool(talent_data.get("found", False))

    if artist_found:
        spotify_info = talent_data.get("spotify_data", {})
        artist_pop = spotify_info.get("artist_popularity", 0.0)
        followers = spotify_info.get("followers", 0)
        genres = parse_genres(spotify_info.get("genres", []))
        catalog_mean, catalog_count = artist_catalog(spotify_info.get("spotify_id", ""))
        print(f"pronađen u datasetu: Popularnost={artist_pop}, Pratioci={followers}, Katalog={catalog_count} pesama")
    else:
        # Neutralan profil (medijana trening skupa) umesto kažnjavanja nepoznatog izvođača
        artist_pop = meta["artist_fallback"]["artist_popularity"]
        followers = meta["artist_fallback"]["followers"]
        genres, catalog_mean, catalog_count = [], None, 0
        print(f"Izvođač '{artist_name}' nije pronađen u datasetu. Koristim tipičan (medijana) profil izvođača.")

    row_data = build_feature_row(
        feature_cols,
        {**librosa_features, **(audiobox_features or {})},
        artist_pop, followers, genres, catalog_mean, catalog_count,
    )
    x_input = np.array([[row_data[col] for col in feature_cols]], dtype=np.float32)

    predicted_popularity = float(np.clip(model.predict(x_input)[0], 0.0, 100.0))

    # Doprinos svakog obeležja predikciji (SHAP vrednosti ugrađene u XGBoost)
    contribs = model.get_booster().predict(xgb.DMatrix(x_input), pred_contribs=True)[0]
    feature_contributions = {col: float(c) for col, c in zip(feature_cols, contribs[:-1])}

    return {
        "predicted_popularity": round(predicted_popularity, 2),
        "hit_probability": round(hit_probability(predicted_popularity, meta), 4),
        "percentile": round(popularity_percentile(predicted_popularity, meta), 1),
        "artist_name": artist_name,
        "artist_found": artist_found,
        "talent_data_used": talent_data,
        "model_inputs": row_data,
        "feature_contributions": feature_contributions,
        "base_value": float(contribs[-1]),
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

    # 1. Gemini i GPT slušaju fajl paralelno; pad jednog modela ne prekida ostale
    with ThreadPoolExecutor(max_workers=2) as pool:
        gemini_future = pool.submit(Gemini().get_analysis, prompt, audio_file_path=audio_file_path)
        gpt_future = pool.submit(Gpt().get_analysis, prompt, audio_file_path=audio_file_path)
        gemini_analysis = gemini_future.result()
        gpt_analysis = gpt_future.result()

    errors = []
    if _llm_failed(gemini_analysis):
        errors.append("Gemini")
        gemini_analysis_ok = None
    else:
        gemini_analysis_ok = gemini_analysis
    if _llm_failed(gpt_analysis):
        errors.append("GPT")
        gpt_analysis_ok = None
    else:
        gpt_analysis_ok = gpt_analysis

    # 2. DeepSeek meta-arbitraža nad modelima koji su uspeli
    deep_seek_analysis = None
    if gemini_analysis_ok or gpt_analysis_ok:
        deep_seek_analysis = Deepseek().get_analysis(gemini_analysis_ok, gpt_analysis_ok)
        if _llm_failed(deep_seek_analysis):
            errors.append("DeepSeek")

    return {
        "error": f"{', '.join(errors)} API call failed." if errors else None,
        "deep_seek": deep_seek_analysis,
        "gemini": gemini_analysis,
        "gpt": gpt_analysis
    }


def _llm_failed(text) -> bool:
    return not text or text.startswith("Error communicating")


def main():
    print("Executing inference pipeline...")

    audio_file_path = os.getenv("AUDIO_FILE_PATH")
    artist_name = os.getenv("ARTIST_NAME", "Unknown Artist")

    if not audio_file_path or not os.path.exists(audio_file_path):
        print(f"Upozorenje: AUDIO_FILE_PATH nije pronađen ili fajl ne postoji: {audio_file_path}")

    # 1. 30s prozor najveće energije (model je treniran na 30s preview snimcima)
    window_path = str(PROJECT_ROOT / "extracted_30s_window.wav")
    extract_peak_window(audio_file_path, window_path)

    # 2. Ekstrakcija librosa i Audiobox karakteristika za novu pesmu
    audio_data = get_librosa_data(window_path)
    audiobox_data = get_audiobox_data(window_path)

    # 3. XGBoost predikcija sa stvarnim podacima iz TalentAgent-a
    xgboost_prediction = predict_xgboost_popularity(audio_data, artist_name=artist_name, audiobox_features=audiobox_data)
    artist_name_used = xgboost_prediction.get("artist_name", artist_name)
    talent_data_used = xgboost_prediction.get("talent_data_used", {})

    # 4. LLM AI analize
    ai_analysis = get_ai_analysis(str(audio_data), audio_file_path=audio_file_path, artist_name=artist_name_used, talent_data=talent_data_used)

    # Kombinovani podaci
    data = {
        "xgboost_prediction": xgboost_prediction,
        "ai_analysis": ai_analysis
    }

    print(json.dumps(data, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()