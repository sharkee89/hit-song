"""
Tanak sloj iznad postojećeg istraživačkog koda koji ga prilagođava za HTTP API.
Sva logika predikcije i LLM analize dolazi iz:
  - audio/utils/audio_utils.py      (30s prozor, Librosa i Audiobox obeležja)
  - inference/inference_hit.py      (TalentAgent + XGBoost, Gemini + GPT + DeepSeek)
"""
import ast
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from audio.utils.audio_utils import extract_peak_window, get_audiobox_data, get_librosa_data
from inference.hit_model import load_hit_model
from inference.inference_hit import get_ai_analysis, predict_xgboost_popularity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTISTS_CSV = PROJECT_ROOT / "data" / "dataset" / "spotify_artists.csv"

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

FEATURE_LABELS = {
    "PQ": "Production Quality",
    "PC": "Production Complexity",
    "CE": "Content Enjoyment",
    "CU": "Content Usefulness",
    "artist_popularity": "Artist popularity",
    "followers": "Followers",
    "catalog_mean_popularity": "Artist track record",
    "catalog_track_count": "Artist catalog size",
    "librosa_tempo": "Tempo",
    "librosa_energy": "Energy (RMS)",
    "librosa_spectral_centroid": "Spectral centroid",
    "librosa_spectral_rolloff": "Spectral rolloff",
    "librosa_zcr": "Zero-crossing rate",
    "librosa_spectral_bandwidth": "Spectral bandwidth",
    "librosa_spectral_flatness": "Spectral flatness",
    **{f"librosa_mfcc_{i}": f"MFCC {i}" for i in range(1, 14)},
    **{f"librosa_chroma_{i + 1}": f"Chroma {p}" for i, p in enumerate(PITCH_CLASSES)},
}

# Iste grupe kao u neural/mlp/get_audio_features_for_example_song.py
FEATURE_GROUPS = {
    "Production agent": ["PQ", "PC", "CE", "CU"],
    "Talent agent": ["artist_popularity", "followers", "catalog_mean_popularity", "catalog_track_count"],
}

LLM_METRIC_LABELS = {
    "prediction_grade": "Hit potential",
    "production_quality": "Production quality",
    "audio_quality": "Audio quality",
    "genre_placement": "Genre placement",
    "possible_improvements": "Release readiness",
    "prediction_detail_analysis": "Commercial outlook",
    "analysis": "Overall analysis",
}


def _json_number(value):
    """NaN (nepoznata vrednost) nije validan JSON - šaljemo null."""
    return None if value is None or math.isnan(value) else value


def feature_label(feature: str) -> str:
    if feature.startswith("genre_"):
        return f"Genre: {feature[len('genre_'):]}"
    return FEATURE_LABELS.get(feature, feature)


def feature_group(feature: str) -> str:
    if feature.startswith("genre_"):
        return "Talent agent"
    for group, cols in FEATURE_GROUPS.items():
        if feature in cols:
            return group
    return "Audio agent"


@lru_cache(maxsize=1)
def model_info() -> dict:
    _, meta = load_hit_model()
    return {
        "model": meta["model"],
        "n_features": len(meta["feature_cols"]),
        "params": meta["params"],
        "metrics": meta["metrics"],
        "hit_threshold": meta["hit_threshold"],
        "llms": ["Gemini", "GPT (audio)", "DeepSeek (consensus)"],
    }


@lru_cache(maxsize=1)
def _artists_df() -> pd.DataFrame:
    df = pd.read_csv(ARTISTS_CSV, usecols=["name", "artist_popularity", "followers"])
    df = df.dropna(subset=["name"]).drop_duplicates("name")
    df["name_lower"] = df["name"].str.lower()
    return df.sort_values("artist_popularity", ascending=False)


def search_artists(query: str, limit: int = 8) -> list[dict]:
    q = query.strip().lower()
    if len(q) < 2:
        return []
    df = _artists_df()
    starts = df[df["name_lower"].str.startswith(q, na=False)]
    contains = df[df["name_lower"].str.contains(q, regex=False, na=False)]
    hits = pd.concat([starts, contains]).drop_duplicates("name").head(limit)
    return [
        {"name": r.name, "artist_popularity": float(r.artist_popularity), "followers": int(r.followers)}
        for r in hits.itertuples()
    ]


def popularity_tier(score: float) -> str:
    if score >= 70:
        return "Hit potential"
    if score >= 50:
        return "Strong performer"
    if score >= 30:
        return "Moderate reach"
    return "Niche / low reach"


def _artist_summary(talent_data: dict) -> dict:
    spotify = talent_data.get("spotify_data", {}) or {}
    mb = talent_data.get("musicbrainz_data", {}) or {}
    genres = spotify.get("genres", [])
    if isinstance(genres, str):
        try:
            genres = ast.literal_eval(genres)
        except (ValueError, SyntaxError):
            genres = []
    return {
        "name": talent_data.get("name"),
        "found_in_dataset": bool(talent_data.get("found")),
        "artist_popularity": spotify.get("artist_popularity"),
        "followers": spotify.get("followers"),
        "genres": genres[:6],
        "known_tracks": len(spotify.get("tracks", []) or []),
        "country": mb.get("country"),
        "artist_type": mb.get("artist_type"),
    }


def run_prediction(audio_path: str, window_path: str, artist_name: str) -> dict:
    """Brza faza: 30s prozor -> Librosa + Audiobox -> TalentAgent -> XGBoost."""
    window = extract_peak_window(audio_path, window_path)
    librosa_features = get_librosa_data(window_path)
    audiobox_features = get_audiobox_data(window_path)

    xgb_result = predict_xgboost_popularity(
        librosa_features, artist_name=artist_name, audiobox_features=audiobox_features
    )
    if "error" in xgb_result:
        raise RuntimeError(xgb_result["error"])

    contributions = sorted(
        (
            {
                "feature": f,
                "label": feature_label(f),
                "group": feature_group(f),
                "value": _json_number(xgb_result["model_inputs"].get(f)),
                "contribution": round(c, 3),
            }
            for f, c in xgb_result["feature_contributions"].items()
        ),
        key=lambda item: -abs(item["contribution"]),
    )
    group_totals = {}
    for item in contributions:
        group_totals[item["group"]] = round(group_totals.get(item["group"], 0.0) + item["contribution"], 3)

    score = xgb_result["predicted_popularity"]
    return {
        "prediction": {
            "popularity": score,
            "tier": popularity_tier(score),
            "base_value": round(xgb_result["base_value"], 2),
            "hit_probability": xgb_result["hit_probability"],
            "hit_threshold": model_info()["hit_threshold"],
            "percentile": xgb_result["percentile"],
            "artist_found": xgb_result["artist_found"],
        },
        "window": window,
        "artist": _artist_summary(xgb_result["talent_data_used"]),
        "audiobox": {k: round(v, 3) for k, v in audiobox_features.items()},
        "audio_features": {
            FEATURE_LABELS[k]: round(v, 4)
            for k, v in librosa_features.items()
            if not k.startswith(("librosa_mfcc", "librosa_chroma"))
        },
        "chroma": [round(librosa_features.get(f"librosa_chroma_{i}", 0.0), 4) for i in range(1, 13)],
        "contributions": contributions[:12],
        "group_contributions": group_totals,
        # Interno stanje potrebno za LLM fazu (ne šalje se klijentu)
        "_librosa_features": librosa_features,
        "_talent_data": xgb_result["talent_data_used"],
    }


def _parse_llm_json(text):
    if not text or not isinstance(text, str):
        return None
    cleaned = re.sub(r"```(?:json)?", "", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None


def _normalize_llm_output(text) -> dict:
    """Gemini/GPT/DeepSeek vraćaju različite ključeve (numeric, value, numeric_value) - ujednačavamo ih."""
    parsed = _parse_llm_json(text)
    if parsed is None:
        return {"metrics": [], "raw": text, "parsed": False}

    metrics = []
    for key, item in parsed.items():
        norm_key = key.strip().lower().replace(" ", "_")
        if isinstance(item, dict):
            score = next(
                (item[k] for k in ("numeric_value", "value", "numeric", "score") if k in item),
                None,
            )
            analysis = item.get("analysis", "")
        else:
            score, analysis = None, str(item)
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        metrics.append({
            "key": norm_key,
            "label": LLM_METRIC_LABELS.get(norm_key, norm_key.replace("_", " ").capitalize()),
            "score": score,
            "analysis": analysis,
        })
    return {"metrics": metrics, "raw": text, "parsed": True}


def run_llm_analysis(audio_path: str, artist_name: str, librosa_features: dict, talent_data: dict) -> dict:
    """Spora faza: Gemini + GPT slušaju fajl, DeepSeek pravi konsenzus nad modelima koji su uspeli."""
    result = get_ai_analysis(
        str(librosa_features),
        audio_file_path=audio_path,
        artist_name=artist_name,
        talent_data=talent_data,
    )
    reports = {
        "deepseek": result.get("deep_seek"),
        "gemini": result.get("gemini"),
        "gpt": result.get("gpt"),
    }
    models = {}
    for name, text in reports.items():
        failed = not text or text.startswith("Error communicating")
        models[name] = {
            "ok": not failed,
            "error": (text or "Not run (no upstream model succeeded).") if failed else None,
            **(_normalize_llm_output(text) if not failed else {"metrics": [], "raw": None, "parsed": False}),
        }
    return {"error": result.get("error"), "models": models}
