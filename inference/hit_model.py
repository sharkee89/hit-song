"""
Zajednička logika XGBoost hit modela - koristi je i trening (neural/mlp/train_xgboost.py)
i inferencija (inference/inference_hit.py), kako bi obeležja bila identično formirana.

Obeležja:
  - Audiobox (PQ, PC, CE, CU) i Librosa obeležja 30s prozora
  - Talent: popularnost izvođača, log(pratioci), top žanrovi izvođača (multi-hot)
  - Istorija kataloga: prosečna popularnost OSTALIH pesama izvođača i njihov broj
"""
import ast
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "models"
MODEL_PATH = MODEL_DIR / "xgboost_hit_model.json"
META_PATH = MODEL_DIR / "xgboost_hit_meta.json"
CATALOG_STATS_PATH = MODEL_DIR / "artist_catalog_stats.parquet"
HOLDOUT_PATH = MODEL_DIR / "holdout_tracks.csv"

HIT_THRESHOLD = 60  # popularnost >= 60 je ~7% pesama u datasetu

AUDIOBOX_FEATURES = ["PQ", "PC", "CE", "CU"]
TALENT_FEATURES = ["artist_popularity", "followers"]
CATALOG_FEATURES = ["catalog_mean_popularity", "catalog_track_count"]


def parse_genres(value) -> list[str]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(g) for g in value]
    if isinstance(value, str) and value.startswith("["):
        try:
            return [str(g) for g in ast.literal_eval(value)]
        except (ValueError, SyntaxError):
            return []
    return []


def genre_column(genre: str) -> str:
    return f"genre_{genre}"


def build_feature_row(
    feature_cols: list[str],
    audio_features: dict,
    artist_popularity: float,
    followers: float,
    genres: list[str],
    catalog_mean: float | None,
    catalog_count: int,
) -> dict:
    """Jedan red obeležja u prostoru modela (followers je već log1p, nepoznato = NaN)."""
    row = {col: np.nan for col in feature_cols}
    for k, v in audio_features.items():
        if k in row and v is not None:
            row[k] = float(v)
    row["artist_popularity"] = float(artist_popularity)
    row["followers"] = math.log1p(float(followers))
    for col in feature_cols:
        if col.startswith("genre_"):
            row[col] = 0.0
    for g in genres:
        if genre_column(g) in row:
            row[genre_column(g)] = 1.0
    row["catalog_mean_popularity"] = np.nan if catalog_mean is None else float(catalog_mean)
    row["catalog_track_count"] = float(catalog_count)
    return row


@lru_cache(maxsize=1)
def load_hit_model() -> tuple[XGBRegressor, dict]:
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    model.set_params(device="cpu")
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


@lru_cache(maxsize=1)
def _catalog_stats() -> pd.DataFrame:
    return pd.read_parquet(CATALOG_STATS_PATH).set_index("artist_id")


def artist_catalog(artist_id: str) -> tuple[float | None, int]:
    """Prosečna popularnost i broj pesama izvođača u katalogu (bez pesama iz holdout skupa)."""
    stats = _catalog_stats()
    if artist_id not in stats.index:
        return None, 0
    row = stats.loc[artist_id]
    count = int(row["count"])
    return (float(row["sum"]) / count if count else None), count


def hit_probability(prediction: float, meta: dict) -> float:
    """Kalibrisana verovatnoća da je popularnost >= HIT_THRESHOLD (logistička regresija nad OOF predikcijama)."""
    cal = meta["hit_calibration"]
    return 1.0 / (1.0 + math.exp(-(cal["coef"] * prediction + cal["intercept"])))


def popularity_percentile(prediction: float, meta: dict) -> float:
    """Procenat pesama u datasetu čija je stvarna popularnost niža od predviđene."""
    quantiles = meta["popularity_quantiles"]  # 101 vrednost: 0., 1., ... 100. percentil
    return float(np.interp(prediction, quantiles, np.arange(101), left=0.0, right=100.0))
