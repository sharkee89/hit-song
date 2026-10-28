"""
Trening XGBoost hit modela bez curenja podataka (data leakage).

Pokretanje (iz korena projekta):
    python -m neural.mlp.train_xgboost

Koraci:
  1. Deduplikacija - ista pesma se u datasetu ponavlja za svakog izvođača sa kojim je snimljena,
     zadržavamo jedan red po pesmi (najpopularniji izvođač).
  2. Grupe bez curenja - povezane komponente pesma <-> izvođač, pa ni pesma ni izvođač
     nikada nisu istovremeno u trening i test skupu.
  3. Holdout skup (nikad viđen tokom treninga) za demonstraciju u aplikaciji.
  4. Obeležja: Audiobox + Librosa + talent + top žanrovi + istorija kataloga izvođača.
  5. GridSearch sa GroupKFold unakrsnom validacijom, kalibracija verovatnoće hita.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, cross_val_predict
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.hit_model import (  # noqa: E402
    AUDIOBOX_FEATURES, CATALOG_FEATURES, CATALOG_STATS_PATH, HIT_THRESHOLD, HOLDOUT_PATH,
    META_PATH, MODEL_DIR, MODEL_PATH, TALENT_FEATURES, genre_column, parse_genres,
)

DATASET_PATH = PROJECT_ROOT / "data" / "dataset" / "spotify_tracks_audiobox_librosa.parquet"
CATALOG_PATH = PROJECT_ROOT / "data" / "dataset" / "spotify_tracks_all.parquet"
PREVIEW_DIR = PROJECT_ROOT / "data" / "downloads" / "preview_files"

TOP_GENRES = 40
HOLDOUT_SIZE = 0.15
DEVICE = "cuda"


def _norm(text) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def leak_free_groups(df: pd.DataFrame) -> np.ndarray:
    """Povezane komponente grafa pesma-izvođač (union-find)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for track_id, artist_id in zip(df["id"], df["artist_id"]):
        parent[find("t" + track_id)] = find("a" + artist_id)
    return np.array([find("t" + t) for t in df["id"]])


def local_preview_track_ids(df: pd.DataFrame) -> set[str]:
    """Pesme koje već postoje kao lokalni preview fajlovi (artist-name-popularity.mp3) idu u holdout."""
    keys = {}
    for row in df.itertuples():
        keys.setdefault(_norm(row.artist_name) + _norm(row.name) + str(int(row.popularity)), set()).add(row.id)
    ids = set()
    for f in PREVIEW_DIR.glob("*.mp3"):
        ids |= keys.get(_norm(f.stem), set())
    return ids


def main():
    print(f"Učitavam dataset: {DATASET_PATH}")
    df_all = pd.read_parquet(DATASET_PATH)

    audio_features = AUDIOBOX_FEATURES + [c for c in df_all.columns if c.startswith("librosa_")]
    df_all = df_all.dropna(subset=audio_features + TALENT_FEATURES + ["popularity", "artist_id"]).copy()

    # 1-2. Grupe na punom datasetu (sa duplikatima), pa deduplikacija
    df_all["group"] = leak_free_groups(df_all)
    df = (
        df_all.sort_values("artist_popularity", ascending=False)
        .drop_duplicates("id")
        .reset_index(drop=True)
    )
    print(f"Redova: {len(df_all)} | jedinstvenih pesama: {len(df)} | grupa: {df['group'].nunique()}")

    # 3. Holdout: nasumične grupe + grupe lokalnih preview fajlova
    gss = GroupShuffleSplit(n_splits=1, test_size=HOLDOUT_SIZE, random_state=42)
    _, holdout_idx = next(gss.split(df, groups=df["group"]))
    holdout_groups = set(df.loc[holdout_idx, "group"]) | set(
        df.loc[df["id"].isin(local_preview_track_ids(df)), "group"]
    )
    is_holdout = df["group"].isin(holdout_groups).values
    holdout_ids = set(df.loc[is_holdout, "id"])
    print(f"Trening: {(~is_holdout).sum()} pesama | Holdout (demo, nikad viđen): {is_holdout.sum()} pesama")

    # 4a. Istorija kataloga: zbir/broj popularnosti po izvođaču, bez holdout pesama
    catalog = (
        pd.read_parquet(CATALOG_PATH, columns=["id", "artist_id", "popularity"])
        .dropna()
        .drop_duplicates(["id", "artist_id"])
    )
    catalog = catalog[~catalog["id"].isin(holdout_ids)]
    stats = catalog.groupby("artist_id")["popularity"].agg(["sum", "count"]).reset_index()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(CATALOG_STATS_PATH, index=False)

    # Za trening pesme oduzimamo samu pesmu iz statistike (samo OSTALE pesme izvođača)
    df = df.merge(stats, on="artist_id", how="left").fillna({"sum": 0.0, "count": 0})
    own = df.merge(catalog, on=["id", "artist_id"], how="left", suffixes=("", "_cat"))["popularity_cat"]
    in_catalog = own.notna().values
    other_sum = df["sum"].values - np.where(in_catalog, own.fillna(0).values, 0.0)
    other_count = df["count"].values - in_catalog
    df["catalog_track_count"] = other_count
    df["catalog_mean_popularity"] = np.where(other_count > 0, other_sum / np.maximum(other_count, 1), np.nan)

    # 4b. Top žanrovi (izabrani samo na trening skupu)
    genres = df["artist_genres"].apply(parse_genres)
    top_genres = (
        pd.Series([g for gl in genres[~is_holdout] for g in gl]).value_counts().head(TOP_GENRES).index.tolist()
    )
    for g in top_genres:
        df[genre_column(g)] = genres.apply(lambda gl, g=g: float(g in gl))

    df["followers"] = np.log1p(df["followers"])
    feature_cols = (
        AUDIOBOX_FEATURES + TALENT_FEATURES + CATALOG_FEATURES
        + [c for c in audio_features if c.startswith("librosa_")]
        + [genre_column(g) for g in top_genres]
    )
    print(f"Ukupno obeležja: {len(feature_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    y = df["popularity"].values.astype(np.float32)
    groups = df["group"].values
    X_train, y_train, g_train = X[~is_holdout], y[~is_holdout], groups[~is_holdout]
    X_hold, y_hold = X[is_holdout], y[is_holdout]

    # 5. GridSearch sa GroupKFold (bez curenja i unutar pretrage)
    cv = GroupKFold(n_splits=5)
    param_grid = {
        "n_estimators": [300, 600],
        "learning_rate": [0.03, 0.05],
        "max_depth": [4, 5, 6],
        "min_child_weight": [1, 5],
        "subsample": [0.8],
        "colsample_bytree": [0.7, 0.9],
    }
    print("\nPokrećem GridSearch (GroupKFold, 5 foldova)...")
    grid = GridSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, device=DEVICE),
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        verbose=1,
    )
    grid.fit(X_train, y_train, groups=g_train)
    best_params = grid.best_params_
    print(f"Najbolji parametri: {best_params}")

    # Out-of-fold predikcije -> metrike unakrsne validacije i kalibracija verovatnoće hita
    oof = cross_val_predict(
        XGBRegressor(random_state=42, n_jobs=-1, device=DEVICE, **best_params),
        X_train, y_train, groups=g_train, cv=cv,
    )
    calibrator = LogisticRegression().fit(oof.reshape(-1, 1), (y_train >= HIT_THRESHOLD).astype(int))

    model = grid.best_estimator_
    model.set_params(device="cpu")
    pred_hold = model.predict(X_hold)

    def metrics(y_true, y_pred) -> dict:
        return {
            "r2": round(float(r2_score(y_true, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
            "hit_auc": round(float(roc_auc_score(y_true >= HIT_THRESHOLD, y_pred)), 4),
            "n": int(len(y_true)),
        }

    cv_metrics, holdout_metrics = metrics(y_train, oof), metrics(y_hold, pred_hold)
    print("\n" + "=" * 50)
    print(f"Unakrsna validacija (GroupKFold): {cv_metrics}")
    print(f"Holdout (nikad viđen):            {holdout_metrics}")
    print("=" * 50)

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 15 najvažnijih obeležja:")
    print(importance.head(15).round(4).to_string())

    # Čuvanje modela (nativni XGBoost format), metapodataka i holdout liste
    model.save_model(MODEL_PATH)
    meta = {
        "model": "XGBoost Regressor",
        "feature_cols": feature_cols,
        "top_genres": top_genres,
        "params": best_params,
        "hit_threshold": HIT_THRESHOLD,
        "hit_calibration": {
            "coef": float(calibrator.coef_[0][0]),
            "intercept": float(calibrator.intercept_[0]),
        },
        "popularity_quantiles": np.percentile(y, np.arange(101)).round(3).tolist(),
        # Neutralan profil za izvođača koji nije u bazi (medijana trening skupa)
        "artist_fallback": {
            "artist_popularity": float(np.median(df.loc[~is_holdout, "artist_popularity"])),
            "followers": float(np.expm1(np.median(df.loc[~is_holdout, "followers"]))),
        },
        "metrics": {"cross_validation": cv_metrics, "holdout": holdout_metrics},
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    holdout = df.loc[is_holdout, ["id", "name", "artist_name", "popularity", "preview_url"]].copy()
    holdout["predicted"] = pred_hold.round(2)
    holdout.sort_values("popularity", ascending=False).to_csv(HOLDOUT_PATH, index=False)

    print(f"\nModel i prateći fajlovi sačuvani u: {MODEL_DIR}")


if __name__ == "__main__":
    main()
