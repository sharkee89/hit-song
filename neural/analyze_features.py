import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os


def analyze_dataset(file_path):
    if not os.path.exists(file_path):
        alternative_path = os.path.join('..', 'data', 'dataset', 'dataset_enriched.csv')
        if os.path.exists(alternative_path):
            file_path = alternative_path
        else:
            print(f"❌ Error: File {file_path} not found.")
            return

    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df)} tracks.")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    df = df.dropna(subset=['popularity'])

    le = LabelEncoder()
    if 'track_genre' in df.columns:
        df['track_genre'] = le.fit_transform(df['track_genre'].astype(str))

    features = [
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'duration_ms', 'track_genre',
        'trend_alignment'
    ]

    available_features = [f for f in features if f in df.columns]

    X = df[available_features]
    y = df['popularity']

    print("🚀 Training Random Forest Regressor (this may take a moment)...")
    model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': available_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n📊 FEATURE IMPORTANCE RANKING:")
    print("-" * 35)
    print(feature_importance_df.to_string(index=False))
    print("-" * 35)

    return feature_importance_df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset_path = os.path.join(current_dir, '..', 'data', 'dataset', 'dataset_enriched.csv')

    analyze_dataset(default_dataset_path)