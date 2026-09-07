import pandas as pd
import numpy as np
import os
import json


def generate_data(n_samples=2000):
    """
    Generates training data focused on popular genres to better
    align with Trend Agent logic.
    """
    # 1. Setup paths correctly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    file_path = os.path.join(parent_dir ,'data', 'dataset', 'dataset_enriched.csv')

    if not os.path.exists(file_path):
        print(f"❌ Error: enriched dataset not found at {file_path}")
        print("Please run enrich_dataset_with_trend.py first!")
        return

    # 2. Load and filter data
    df = pd.read_csv(file_path)

    # Filtering for genres where the Trend Vector is most relevant
    # You can expand this list based on your specific trend focus
    target_genres = ['pop', 'hip-hop', 'dance', 'r&b', 'reggaeton', 'indie-pop']

    # Filtering (case-insensitive)
    mask = df['track_genre'].str.lower().isin(target_genres)
    df_filtered = df[mask].copy()

    if len(df_filtered) == 0:
        print("⚠️ Warning: No tracks found for target genres. Using full dataset instead.")
        df_filtered = df.copy()
    else:
        print(f"🎯 Filtered dataset to {len(df_filtered)} tracks within target genres.")

    # Calculate genre score for the Context Agent (on the filtered set)
    genre_means = df_filtered.groupby('track_genre')['popularity'].mean() / 100
    df_filtered['track_genre_score'] = df_filtered['track_genre'].map(genre_means)

    # Take sample from the filtered list
    df_sample = df_filtered.sample(min(n_samples, len(df_filtered)))

    training_set = []

    for _, row in df_sample.iterrows():
        # --- AUDIO AGENT ---
        norm_loudness = np.clip((row['loudness'] + 60) / 60, 0, 1)
        audio_sig = (row['energy'] + norm_loudness + (1 - row['acousticness']) + (1 - row['instrumentalness'])) / 4

        # --- TREND AGENT ---
        # Pre-calculated alignment from NeuralTrendAgent logic
        trend_sig = row['trend_alignment']

        # --- CONTEXT AGENT ---
        explicit_val = 1.0 if row['explicit'] else 0.0
        context_sig = (row['track_genre_score'] * 0.8) + (explicit_val * 0.2)

        # --- VIRAL AGENT ---
        # Viral signal updated to focus on danceability and energy (typical for pop trends)
        viral_sig = (row['danceability'] + row['energy']) / 2

        # --- TARGET ---
        target = row['popularity'] / 100

        training_set.append({
            "signals": [
                float(np.clip(audio_sig, 0, 1)),
                float(np.clip(context_sig, 0, 1)),
                float(np.clip(trend_sig, 0, 1)),
                float(np.clip(viral_sig, 0, 1))
            ],
            "target": float(np.clip(target, 0, 1))
        })

    # 3. Save the training data
    output_path = os.path.join(current_dir, 'training_data.json')
    with open(output_path, 'w') as f:
        json.dump(training_set, f)

    print(f"✅ Success! Generated {len(training_set)} samples from POP-focused data.")
    print(f"📍 Training data saved at: {output_path}")


if __name__ == "__main__":
    generate_data(2000)