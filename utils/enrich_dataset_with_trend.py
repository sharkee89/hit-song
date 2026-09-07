import pandas as pd
import numpy as np
import joblib
import os


def enrich_dataset_with_trend():
    # 1. SETUP ABSOLUTE PATHS
    # This finds the directory where THIS script is located (e.g., .../utils/)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # This points to the parent directory (the main project folder)
    root_dir = os.path.dirname(current_script_dir)
    dataset_dir = os.path.join(root_dir, 'data', 'dataset')

    # Construct absolute paths to your files
    vector_path = os.path.join(root_dir, 'trend_vector_v4.joblib')
    metadata_path = os.path.join(root_dir, 'trend_metadata_v4.joblib')
    dataset_path = os.path.join(dataset_dir, 'dataset_enriched.csv')
    output_path = os.path.join(dataset_dir, 'dataset_enriched.csv')

    # 2. LOAD RESOURCES
    try:
        trend_vector = joblib.load(vector_path)
        trend_metadata = joblib.load(metadata_path)
        print(f"✅ Trend resources loaded from {root_dir}")
    except Exception as e:
        print(f"❌ Error: Trend files missing at {vector_path}! Details: {e}")
        return

    if not os.path.exists(dataset_path):
        print(f"❌ Error: dataset.csv not found at {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    print(f"📊 Processing {len(df)} tracks...")

    # Identify common columns
    common_cols = [c for c in trend_vector.index if c in df.columns]
    print(f"🔗 Aligning on features: {common_cols}")

    # 3. VECTORIZED CALCULATION
    audio_values = df[common_cols].values
    trend_values = trend_vector[common_cols].values

    # Euclidean distance
    distances = np.linalg.norm(audio_values - trend_values, axis=1)

    # Apply exponential alignment formula
    df['trend_alignment'] = np.exp(-distances / 50)

    # 4. SAVE ENRICHED DATASET
    df.to_csv(output_path, index=False)
    print(f"✅ Success! 'trend_alignment' column added to: {output_path}")


if __name__ == "__main__":
    enrich_dataset_with_trend()