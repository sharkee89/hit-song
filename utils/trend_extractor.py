import pandas as pd
import joblib
import os


def run_extraction():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data', 'dataset')

    file_path = os.path.join(data_dir, 'dataset_trend.csv')

    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found!")
        return

    print(f"✅ Loading data for total trend location: {file_path}")
    df = pd.read_csv(file_path)

    audio_features_cols = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]

    required_cols = audio_features_cols + ['key', 'mode']

    initial_count = len(df)
    df_clean = df.dropna(subset=required_cols).copy()

    df_clean = df_clean.drop_duplicates(subset=['track_id'])

    final_count = len(df_clean)
    print(f"🧹 Filtering done:")
    print(f"   - Initial count: {initial_count}")
    print(f"   - Final count: {final_count}")
    print(f"   - Dropped: {initial_count - final_count}")

    if final_count == 0:
        print("❌ Error: After deleting null values, dataset is empty!")
        return

    new_trend_vector = df_clean[audio_features_cols].mean()

    dominant_key = int(df_clean['key'].mode()[0])
    dominant_mode = int(df_clean['mode'].mode()[0])
    mode_name = "Major" if dominant_mode == 1 else "Minor"

    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    dominant_key_name = notes[dominant_key]

    new_trend_metadata = {
        'dominant_key': dominant_key_name,
        'dominant_mode': mode_name,
        'average_popularity': float(df_clean['popularity'].mean()) if 'popularity' in df_clean.columns else 0.0,
        'sample_size': final_count,
        'description': "Trend based on all valid songs from dataset_trend.csv"
    }

    output_path_vector = os.path.join(data_dir, 'trend_vector_v4.joblib')
    output_path_meta = os.path.join(data_dir, 'trend_metadata_v4.joblib')

    joblib.dump(new_trend_vector, output_path_vector)
    joblib.dump(new_trend_metadata, output_path_meta)

    print("\n" + "=" * 30)
    print("🚀 TOTAL TREND SUCCESSFULLY GENERATED")
    print("=" * 30)
    print(f"Songs analyzed: {final_count}")
    print(f"Dominant key and mode name: {dominant_key_name} {mode_name}")
    print(f"Files saved in: {data_dir}")
    print("\nAverage parameters (Total V4):")
    print(new_trend_vector.to_string())


if __name__ == "__main__":
    run_extraction()