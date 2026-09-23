import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler


class AudioAgentFeatureExtractor:
    """
    Data processor configured for loading dataset attributes and simulating
    low-level DSP feature mapping for Random Forest training.
    """

    def __init__(self, csv_file: str):
        df = pd.read_csv(csv_file).dropna(subset=[
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'duration_ms', 'popularity', 'daily_rank'
        ])

        base_feature_cols = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'duration_ms'
        ]

        x_base = df[base_feature_cols].values
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x_base)

        # Expanding 12 base features to 24-dimensional feature vector simulating AudioAgent DSP outputs
        self.features = np.hstack([x_scaled, x_scaled[:, :12]])  # 24 dimensions

        # Define hit criteria based on the new dataset: top 10 daily rank or popularity > 70
        self.targets = ((df['daily_rank'] <= 10) | (df['popularity'] > 70)).astype(int).values

        num_hits = int(np.sum(self.targets))
        total_tracks = len(self.targets)
        hit_percentage = (num_hits / total_tracks) * 100 if total_tracks > 0 else 0
        print(f"Dataset loaded. Total tracks: {total_tracks} | Hits: {num_hits} ({hit_percentage:.2f}%)")


if __name__ == '__main__':
    # Load dataset with relative path check for the new dataset
    dataset_path = '../data/old/universal_top_spotify_songs.csv'
    dataset = AudioAgentFeatureExtractor(dataset_path)

    X = dataset.features
    y = dataset.targets

    # Stratified train-validation split (80% / 20%) to preserve class ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize Random Forest Classifier with balanced class weights to address class imbalance
    print("Training Random Forest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )

    # Fit the model
    rf_model.fit(X_train, y_train)

    # Save trained model and scaler to disk using joblib
    joblib.dump(rf_model, '../joblib/hit_random_forest.joblib')
    joblib.dump(dataset.scaler, '../joblib/scaler.joblib')
    print("💾 Model and scaler successfully saved to disk as .joblib files.")

    # Evaluation phase
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Test prediction on Blinding Lights sample values to verify pipeline integration
    print("\n--- Sanity Check: Blinding Lights Verification ---")
    sample_raw = np.array([[
        0.790304, 0.374006, 5.0, -13.6694, 0.0,
        0.413122, 0.400000, 0.218384, 0.218384, 0.346687,
        172.265625, 262524.807256
    ]])
    sample_scaled = dataset.scaler.transform(sample_raw)
    sample_vector = np.hstack([sample_scaled, sample_scaled[:, :12]])

    bl_pred = rf_model.predict(sample_vector)
    bl_proba = rf_model.predict_proba(sample_vector)

    print(f"Blinding Lights Status: {'HIT' if bl_pred[0] == 1 else 'NIJE HIT'}")
    print(f"Blinding Lights Hit Probability: {bl_proba[0][1] * 100:.2f}%")

    print("Random Forest training and evaluation completed successfully.")