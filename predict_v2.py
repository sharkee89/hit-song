import sys
import librosa
import librosa.feature
import librosa.display
import numpy as np
import joblib
import pandas as pd
import json
import warnings
import musicbrainzngs

warnings.filterwarnings("ignore")

# Konfiguracija MusicBrainz-a
musicbrainzngs.set_useragent("fon_phd_research", "1.0", "sn20245085@student.fon.bg.ac.rs")

# 1. Učitavanje artefakata
try:
    model = joblib.load('music_popularity_model_v2.joblib')
    model_columns = joblib.load('model_columns_v2.joblib')
    trend_vector = joblib.load('trend_vector_v2.joblib')
    # NOVO: Učitavanje metapodataka o skali trenda
    trend_metadata = joblib.load('trend_metadata_v2.joblib')
except Exception as e:
    print(json.dumps({"status": "error", "message": f"Greška pri učitavanju modela: {str(e)}"}))
    sys.exit(1)


def get_artist_context(artist_name):
    if not artist_name or artist_name.lower() == "unknown":
        return {"authority_score": 1.0, "area": "Unknown", "type": "Person"}

    try:
        result = musicbrainzngs.search_artists(artist=artist_name, limit=1)
        if result['artist-list']:
            artist = result['artist-list'][0]
            tag_count = len(artist.get('tag-list', []))
            authority = 1.0 + (tag_count * 0.05)
            return {
                "authority_score": round(min(authority, 2.0), 2),
                "area": artist.get('area', {}).get('name', 'Unknown'),
                "type": artist.get('type', 'Person')
            }
    except:
        pass
    return {"authority_score": 1.0, "area": "Unknown", "type": "Person"}


def  analyze_audio(file_path):
    """Vrši duboku MIR analizu audio fajla"""
    y, sr = librosa.load(file_path, duration=60)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    energy_val = np.mean(rms)

    # Chroma Analysis za ključ i skalu
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_idx = np.argmax(mean_chroma)

    detected_key = notes[key_idx]
    major_score = mean_chroma[(key_idx + 4) % 12]
    minor_score = mean_chroma[(key_idx + 3) % 12]

    detected_mode_val = 1 if major_score >= minor_score else 0
    mode_name = "Major" if detected_mode_val == 1 else "Minor"

    audio_features = {
        'bpm': float(tempo),
        'energy_%': min(float(energy_val * 500), 100),
        'danceability_%': min(float(energy_val * 450), 100),
        'acousticness_%': max(100 - float(energy_val * 600), 0),
        'valence_%': 50.0,
        'mode_binary': detected_mode_val,
        'detected_key_name': detected_key,
        'detected_mode_name': mode_name
    }
    return audio_features


def analyze_segments(file_path, window_size_sec=5, hop_length_sec=2):
    y, sr = librosa.load(file_path, duration=120)

    duration = librosa.get_duration(y=y, sr=sr)
    step_samples = hop_length_sec * sr
    window_samples = window_size_sec * sr

    heatmap_data = []

    for start in range(0, len(y) - window_samples, step_samples):
        end = start + window_samples
        y_segment = y[start:end]

        rms = librosa.feature.rms(y=y_segment)
        energy_val = np.mean(rms)

        segment_potential = min(float(energy_val * 800), 100)

        timestamp = start / sr
        heatmap_data.append({
            "time_sec": round(timestamp, 2),
            "potential": round(segment_potential, 1)
        })

    return heatmap_data


def identify_representative_segments(file_path):
    """
    Faza 1: Strukturni Agent koji pronalazi delove pesme.
    Deli pesmu na segmente na osnovu naglih promena u energiji i spektru.
    """
    y, sr = librosa.load(file_path, duration=120)

    # Koristimo onset strength da vidimo gde su "udarne" promene (prelazi)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Detektujemo tačke promena (backtrack=True pomaže da nađemo tačan početak beata)
    bounds = librosa.segment.agglomerative(librosa.feature.melspectrogram(y=y, sr=sr), 5)
    bound_times = librosa.frames_to_time(bounds, sr=sr)

    segments = []
    for i in range(len(bound_times) - 1):
        start_t = bound_times[i]
        end_t = bound_times[i + 1]

        y_seg = y[int(start_t * sr):int(end_t * sr)]

        rms = librosa.feature.rms(y=y_seg)
        energy = np.mean(rms)

        centroid = np.mean(librosa.feature.spectral_centroid(y=y_seg, sr=sr))

        label = "Stable/Verse"
        if energy > np.mean(librosa.feature.rms(y=y)) * 1.2:
            label = "High Energy/Chorus"
        elif centroid > 3000:
            label = "Vocal Focused"

        segments.append({
            "id": i + 1,
            "start": round(start_t, 2),
            "end": round(end_t, 2),
            "duration": round(end_t - start_t, 2),
            "label_candidate": label,
            "energy_score": float(energy),
            "vocal_presence_index": float(centroid / 5000)
        })

    return segments


def main(file_path, artist_name="Unknown"):
    audio_data = analyze_audio(file_path)

    heatmap_results = analyze_segments(file_path) # Tvoj originalni 5s prozor
    representative_parts = identify_representative_segments(file_path) # PROFESOROV PREDLOG

    context = get_artist_context(artist_name)
    authority = context['authority_score']

    common_cols = [c for c in trend_vector.index if c in audio_data]
    audio_values = np.array([audio_data[c] for c in common_cols])
    trend_values = trend_vector[common_cols].values
    dist_from_trend = np.linalg.norm(audio_values - trend_values)

    matching_score = float(np.clip(100 - (dist_from_trend * 1.2), 5, 100))

    dom_key = trend_metadata['dominant_key']
    dom_mode = trend_metadata['dominant_mode']
    harmonic_match = False
    harmonic_bonus = 1.0

    if audio_data['detected_key_name'] == dom_key and audio_data['detected_mode_name'] == dom_mode:
        harmonic_match = True
        harmonic_bonus = 1.15

    penalty = (matching_score / 100) ** (2.0 / authority)

    input_df = pd.DataFrame([audio_data])
    key_col = f"key_{audio_data['detected_key_name']}"
    if key_col in model_columns:
        input_df[key_col] = 1

    for col in model_columns:
        if col not in input_df.columns: input_df[col] = 0

    input_df['in_spotify_playlists'] = 1200 * authority
    input_df['dist_from_trend'] = dist_from_trend
    input_df = input_df[model_columns]

    pred_log = model.predict(input_df)
    raw_val = int(np.expm1(pred_log)[0])
    final_val = int(raw_val * penalty * harmonic_bonus)

    output = {
        "status": "success",
        "analysis": {
            "detected_scale": f"{audio_data['detected_key_name']} {audio_data['detected_mode_name']}",
            "trend_dominant_scale": f"{dom_key} {dom_mode}",
            "harmonic_match": harmonic_match,
            "bpm": round(audio_data['bpm'], 1),
            "energy": f"{round(audio_data['energy_%'], 1)}%"
        },
        "artist_info": context,
        "prediction": {
            "trend_match_score": f"{round(matching_score, 1)}%",
            "potential_streams": final_val,
            "authority_multiplier": authority,
            "harmonic_bonus_applied": harmonic_match,
            "popularity_heatmap": heatmap_results,
            "representative_segments": representative_parts
        }
    }
    print(json.dumps(output, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    artist = sys.argv[2] if len(sys.argv) > 2 else "Unknown"
    if path:
        main(path, artist)