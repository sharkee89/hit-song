import json
import os
from dotenv import load_dotenv
from audio.ai.deepseek import Deepseek
from audio.ai.gemini import Gemini
from audio.ai.gpt import Gpt
from audio.utils.audio_utils import get_librosa_data

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_ai_analysis(features_str: str, audio_file_path: str = None) -> dict:
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
        f"Guidelines for a balanced, genre-agnostic analysis:\n"
        f"1. Contextualize the metrics and actual sound: Compare what you hear and the track's features against the appropriate genre baseline above.\n"
        f"2. Evaluate objectively: Do not penalize a track solely for moderate energy or specific spectral values if it sounds great and fits its target market.\n"
        f"3. Provide a detailed, professional assessment covering:\n"
        f"   - Commercial and streaming potential.\n"
        f"   - Most accurate genre and playlist fit.\n"
        f"   - Market placement and target audience.\n"
        f"   - Quality of audio.\n"
        f"   - Actionable production or mixing advice.\n"
    )
    print(prompt)

    deep_seek_analysis = Deepseek().get_analysis(prompt)

    gemini_analysis = Gemini().get_analysis(prompt, audio_file_path=audio_file_path)

    gpt_analysis = Gpt().get_analysis(prompt)

    return {
        "deep_seek": deep_seek_analysis,
        "gemini": gemini_analysis,
        "gpt": gpt_analysis
    }


def main():
    print("Executing")

    audio_file_path = os.getenv("AUDIO_FILE_PATH")

    if not audio_file_path or not os.path.exists(audio_file_path):
        print(
            f"Upozorenje: AUDIO_FILE_PATH nije pronađen u .env fajlu ili fajl ne postoji na putanji: {audio_file_path}")

    audio_data = get_librosa_data()
    ai_analysis = get_ai_analysis(str(audio_data), audio_file_path=audio_file_path)

    data = {
        "value": audio_data,
        "ai": ai_analysis
    }
    print(json.dumps(data, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()