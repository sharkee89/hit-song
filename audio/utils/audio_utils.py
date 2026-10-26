import librosa
import numpy as np
import os
import requests
import soundfile as sf
import torch
from audio_quality.audio_constants import DTYPE_TO_BYTES
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def load_audio_file(file_path: str = None):
  target_path = file_path or os.getenv("AUDIO_FILE_PATH")
  if not target_path:
    raise ValueError(
        "No audio file provided and AUDIO_FILE_PATH is not set as environment"
        " variable"
    )
  try:
    data, sample_rate = sf.read(target_path, always_2d=True)
    audio_data = data.T.astype(np.float32)
    return audio_data, sample_rate
  except Exception as e:
    raise RuntimeError(f"Failed to load audio file '{target_path}': {e}")


def get_duration(sr, audio_shape):
  num_samples = audio_shape[1]
  return num_samples / sr


def get_size(audio_shape, audio_dtype):
    channels, num_samples = audio_shape
    dtype_str = str(audio_dtype)
    bytes_per_sample = DTYPE_TO_BYTES.get(dtype_str, 4)

    return channels * num_samples * bytes_per_sample


def rms(waveform: torch.Tensor, frame_size: int = 2048):
  channels, num_samples = waveform.shape
  num_frames = num_samples // frame_size
  waveform = waveform[:, : num_frames * frame_size]
  frames = waveform.view(channels, num_frames, frame_size)
  return torch.sqrt(torch.mean(frames ** 2, dim=2))


def extract_peak_window(source_path: str, output_path: str, duration: float = 30.0) -> dict:
    """
    Izdvaja 30-sekundni prozor sa najvećom RMS energijom (isti pristup kao u
    testing/test_audio_agent.py) kako bi ulaz odgovarao 30s Spotify preview
    snimcima na kojima su trenirani XGBoost, Audiobox i Librosa obeležja.
    """
    y, sr = librosa.load(source_path, sr=None, mono=True)
    total_duration = len(y) / sr
    target_samples = int(duration * sr)

    start_sample = 0
    if len(y) > target_samples:
        rms_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=sr)[0]
        window = int(duration)
        n_windows = (len(y) - target_samples) // sr + 1
        window_means = [np.mean(rms_frames[i:i + window]) for i in range(n_windows)]
        start_sample = int(np.argmax(window_means)) * sr

    segment = y[start_sample:start_sample + target_samples]
    sf.write(output_path, segment, sr)

    return {
        "start_sec": round(start_sample / sr, 2),
        "end_sec": round((start_sample + len(segment)) / sr, 2),
        "total_duration_sec": round(total_duration, 2),
    }


def get_librosa_data(audio_path: str = None):
    audio_path_env = audio_path or os.getenv("AUDIO_FILE_PATH")
    if not audio_path_env:
        raise ValueError("Error: AUDIO_FILE_PATH not found in .env file!")
    audio_path = Path(audio_path_env)
    target_path = audio_path
    temp_downloaded = False

    if audio_path_env.startswith("http"):
        try:
            print(f"Skidam audio sa URL-a: {audio_path_env}")
            response = requests.get(audio_path_env, timeout=15)
            response.raise_for_status()

            PROJECT_ROOT = Path(__file__).resolve().parent.parent
            target_path = PROJECT_ROOT / "data" / "test" / "temp_inference_audio.mp3"
            target_path.parent.mkdir(parents=True, exist_ok=True)

            with open(target_path, "wb") as f:
                f.write(response.content)
            temp_downloaded = True
        except Exception as e:
            print(f"Greška pri preuzimanju audio fajla sa URL-a: {e}")
            return {}

    if not target_path.exists():
        raise FileNotFoundError(f"Audio fajl nije pronađen na putanji: {target_path}")

    print(f"Getting Librosa data for entire track: {target_path}")
    features_dict = {}

    try:
        # Učitavamo CELU pesmu (duration=None) umesto samo prvih 30 sekundi
        y, sr = librosa.load(str(target_path.resolve()), sr=22050, duration=None)
        if len(y) == 0:
            return {}

        try:
            # Robustnija procena tempa preko dinamičkog praćenja ritma
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features_dict["librosa_tempo"] = float(np.atleast_1d(tempo)[0])
        except Exception:
            features_dict["librosa_tempo"] = np.nan

        try:
            # Globalne statistike za celu pesmu (mean + opcionalno std za stabilnost)
            rms = librosa.feature.rms(y=y)
            features_dict["librosa_energy"] = float(np.mean(rms))

            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features_dict["librosa_spectral_centroid"] = float(np.mean(centroid))

            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features_dict["librosa_spectral_rolloff"] = float(np.mean(rolloff))

            zcr = librosa.feature.zero_crossing_rate(y)
            features_dict["librosa_zcr"] = float(np.mean(zcr))

            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features_dict["librosa_spectral_bandwidth"] = float(np.mean(bandwidth))

            flatness = librosa.feature.spectral_flatness(y=y)
            features_dict["librosa_spectral_flatness"] = float(np.mean(flatness))
        except Exception:
            pass

        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features_dict[f"librosa_mfcc_{i + 1}"] = float(np.mean(mfccs[i]))
        except Exception:
            for i in range(13):
                features_dict[f"librosa_mfcc_{i + 1}"] = np.nan

        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            for i in range(12):
                features_dict[f"librosa_chroma_{i + 1}"] = float(np.mean(chroma[i]))
        except Exception:
            for i in range(12):
                features_dict[f"librosa_chroma_{i + 1}"] = np.nan

    except Exception as e:
        print(f"Greška tokom Librosa obrade: {e}")

    if temp_downloaded and target_path.exists():
        target_path.unlink()

    return features_dict

_AUDIOBOX_PREDICTOR = None


def get_audiobox_data(audio_path: str) -> dict:
    """
    Meta Audiobox Aesthetics ocene (PQ, PC, CE, CU) - isti model koji je korišćen
    u data/edit/build_audiobox_dataset.py za kreiranje trening skupa.
    """
    global _AUDIOBOX_PREDICTOR
    if _AUDIOBOX_PREDICTOR is None:
        from audiobox_aesthetics.infer import initialize_predictor
        _AUDIOBOX_PREDICTOR = initialize_predictor()

    preds = _AUDIOBOX_PREDICTOR.forward([{"path": str(Path(audio_path).resolve())}])
    res = preds[0] if preds else {}
    return {k: float(res[k]) for k in ("PQ", "PC", "CE", "CU") if res.get(k) is not None}
