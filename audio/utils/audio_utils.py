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

def get_librosa_data():
    audio_path_env = os.getenv("AUDIO_FILE_PATH")
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

    print(f"Getting Librosa data: {target_path}")
    features_dict = {}

    try:
        y, sr = librosa.load(str(target_path.resolve()), sr=22050, duration=30)
        if len(y) == 0:
            return {}

        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features_dict["librosa_tempo"] = float(np.atleast_1d(tempo)[0])
        except Exception:
            features_dict["librosa_tempo"] = np.nan

        try:
            features_dict["librosa_energy"] = float(np.mean(librosa.feature.rms(y=y)))
            features_dict["librosa_spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            features_dict["librosa_spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            features_dict["librosa_zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            features_dict["librosa_spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            features_dict["librosa_spectral_flatness"] = float(np.mean(librosa.feature.spectral_flatness(y=y)))
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