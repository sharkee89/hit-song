import os
import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from audio_quality.audio_constants import DTYPE_TO_BYTES

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
