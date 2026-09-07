import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

from audio.utils.audio_utils import get_duration, get_size, load_audio_file
from audio_quality.features.rms.rms_feature import extract_rms_feature


class AudioQualityService:

  def get_data(self):
    audio, sr = load_audio_file()
    audio_tensor = torch.from_numpy(audio)
    duration = get_duration(sr, audio.shape)
    size = get_size(audio.shape, audio.dtype)
    rms_data = extract_rms_feature(audio_tensor)
    return {
        "sample_rate": sr,
        "shape": audio.shape,
        "duration": duration,
        "size": size,
        "rms": rms_data
    }
