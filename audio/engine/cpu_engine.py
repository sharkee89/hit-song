import os
import sys
import time
import torch
import torchaudio

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audio.utils.audio_utils import rms


class CPUAudioEngine:

    def __init__(self):
        self.device = torch.device("cpu")


    def load(self, file_path: str):
        start_time = time.perf_counter()
        waveform, sample_rate = torchaudio.load(file_path)
        decode_time = time.perf_counter()

        rms_values = rms(waveform, frame_size=2048)
        analysis_time = time.perf_counter()
        data = rms_values

        return waveform, sample_rate, data, {
            "decode_time_ms": round(
                (decode_time - start_time) * 1000, 2
            ),
            "cpu_to_gpu_time_ms": 0.0,
            "total_load_time_ms": round(
                (analysis_time - start_time) * 1000, 2
            ),
            "dtype": str(waveform.dtype),
            "device": str(waveform.device),
            "shape": list(waveform.shape),
            "contiguous": waveform.is_contiguous()
        }