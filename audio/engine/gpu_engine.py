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
from audio.kernels.rms_kernel import triton_rms
from audio.bench.bench import bench_rms


class GPUAudioEngine:

    def __init__(self):
        self.device = torch.device("cuda")

    def rms_calculate_and_compare_triton_and_pytorch(self, waveform, rms_values):
        triton_result = triton_rms(waveform)
        pytorch_ms, triton_ms = bench_rms(waveform)
        print("Triton:", triton_result[:, :10])
        print("PyTorch:", rms_values[:, :10])
        print("Triton shape: ", triton_result.shape)
        print("PyTorch shape: ", rms_values.shape)
        print("All close: ", torch.allclose(triton_result, rms_values))
        print("Max error: ", torch.max(torch.abs(triton_result - rms_values)))

    def load(self, file_path: str):
        start_time = time.perf_counter()
        waveform, sample_rate = torchaudio.load(file_path)
        decode_time = time.perf_counter()
        waveform = waveform.to(self.device)
        torch.cuda.synchronize(self.device)
        rms_values = rms(waveform, frame_size=2048)
        self.rms_calculate_and_compare_triton_and_pytorch(waveform, rms_values)
        transfer_time = time.perf_counter()

        frame = waveform[0, :2048]
        spectrum = torch.fft.rfft(frame)
        magnitude = torch.abs(spectrum)
        max = torch.argmax(magnitude)

        print("spectrum: ", spectrum)
        print("magnitude: ", magnitude)
        print("max: ", max)

        return waveform, sample_rate, {
            "decode_time_ms": round(
                (decode_time - start_time) * 1000, 2
            ),
            "cpu_to_gpu_time_ms": round(
                (transfer_time - decode_time) * 1000, 2
            ),
            "total_load_time_ms": round(
                (transfer_time - start_time) * 1000, 2
            ),
            "dtype": str(waveform.dtype),
            "device": str(waveform.device),
            "shape": list(waveform.shape),
            "contiguous": waveform.is_contiguous()
        }
