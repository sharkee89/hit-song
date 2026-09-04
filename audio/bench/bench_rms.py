import os
import sys
import torch
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from audio.utils.audio_utils import rms
from audio.kernels.rms_kernel import triton_rms

def bench_rms_config(waveform, block_size, num_warps, iterations=100):
    # Warmup
    for _ in range(20):
        rms(waveform, frame_size=2048)
        triton_rms(
            waveform,
            frame_size=2048,
            block_size=block_size,
            num_warps=num_warps,
        )

    torch.cuda.synchronize()

    # PyTorch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        rms(waveform, frame_size=2048)
    end.record()

    torch.cuda.synchronize()
    pytorch_ms = start.elapsed_time(end) / iterations

    # Triton
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        triton_rms(
            waveform,
            frame_size=2048,
            block_size=block_size,
            num_warps=num_warps,
        )
    end.record()

    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / iterations

    return pytorch_ms, triton_ms, pytorch_ms / triton_ms