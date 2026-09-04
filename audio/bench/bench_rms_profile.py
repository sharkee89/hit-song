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

def benchmark_all_configs(waveform):
    block_sizes = [512, 1024, 2048, 4096]
    num_warps_list = [2, 4, 8]

    results = []

    print()
    print("Triton configuration sweep")
    print("=" * 60)

    # Warmup PyTorch
    for _ in range(20):
        rms(waveform, frame_size=2048)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(1000):
        rms(waveform, frame_size=2048)
    end.record()

    torch.cuda.synchronize()
    pytorch_ms = start.elapsed_time(end) / 1000

    print(f"PyTorch baseline: {pytorch_ms:.6f} ms")
    print()

    for block_size in block_sizes:
        for num_warps in num_warps_list:

            # Warmup
            for _ in range(20):
                triton_rms(
                    waveform,
                    frame_size=2048,
                    block_size=block_size,
                    num_warps=num_warps,
                )

            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            for _ in range(1000):
                triton_rms(
                    waveform,
                    frame_size=2048,
                    block_size=block_size,
                    num_warps=num_warps,
                )

            end.record()

            torch.cuda.synchronize()

            triton_ms = start.elapsed_time(end) / 1000
            speedup = pytorch_ms / triton_ms

            results.append(
                (block_size, num_warps, triton_ms, speedup)
            )

            print(
                f"BLOCK_SIZE={block_size:4d}  "
                f"warps={num_warps}  "
                f"Triton={triton_ms:.6f} ms  "
                f"speedup={speedup:.2f}x"
            )

    print()
    print("=" * 60)

    best = min(results, key=lambda x: x[2])

    print(
        f"BEST: BLOCK_SIZE={best[0]}, "
        f"num_warps={best[1]}, "
        f"{best[2]:.6f} ms, "
        f"{best[3]:.2f}x"
    )

    return results