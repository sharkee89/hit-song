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

from audio.kernels.rms_kernel import triton_rms


def main():
    waveform = torch.randn(
        (5577166, 2),
        device="cuda",
        dtype=torch.float32,
    ).transpose(0, 1)

    # Warmup
    for _ in range(20):
        triton_rms(
            waveform,
            frame_size=2048,
            block_size=2048,
            num_warps=8,
        )

    torch.cuda.synchronize()

    # Profiling workload
    for _ in range(100):
        triton_rms(
            waveform,
            frame_size=2048,
            block_size=2048,
            num_warps=8,
        )

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()