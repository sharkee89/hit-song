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


def benchmark(name, waveform, block_size=1024, num_warps=4):
    # Warmup
    for _ in range(50):
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

    ms = start.elapsed_time(end) / 1000

    print(f"{name}: {ms:.6f} ms")

    return ms


def main():
    waveform = torch.randn(
        (5577166, 2),
        device="cuda",
        dtype=torch.float32,
    ).transpose(0, 1)

    print("Original")
    print("Shape   :", waveform.shape)
    print("Stride  :", waveform.stride())
    print("Contig  :", waveform.is_contiguous())

    # ---------------------------------------------------------
    # 1. ORIGINAL INTERLEAVED
    # ---------------------------------------------------------

    result_original = triton_rms(
        waveform,
        frame_size=2048,
        block_size=1024,
        num_warps=4,
    )

    pytorch_result = rms(waveform, frame_size=2048)

    print()
    print("Correctness original:")
    print(torch.allclose(
        pytorch_result,
        result_original,
        atol=1e-5,
        rtol=1e-5,
    ))

    original_ms = benchmark(
        "Triton interleaved",
        waveform,
    )

    # ---------------------------------------------------------
    # 2. CONTIGUOUS COPY
    # ---------------------------------------------------------

    waveform_contig = waveform.contiguous()

    print()
    print("Contiguous")
    print("Shape   :", waveform_contig.shape)
    print("Stride  :", waveform_contig.stride())
    print("Contig  :", waveform_contig.is_contiguous())

    result_contig = triton_rms(
        waveform_contig,
        frame_size=2048,
        block_size=1024,
        num_warps=4,
    )

    print()
    print("Correctness contiguous:")
    print(torch.allclose(
        pytorch_result,
        result_contig,
        atol=1e-5,
        rtol=1e-5,
    ))

    contig_rms_ms = benchmark(
        "Triton contiguous",
        waveform_contig,
    )

    # ---------------------------------------------------------
    # 3. COPY + RMS
    # ---------------------------------------------------------

    for _ in range(50):
        x = waveform.contiguous()
        triton_rms(
            x,
            frame_size=2048,
            block_size=1024,
            num_warps=4,
        )

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(1000):
        x = waveform.contiguous()

        triton_rms(
            x,
            frame_size=2048,
            block_size=1024,
            num_warps=4,
        )

    end.record()

    torch.cuda.synchronize()

    copy_plus_rms_ms = start.elapsed_time(end) / 1000

    print()
    print(f"Copy + Triton RMS: {copy_plus_rms_ms:.6f} ms")

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"Interleaved RMS : {original_ms:.6f} ms")
    print(f"Contiguous RMS  : {contig_rms_ms:.6f} ms")
    print(f"Copy + RMS      : {copy_plus_rms_ms:.6f} ms")

    print()
    print(
        f"RMS improvement : "
        f"{original_ms / contig_rms_ms:.2f}x"
    )

    print(
        f"End-to-end      : "
        f"{original_ms / copy_plus_rms_ms:.2f}x"
    )


if __name__ == "__main__":
    main()