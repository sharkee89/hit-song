# import librosa
# import os
# import sys
# import torch
#
# PROJECT_ROOT = os.path.dirname(
#     os.path.dirname(
#         os.path.dirname(os.path.abspath(__file__))
#     )
# )
#
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)
#
# from audio.kernels.spectral_centroid_kernel import spectral_centroid_kernel
#
#
# audio_path = r"C:\Users\Sharkee\Downloads\Aylex_-_Live_It_(freetouse.com).mp3"
#
# waveform, sample_rate = librosa.load(
#     audio_path,
#     sr=None,
#     mono=False,
# )
#
# waveform = waveform[0]
#
# n_fft = 2048
# hop_length = 512
#
# stft = librosa.stft(
#     waveform,
#     n_fft=n_fft,
#     hop_length=hop_length,
# )
#
# magnitude = torch.from_numpy(
#     abs(stft).T
# ).cuda().float()
#
# frequencies = torch.fft.rfftfreq(
#     n_fft,
#     d=1.0 / sample_rate,
# ).cuda().float()
#
# print("sample rate:", sample_rate)
# print("waveform:", waveform.shape)
# print("magnitude:", magnitude.shape)
# print("frequencies:", frequencies.shape)
#
#
# def bench_centroid():
#     BLOCK_SIZE = 1024
#
#     x = frequencies
#     y = magnitude
#
#     n_bins = x.numel()
#     num_frames = y.shape[0]
#
#     num_programs = num_frames
#
#     output = torch.empty(
#         num_programs,
#         device="cuda",
#         dtype=torch.float32,
#     )
#
#     grid = (num_programs,)
#
#     # Warmup Triton
#     for _ in range(10):
#         spectral_centroid_kernel[grid](
#             x,
#             y,
#             output,
#             n_bins,
#             BLOCK_SIZE,
#         )
#
#     torch.cuda.synchronize()
#
#     # Triton timing
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#
#     start.record()
#
#     for _ in range(100):
#         spectral_centroid_kernel[grid](
#             x,
#             y,
#             output,
#             n_bins,
#             BLOCK_SIZE,
#         )
#
#     end.record()
#     torch.cuda.synchronize()
#
#     elapsed = start.elapsed_time(end)
#     triton_time = elapsed / 100
#
#     print(f"Triton:  {triton_time:.6f} ms")
#
#     # PyTorch timing
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#
#     start.record()
#
#     for _ in range(100):
#         reference = (
#             torch.sum(x * y, dim=1)
#             / torch.sum(y, dim=1)
#         )
#
#     end.record()
#     torch.cuda.synchronize()
#
#     elapsed = start.elapsed_time(end)
#     pytorch_time = elapsed / 100
#
#     print(f"PyTorch: {pytorch_time:.6f} ms")
#
#     # Correctness
#     print("output shape:", output.shape)
#     print("reference shape:", reference.shape)
#
#     print(
#         "allclose:",
#         torch.allclose(
#             reference,
#             output,
#             equal_nan=True,
#         ),
#     )
#
#     print("Triton:", output[:10])
#     print("PyTorch:", reference[:10])
#
#     # Ignore NaN frames when calculating max error
#     valid = ~torch.isnan(reference)
#
#     max_error = torch.max(
#         torch.abs(
#             reference[valid] - output[valid]
#         )
#     )
#
#     print("max error:", max_error)
#
#     # Diagnostics
#     print("frequencies[:10]:", frequencies[:10])
#     print("magnitude[0, :10]:", magnitude[0, :10])
#
#     print("NaN Triton:", torch.isnan(output).sum())
#     print("NaN PyTorch:", torch.isnan(reference).sum())
#
#     print(
#         "Zero magnitude frames:",
#         (y.sum(dim=1) == 0).sum(),
#     )
#
#     nan_frames = torch.isnan(reference).nonzero().flatten()
#
#     print("NaN frameovi:", nan_frames[:20])
#
#
# if __name__ == "__main__":
#     bench_centroid()
import librosa
import os
import sys
import statistics
import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audio.kernels.spectral_centroid_kernel import spectral_centroid_kernel


audio_path = r"C:\Users\Sharkee\Downloads\Aylex_-_Live_It_(freetouse.com).mp3"

waveform, sample_rate = librosa.load(
    audio_path,
    sr=None,
    mono=False,
)

waveform = waveform[0]

n_fft = 2048
hop_length = 512

stft = librosa.stft(
    waveform,
    n_fft=n_fft,
    hop_length=hop_length,
)

magnitude = torch.from_numpy(
    abs(stft).T
).cuda().float()

frequencies = torch.fft.rfftfreq(
    n_fft,
    d=1.0 / sample_rate,
).cuda().float()

print("sample rate:", sample_rate)
print("waveform:", waveform.shape)
print("magnitude:", magnitude.shape)
print("frequencies:", frequencies.shape)


def benchmark_triton(
    x,
    y,
    output,
    grid,
    n_bins,
    block_size,
    num_warps,
    warmup=10,
    iterations=100,
    repeats=5,
):
    for _ in range(warmup):
        spectral_centroid_kernel[grid](
            x,
            y,
            output,
            n_bins,
            block_size,
            num_warps=num_warps,
        )

    torch.cuda.synchronize()

    timings = []

    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        for _ in range(iterations):
            spectral_centroid_kernel[grid](
                x,
                y,
                output,
                n_bins,
                block_size,
                num_warps=num_warps,
            )

        end.record()

        torch.cuda.synchronize()

        elapsed = start.elapsed_time(end) / iterations
        timings.append(elapsed)

    return statistics.median(timings)


def benchmark_pytorch(
    x,
    y,
    warmup=10,
    iterations=100,
    repeats=5,
):
    for _ in range(warmup):
        reference = (
            torch.sum(x * y, dim=1)
            / torch.sum(y, dim=1)
        )

    torch.cuda.synchronize()

    timings = []

    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        for _ in range(iterations):
            reference = (
                torch.sum(x * y, dim=1)
                / torch.sum(y, dim=1)
            )

        end.record()

        torch.cuda.synchronize()

        elapsed = start.elapsed_time(end) / iterations
        timings.append(elapsed)

    return statistics.median(timings), reference


def check_correctness(
    reference,
    output,
):
    allclose = torch.allclose(
        reference,
        output,
        equal_nan=True,
    )

    valid = ~torch.isnan(reference)

    if valid.any():
        max_error = torch.max(
            torch.abs(
                reference[valid] - output[valid]
            )
        )
    else:
        max_error = torch.tensor(
            0.0,
            device=reference.device,
        )

    return allclose, max_error


def bench_centroid():
    BLOCK_SIZE = 1024

    WARPS_TO_TEST = [
        1,
        2,
        4,
        8,
    ]

    x = frequencies
    y = magnitude

    n_bins = x.numel()
    num_frames = y.shape[0]

    output = torch.empty(
        num_frames,
        device="cuda",
        dtype=torch.float32,
    )

    grid = (num_frames,)

    # PyTorch baseline
    pytorch_time, reference = benchmark_pytorch(
        x,
        y,
    )

    print()
    print(f"PyTorch baseline: {pytorch_time:.6f} ms")

    # Test Triton configurations
    results = []

    for num_warps in WARPS_TO_TEST:
        triton_time = benchmark_triton(
            x,
            y,
            output,
            grid,
            n_bins,
            BLOCK_SIZE,
            num_warps,
        )

        allclose, max_error = check_correctness(
            reference,
            output,
        )

        speedup = pytorch_time / triton_time

        results.append(
            (
                num_warps,
                triton_time,
                speedup,
                allclose,
                max_error.item(),
            )
        )

    print()
    print(
        "BLOCK_SIZE=1024"
    )
    print(
        "-" * 70
    )
    print(
        f"{'Warps':>8} "
        f"{'Triton (ms)':>14} "
        f"{'Speedup':>12} "
        f"{'Allclose':>12} "
        f"{'Max error':>14}"
    )
    print(
        "-" * 70
    )

    for (
        num_warps,
        triton_time,
        speedup,
        allclose,
        max_error,
    ) in results:
        print(
            f"{num_warps:>8} "
            f"{triton_time:>14.6f} "
            f"{speedup:>11.2f}x "
            f"{str(allclose):>12} "
            f"{max_error:>14.6f}"
        )

    # Select fastest correct configuration
    correct_results = [
        result
        for result in results
        if result[3]
    ]

    if correct_results:
        best = min(
            correct_results,
            key=lambda result: result[1],
        )

        (
            best_warps,
            best_time,
            best_speedup,
            _,
            best_error,
        ) = best

        print()
        print("Best configuration:")
        print(f"  BLOCK_SIZE: {BLOCK_SIZE}")
        print(f"  num_warps:  {best_warps}")
        print(f"  Triton:     {best_time:.6f} ms")
        print(f"  PyTorch:    {pytorch_time:.6f} ms")
        print(f"  Speedup:    {best_speedup:.2f}x")
        print(f"  Max error:  {best_error:.6f}")

    # Diagnostics
    print()
    print("Output shape:", output.shape)
    print("Reference shape:", reference.shape)

    print(
        "NaN Triton:",
        torch.isnan(output).sum(),
    )

    print(
        "NaN PyTorch:",
        torch.isnan(reference).sum(),
    )

    print(
        "Zero magnitude frames:",
        (y.sum(dim=1) == 0).sum(),
    )

    nan_frames = (
        torch.isnan(reference)
        .nonzero()
        .flatten()
    )

    print(
        "NaN frameovi:",
        nan_frames[:20],
    )


if __name__ == "__main__":
    bench_centroid()
