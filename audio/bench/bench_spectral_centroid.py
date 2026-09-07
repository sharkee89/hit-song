import librosa
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


def bench_centroid():
    BLOCK_SIZE = 1024

    x = frequencies
    y = magnitude

    n_bins = x.numel()
    num_frames = y.shape[0]

    num_programs = num_frames

    output = torch.empty(
        num_programs,
        device="cuda",
        dtype=torch.float32,
    )

    grid = (num_programs,)

    # Warmup Triton
    for _ in range(10):
        spectral_centroid_kernel[grid](
            x,
            y,
            output,
            n_bins,
            BLOCK_SIZE,
        )

    torch.cuda.synchronize()

    # Triton timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(1):
        spectral_centroid_kernel[grid](
            x,
            y,
            output,
            n_bins,
            BLOCK_SIZE,
        )

    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end)
    triton_time = elapsed / 1

    print(f"Triton:  {triton_time:.6f} ms")

    # PyTorch timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(1):
        reference = (
            torch.sum(x * y, dim=1)
            / torch.sum(y, dim=1)
        )

    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end)
    pytorch_time = elapsed / 1

    print(f"PyTorch: {pytorch_time:.6f} ms")

    # Correctness
    print("output shape:", output.shape)
    print("reference shape:", reference.shape)

    print(
        "allclose:",
        torch.allclose(
            reference,
            output,
            equal_nan=True,
        ),
    )

    print("Triton:", output[:10])
    print("PyTorch:", reference[:10])

    # Ignore NaN frames when calculating max error
    valid = ~torch.isnan(reference)

    max_error = torch.max(
        torch.abs(
            reference[valid] - output[valid]
        )
    )

    print("max error:", max_error)

    # Diagnostics
    print("frequencies[:10]:", frequencies[:10])
    print("magnitude[0, :10]:", magnitude[0, :10])

    print("NaN Triton:", torch.isnan(output).sum())
    print("NaN PyTorch:", torch.isnan(reference).sum())

    print(
        "Zero magnitude frames:",
        (y.sum(dim=1) == 0).sum(),
    )

    nan_frames = torch.isnan(reference).nonzero().flatten()

    print("NaN frameovi:", nan_frames[:20])


if __name__ == "__main__":
    bench_centroid()
