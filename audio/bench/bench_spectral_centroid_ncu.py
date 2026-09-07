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

from audio.kernels.spectral_centroid_kernel import (
    spectral_centroid_kernel,
)


audio_path = (
    r"C:\Users\Sharkee\Downloads"
    r"\Aylex_-_Live_It_(freetouse.com).mp3"
)

BLOCK_SIZE = 1024
NUM_WARPS = 1


def load_data():
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

    return (
        sample_rate,
        waveform,
        magnitude,
        frequencies,
    )


def run_triton(magnitude, frequencies):
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

    spectral_centroid_kernel[grid](
        x,
        y,
        output,
        n_bins,
        BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )

    return output


def main():
    (
        sample_rate,
        waveform,
        magnitude,
        frequencies,
    ) = load_data()

    print("sample rate:", sample_rate)
    print("waveform:", waveform.shape)
    print("magnitude:", magnitude.shape)
    print("frequencies:", frequencies.shape)
    print("BLOCK_SIZE:", BLOCK_SIZE)
    print("num_warps:", NUM_WARPS)

    # Compile + warmup.
    for _ in range(20):
        run_triton(
            magnitude,
            frequencies,
        )

    torch.cuda.synchronize()

    print()
    print("Starting profiled Triton section...")
    print()

    # From this point onward we deliberately run ONLY
    # the Triton spectral centroid kernel.
    for _ in range(20):
        run_triton(
            magnitude,
            frequencies,
        )

    torch.cuda.synchronize()

    print()
    print("Done.")


if __name__ == "__main__":
    main()