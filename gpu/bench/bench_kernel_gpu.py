import statistics
import sys
from pathlib import Path

import torch
import torchaudio
import triton

# ============================================================================
# 1. SETUP PATHA I UVOZ MODULA
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze_kernel_performance import analyze_kernel_performance
from gpu.kernel.triton_chroma_kernel import triton_chroma_kernel
from gpu.kernel.triton_kernel_utils import (
    build_chroma_filterbank,
    get_mel_filters,
)
from gpu.kernel.triton_mel_kernel import triton_mel_kernel

# ============================================================================
# 2. KONFIGURACIJA I DEFINISANJE PUTANJA
# ============================================================================
DOWNLOADS_DIR = Path(r"C:\Users\Sharkee\Downloads")

FILE_PATHS = [
                 DOWNLOADS_DIR / "Aylex_-_Live_It_(freetouse.com).mp3",
                 DOWNLOADS_DIR / "Aylex_-_Live_It_(freetouse.com) - Copy.mp3",
             ] + [
                 DOWNLOADS_DIR / f"Aylex_-_Live_It_(freetouse.com) - Copy ({i}).mp3"
                 for i in range(2, 10)
             ]

N_FFT, HOP_LENGTH, N_MELS = 2048, 512, 80
NUM_BINS = N_FFT // 2 + 1
FRAME_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768]


# ============================================================================
# 3. BENCHMARKING I REFERENCE FUNKCIJE
# ============================================================================
def benchmark_cuda(fn, warmup=20, runs=100):
    # Prvo sinhronizujemo sve prethodne radnje na GPU
    torch.cuda.synchronize()

    # Warmup faza
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    timings = []

    for _ in range(runs):
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()
        timings.append(start_event.elapsed_time(end_event))

    return statistics.median(timings)


def pytorch_batched_mel(spec, mel_filters, out):
    torch.matmul(mel_filters.unsqueeze(0), spec, out=out)
    return out.add_(1e-6).log_()


def pytorch_batched_chroma(spec, chroma_map, out):
    return torch.matmul(chroma_map.unsqueeze(0), spec, out=out)


def call_triton_mel(spec, mel_filters, mel_out, num_bins, num_mels, num_frames):
    b_size = spec.shape[0]

    grid = lambda META: (
        triton.cdiv(num_mels, META["BLOCK_SIZE_M"]),
        triton.cdiv(num_frames, META["BLOCK_SIZE_N"]),
        b_size,
    )

    triton_mel_kernel[grid](
        spec,
        mel_filters,
        mel_out,
        num_bins,
        num_mels,
        num_frames,
        spec.stride(0),
        spec.stride(1),
        spec.stride(2),
        mel_out.stride(0),
        mel_out.stride(1),
        mel_out.stride(2),
    )


def call_triton_chroma(spec, chroma_map, chroma_out, num_bins, num_frames):
    b_size = spec.shape[0]

    grid = lambda META: (
        triton.cdiv(num_frames, META["BLOCK_SIZE_N"]),
        b_size,
    )

    triton_chroma_kernel[grid](
        spec,
        chroma_map,
        chroma_out,
        num_bins,
        num_frames,
        spec.stride(0),
        spec.stride(1),
        spec.stride(2),
        chroma_out.stride(0),
        chroma_out.stride(1),
        chroma_out.stride(2),
    )


# ============================================================================
# 4. GLAVNA RUNNER FUNKCIJA
# ============================================================================
def run_benchmark():
    waveforms = []
    sr = None
    min_length = float("inf")

    print("--- UČITAVANJE 10 AUDIO FAJLOVA ---")
    for file_path in FILE_PATHS:
        if not file_path.exists():
            raise FileNotFoundError(f"Fajl nije pronađen: {file_path}")

        wf, file_sr = torchaudio.load(file_path)
        wf = wf.mean(dim=0)  # Konverzija u Mono

        if sr is None:
            sr = file_sr

        if wf.numel() < min_length:
            min_length = wf.numel()

        waveforms.append(wf)

    waveforms = [wf[:min_length] for wf in waveforms]
    batch_waveform = torch.stack(waveforms).cuda()

    window = torch.hann_window(N_FFT, device="cuda", dtype=batch_waveform.dtype)
    stft = torch.stft(
        batch_waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True,
    )
    spec = torch.abs(stft).contiguous()
    total_frames = spec.shape[2]

    mel_filters = get_mel_filters(sr, NUM_BINS, N_MELS).cuda()
    mel_filters = mel_filters / (mel_filters.sum(dim=-1, keepdim=True) + 1e-8)
    chroma_map = build_chroma_filterbank(sr, NUM_BINS).cuda()

    print(
        f"Učitano 10 fajlova: {total_frames} frejmova po fajlu ({min_length / sr:.2f}s) | GPU: {torch.cuda.get_device_name(0)}\n"
    )

    def evaluate(spec_slice, label):
        b_size, _, num_frames = spec_slice.shape

        # MEL BENCHMARK
        py_mel_out = torch.empty((b_size, N_MELS, num_frames), device="cuda")
        tr_mel_out = torch.empty_like(py_mel_out)

        # Trka i merenje za PyTorch MEL
        py_mel_ms = benchmark_cuda(
            lambda: pytorch_batched_mel(spec_slice, mel_filters, py_mel_out)
        )

        # Trka i merenje za Triton MEL
        tr_mel_ms = benchmark_cuda(
            lambda: call_triton_mel(
                spec_slice, mel_filters, tr_mel_out, NUM_BINS, N_MELS, num_frames
            )
        )

        mel_diff = (py_mel_out - tr_mel_out).abs()

        # CHROMA BENCHMARK
        py_chr_out = torch.empty((b_size, 12, num_frames), device="cuda")
        tr_chr_out = torch.zeros((b_size, 16, num_frames), device="cuda")

        # Trka i merenje za PyTorch CHROMA
        py_chr_ms = benchmark_cuda(
            lambda: pytorch_batched_chroma(
                spec_slice, chroma_map[:12], py_chr_out
            )
        )

        # Trka i merenje za Triton CHROMA
        tr_chr_ms = benchmark_cuda(
            lambda: call_triton_chroma(
                spec_slice, chroma_map, tr_chr_out, NUM_BINS, num_frames
            )
        )

        chr_diff = (py_chr_out - tr_chr_out[:, :12, :]).abs()

        # PRORAČUN HARDVERSKIH METRIKA
        mel_gb_s, mel_tflops, mel_util = analyze_kernel_performance(
            time_ms=tr_mel_ms,
            batch_size=b_size,
            num_bins=NUM_BINS,
            out_dim=N_MELS,
            num_frames=num_frames,
            gpu_max_bandwidth_gbs=936.0,
        )

        chr_gb_s, chr_tflops, chr_util = analyze_kernel_performance(
            time_ms=tr_chr_ms,
            batch_size=b_size,
            num_bins=NUM_BINS,
            out_dim=16,
            num_frames=num_frames,
            gpu_max_bandwidth_gbs=936.0,
        )

        print(f"--- {label} ({b_size} batch x {num_frames} frames) ---")
        print(
            f"MEL    | PyTorch: {py_mel_ms:.4f} ms | Triton: {tr_mel_ms:.4f} ms | "
            f"Speedup: {py_mel_ms / tr_mel_ms:.2f}x | "
            f"Bandwidth: {mel_gb_s:.1f} GB/s ({mel_util:.1f}% max) | Max Err: {mel_diff.max():.2e}"
        )
        print(
            f"CHROMA | PyTorch: {py_chr_ms:.4f} ms | Triton: {tr_chr_ms:.4f} ms | "
            f"Speedup: {py_chr_ms / tr_chr_ms:.2f}x | "
            f"Bandwidth: {chr_gb_s:.1f} GB/s ({chr_util:.1f}% max) | Max Err: {chr_diff.max():.2e}\n"
        )

    # Izvršavanje nad celim audio zapisom
    evaluate(spec, "FULL BATCH AUDIO")

    # Skaliranje po frejmovima
    for size in [s for s in FRAME_SIZES if s <= total_frames]:
        evaluate(spec[:, :, :size].contiguous(), f"SCALING {size}")


if __name__ == "__main__":
    run_benchmark()