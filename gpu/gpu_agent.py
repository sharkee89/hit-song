import os
import statistics
import torch
import torchaudio
import torchaudio.functional as F
import triton
import triton.language as tl

# ============================================================================
# CONFIG & SETTINGS
# ============================================================================
FILE_PATH = r"C:\Users\Sharkee\Downloads\Aylex_-_Live_It_(freetouse.com).mp3"
N_FFT, HOP_LENGTH, N_MELS = 2048, 512, 80
NUM_BINS = N_FFT // 2 + 1
FRAME_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768]


# ============================================================================
# TRITON KERNELS
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=4),
        # Dodate konfiguracije za bolje performanse na velikim matrica (>10k frejmova)
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['num_bins', 'num_mels', 'num_frames'],
)
@triton.jit
def triton_mel_kernel(
        spec_ptr, mel_filters_ptr, mel_out_ptr,
        num_bins: tl.constexpr, num_mels: tl.constexpr, num_frames,
        stride_spec_bin, stride_spec_frame,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, num_bins, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)

        mel_ptrs = mel_filters_ptr + offs_m[:, None] * num_bins + offs_k[None, :]
        mel_tile = tl.load(mel_ptrs, mask=(offs_m[:, None] < num_mels) & (offs_k[None, :] < num_bins), other=0.0)

        spec_ptrs = spec_ptr + offs_k[:, None] * stride_spec_bin + offs_n[None, :] * stride_spec_frame
        spec_tile = tl.load(spec_ptrs, mask=(offs_k[:, None] < num_bins) & (offs_n[None, :] < num_frames), other=0.0)

        acc += tl.dot(mel_tile, spec_tile)

    log_mel = tl.log(acc + 1e-6)
    out_ptrs = mel_out_ptr + offs_m[:, None] * num_frames + offs_n[None, :]
    tl.store(out_ptrs, log_mel, mask=(offs_m[:, None] < num_mels) & (offs_n[None, :] < num_frames))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['num_bins', 'num_frames'],
)
@triton.jit
def triton_chroma_kernel(
        spec_ptr, chroma_map_ptr, chroma_out_ptr,
        num_bins: tl.constexpr, num_frames,
        stride_spec_bin, stride_spec_frame,
        BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, 16)

    acc = tl.zeros((16, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, num_bins, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)

        chroma_ptrs = chroma_map_ptr + offs_m[:, None] * num_bins + offs_k[None, :]
        chroma_tile = tl.load(chroma_ptrs, mask=(offs_m[:, None] < 12) & (offs_k[None, :] < num_bins), other=0.0)

        spec_ptrs = spec_ptr + offs_k[:, None] * stride_spec_bin + offs_n[None, :] * stride_spec_frame
        spec_tile = tl.load(spec_ptrs, mask=(offs_k[:, None] < num_bins) & (offs_n[None, :] < num_frames), other=0.0)

        acc += tl.dot(chroma_tile, spec_tile)

    out_ptrs = chroma_out_ptr + offs_m[:, None] * num_frames + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < 12) & (offs_n[None, :] < num_frames))


# ============================================================================
# HELPER FUNCTIONS & ENGINE
# ============================================================================
def build_chroma_filterbank(sample_rate):
    fft_freqs = torch.linspace(0, sample_rate / 2.0, NUM_BINS)
    midi_pitch = torch.zeros_like(fft_freqs)
    nonzero = fft_freqs > 0
    midi_pitch[nonzero] = 69.0 + 12.0 * torch.log2(fft_freqs[nonzero] / 440.0)
    pitch_classes = (torch.round(midi_pitch) % 12).long()

    chroma_map = torch.zeros((16, NUM_BINS), dtype=torch.float32)
    rows = torch.arange(12).unsqueeze(1)
    active = (rows == pitch_classes.unsqueeze(0)) & nonzero.unsqueeze(0)
    tmp = torch.zeros((12, NUM_BINS), dtype=torch.float32)
    tmp[active] = 1.0
    tmp /= (tmp.sum(dim=1, keepdim=True) + 1e-6)
    chroma_map[:12] = tmp
    return chroma_map.cuda()


def benchmark_cuda(fn, warmup=20, runs=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(runs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record();
        fn();
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)


def pytorch_mel(spec, mel_filters, out):
    torch.mm(mel_filters, spec, out=out)
    return out.add_(1e-6).log_()


def pytorch_chroma(spec, chroma_map, out):
    return torch.mm(chroma_map, spec, out=out)


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================
def run_benchmark():
    waveform, sr = torchaudio.load(FILE_PATH)
    waveform = waveform.mean(dim=0).cuda()

    window = torch.hann_window(N_FFT, device="cuda", dtype=waveform.dtype)
    stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)
    spec = torch.abs(stft).contiguous()
    total_frames = spec.shape[1]

    mel_filters = F.melscale_fbanks(NUM_BINS, 0.0, sr / 2.0, N_MELS, sr, norm="slaney").T.contiguous().cuda()
    chroma_map = build_chroma_filterbank(sr)

    print(f"Loaded MP3: {total_frames} frames ({waveform.numel() / sr:.2f}s) | GPU: {torch.cuda.get_device_name(0)}\n")

    def evaluate(spec_slice, label):
        num_frames = spec_slice.shape[1]

        # MEL
        py_mel_out = torch.empty((N_MELS, num_frames), device="cuda")
        tr_mel_out = torch.empty_like(py_mel_out)

        py_mel_ms = benchmark_cuda(lambda: pytorch_mel(spec_slice, mel_filters, py_mel_out))
        grid_mel = lambda META: (
        triton.cdiv(N_MELS, META["BLOCK_SIZE_M"]), triton.cdiv(num_frames, META["BLOCK_SIZE_N"]))
        tr_mel_ms = benchmark_cuda(
            lambda: triton_mel_kernel[grid_mel](spec_slice, mel_filters, tr_mel_out, NUM_BINS, N_MELS, num_frames,
                                                spec_slice.stride(0), spec_slice.stride(1)))

        mel_diff = (py_mel_out - tr_mel_out).abs()

        # CHROMA
        py_chr_out = torch.empty((12, num_frames), device="cuda")
        tr_chr_out = torch.empty((16, num_frames), device="cuda")

        py_chr_ms = benchmark_cuda(lambda: pytorch_chroma(spec_slice, chroma_map[:12], py_chr_out))
        grid_chr = lambda META: (triton.cdiv(num_frames, META["BLOCK_SIZE_N"]),)
        tr_chr_ms = benchmark_cuda(
            lambda: triton_chroma_kernel[grid_chr](spec_slice, chroma_map, tr_chr_out, NUM_BINS, num_frames,
                                                   spec_slice.stride(0), spec_slice.stride(1)))

        chr_diff = (py_chr_out - tr_chr_out[:12]).abs()

        print(f"--- {label} ({num_frames} frames) ---")
        print(
            f"MEL    | PyTorch: {py_mel_ms:.4f} ms | Triton: {tr_mel_ms:.4f} ms | Speedup: {py_mel_ms / tr_mel_ms:.2f}x | Max Err: {mel_diff.max():.2e}")
        print(
            f"CHROMA | PyTorch: {py_chr_ms:.4f} ms | Triton: {tr_chr_ms:.4f} ms | Speedup: {py_chr_ms / tr_chr_ms:.2f}x | Max Err: {chr_diff.max():.2e}\n")

    # Full Run
    evaluate(spec, "FULL AUDIO")

    # Scaling Run
    for size in [s for s in FRAME_SIZES if s <= total_frames]:
        evaluate(spec[:, :size].contiguous(), f"SCALING {size}")


if __name__ == "__main__":
    run_benchmark()