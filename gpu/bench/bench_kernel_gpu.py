import statistics
import torch
import torchaudio
import triton

from gpu.kernel.triton_chroma_kernel import (
    triton_chroma_kernel
)

from gpu.kernel.triton_kernel_utils import (
    build_chroma_filterbank,
    get_mel_filters
)

from gpu.kernel.triton_mel_kernel import (
    triton_mel_kernel
)

FILE_PATH = r"C:\Users\Sharkee\Downloads\Aylex_-_Live_It_(freetouse.com).mp3"
N_FFT, HOP_LENGTH, N_MELS = 2048, 512, 80
NUM_BINS = N_FFT // 2 + 1
FRAME_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768]


def benchmark_cuda(fn, warmup=20, runs=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(runs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)


def pytorch_mel(spec, mel_filters, out):
    torch.mm(mel_filters, spec, out=out)
    return out.add_(1e-6).log_()


def pytorch_chroma(spec, chroma_map, out):
    return torch.mm(chroma_map, spec, out=out)


def run_benchmark():
    waveform, sr = torchaudio.load(FILE_PATH)
    waveform = waveform.mean(dim=0).cuda()

    window = torch.hann_window(N_FFT, device="cuda", dtype=waveform.dtype)
    stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)
    spec = torch.abs(stft).contiguous()
    total_frames = spec.shape[1]

    mel_filters = get_mel_filters(sr, NUM_BINS, N_MELS)
    chroma_map = build_chroma_filterbank(sr, NUM_BINS)

    print(f"Loaded MP3: {total_frames} frames ({waveform.numel() / sr:.2f}s) | GPU: {torch.cuda.get_device_name(0)}\n")

    def evaluate(spec_slice, label):
        num_frames = spec_slice.shape[1]

        # MEL
        py_mel_out = torch.empty((N_MELS, num_frames), device="cuda")
        tr_mel_out = torch.empty_like(py_mel_out)

        py_mel_ms = benchmark_cuda(lambda: pytorch_mel(spec_slice, mel_filters, py_mel_out))
        grid_mel = lambda META: (
            triton.cdiv(N_MELS, META["BLOCK_SIZE_M"]),
            triton.cdiv(num_frames, META["BLOCK_SIZE_N"])
        )
        tr_mel_ms = benchmark_cuda(
            lambda: triton_mel_kernel[grid_mel](
                spec_slice, mel_filters, tr_mel_out,
                NUM_BINS, N_MELS, num_frames,
                spec_slice.stride(0), spec_slice.stride(1)
            )
        )

        mel_diff = (py_mel_out - tr_mel_out).abs()

        # CHROMA
        py_chr_out = torch.empty((12, num_frames), device="cuda")
        tr_chr_out = torch.empty((16, num_frames), device="cuda")

        py_chr_ms = benchmark_cuda(lambda: pytorch_chroma(spec_slice, chroma_map[:12], py_chr_out))
        grid_chr = lambda META: (triton.cdiv(num_frames, META["BLOCK_SIZE_N"]),)
        tr_chr_ms = benchmark_cuda(
            lambda: triton_chroma_kernel[grid_chr](
                spec_slice, chroma_map, tr_chr_out,
                NUM_BINS, num_frames,
                spec_slice.stride(0), spec_slice.stride(1)
            )
        )

        chr_diff = (py_chr_out - tr_chr_out[:12]).abs()

        print(f"--- {label} ({num_frames} frames) ---")
        print(f"MEL    | PyTorch: {py_mel_ms:.4f} ms | Triton: {tr_mel_ms:.4f} ms | Speedup: {py_mel_ms / tr_mel_ms:.2f}x | Max Err: {mel_diff.max():.2e}")
        print(f"CHROMA | PyTorch: {py_chr_ms:.4f} ms | Triton: {tr_chr_ms:.4f} ms | Speedup: {py_chr_ms / tr_chr_ms:.2f}x | Max Err: {chr_diff.max():.2e}\n")

    evaluate(spec, "FULL AUDIO")

    for size in [s for s in FRAME_SIZES if s <= total_frames]:
        evaluate(spec[:, :size].contiguous(), f"SCALING {size}")


if __name__ == "__main__":
    run_benchmark()