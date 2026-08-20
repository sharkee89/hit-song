import math
import torch
import torchaudio.functional as F
import triton
import triton.language as tl

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ============================================================================
# TRITON KERNELS
# ============================================================================

@triton.jit
def fused_rms_energy_kernel(
    audio_ptr,
    energy_out_ptr,
    total_samples,
    frame_length,
    hop_length,
    BLOCK_SIZE: tl.constexpr
):
    """Izračunava RMS energiju po frejmovima u jednom GPU prolazu."""
    frame_idx = tl.program_id(0)
    start_sample = frame_idx * hop_length

    offsets = tl.arange(0, BLOCK_SIZE)
    current_samples = start_sample + offsets

    mask = (offsets < frame_length) & (current_samples < total_samples)
    audio_samples = tl.load(audio_ptr + current_samples, mask=mask, other=0.0)

    squared_samples = audio_samples * audio_samples
    sum_squares = tl.sum(tl.where(mask, squared_samples, 0.0), axis=0)

    rms = tl.sqrt(sum_squares / frame_length)
    tl.store(energy_out_ptr + frame_idx, rms)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=4),
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
# HELPER FUNCTIONS
# ============================================================================

def build_chroma_filterbank(sample_rate: int, num_bins: int) -> torch.Tensor:
    """Gradi matricu preslikavanja FFT binova u 12 polu-tonskih klasa sa koritom za Triton [16, num_bins]."""
    fft_freqs = torch.linspace(0, sample_rate / 2.0, num_bins)
    midi_pitch = torch.zeros_like(fft_freqs)
    nonzero = fft_freqs > 0
    midi_pitch[nonzero] = 69.0 + 12.0 * torch.log2(fft_freqs[nonzero] / 440.0)
    pitch_classes = (torch.round(midi_pitch) % 12).long()

    chroma_map = torch.zeros((16, num_bins), dtype=torch.float32)
    rows = torch.arange(12).unsqueeze(1)
    active = (rows == pitch_classes.unsqueeze(0)) & nonzero.unsqueeze(0)
    tmp = torch.zeros((12, num_bins), dtype=torch.float32)
    tmp[active] = 1.0
    tmp /= (tmp.sum(dim=1, keepdim=True) + 1e-6)
    chroma_map[:12] = tmp
    return chroma_map.cuda()


def get_mel_filters(sample_rate: int, num_bins: int, n_mels: int) -> torch.Tensor:
    """Kreira Mel scale filterbank na GPU."""
    return F.melscale_fbanks(
        n_freqs=num_bins,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm='slaney'
    ).to('cuda').T.contiguous()