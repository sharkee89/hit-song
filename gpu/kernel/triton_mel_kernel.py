import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Agresivnije BMM i Tensor-friendly konfiguracije sa $2^n$ blokovima
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8, num_stages=4),
    ],
    key=['num_bins', 'num_mels', 'num_frames'],
)
@triton.jit
def triton_mel_kernel(
    spec_ptr,
    mel_filters_ptr,
    mel_out_ptr,
    num_bins: tl.constexpr,
    num_mels: tl.constexpr,
    num_frames,
    stride_spec_batch,
    stride_spec_bin,
    stride_spec_frame,
    stride_out_batch,
    stride_out_mel,
    stride_out_frame,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < num_mels
    mask_n = offs_n < num_frames

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    spec_batch_ptr = spec_ptr + pid_b * stride_spec_batch

    for k_start in range(0, num_bins, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < num_bins

        mel_ptrs = (
            mel_filters_ptr + offs_m[:, None] * num_bins + offs_k[None, :]
        )
        mel_mask = mask_m[:, None] & mask_k[None, :]
        mel_tile = tl.load(mel_ptrs, mask=mel_mask, other=0.0)

        spec_ptrs = (
            spec_batch_ptr
            + offs_k[:, None] * stride_spec_bin
            + offs_n[None, :] * stride_spec_frame
        )
        spec_mask = mask_k[:, None] & mask_n[None, :]
        spec_tile = tl.load(spec_ptrs, mask=spec_mask, other=0.0)

        acc = tl.dot(mel_tile, spec_tile, acc)

    log_mel = tl.log(tl.maximum(acc, 1e-6))

    out_ptrs = (
        mel_out_ptr
        + pid_b * stride_out_batch
        + offs_m[:, None] * stride_out_mel
        + offs_n[None, :] * stride_out_frame
    )
    tl.store(out_ptrs, log_mel, mask=mask_m[:, None] & mask_n[None, :])