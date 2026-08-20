import triton
import triton.language as tl


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