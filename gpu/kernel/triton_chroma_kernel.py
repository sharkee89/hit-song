import triton
import triton.language as tl


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