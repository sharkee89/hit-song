import torch
import triton
import triton.language as tl


@triton.jit
def spectral_centroid_kernel(
    frequencies_ptr,
    magnitude_ptr,
    output_ptr,
    n_bins,
    BLOCK_SIZE: tl.constexpr,
):
    # Which block is processed
    pid_frame = tl.program_id(0)
    numerator = tl.zeros((), dtype=tl.float32)
    denominator = tl.zeros((), dtype=tl.float32)
    offsets = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, n_bins, BLOCK_SIZE):
        # Get elements for a block
        frequency_offsets = i + offsets
        magnitude_offsets = pid_frame * n_bins + i + offsets
        # Mask that is filtering only existing elements of input array
        mask = i + offsets < n_bins
        # Load values from VRAM
        frequencies = tl.load(frequencies_ptr + frequency_offsets, mask=mask, other=0.0)
        magnitude = tl.load(magnitude_ptr + magnitude_offsets, mask=mask, other=0.0)
        # Calculate total for that block
        weighted = frequencies * magnitude
        numerator = numerator + tl.sum(weighted, axis=0)
        denominator = denominator + tl.sum(magnitude, axis=0)
    centroid = numerator / denominator
    tl.store(output_ptr + pid_frame, centroid)
