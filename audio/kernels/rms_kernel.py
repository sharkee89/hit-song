# import torch
# import triton
# import triton.language as tl
#
#
# @triton.jit
# def rms_kernel(
#     waveform_ptr,
#     output_ptr,
#     stride_channel,
#     stride_sample,
#     frame_size,
#     num_frames,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     pid_channel = tl.program_id(0)
#     pid_frame = tl.program_id(1)
#
#     frame_start = pid_frame * frame_size
#
#     offsets = tl.arange(0, BLOCK_SIZE)
#     sum_squared = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
#
#     for i in range(0, 2048, BLOCK_SIZE):
#         sample_offsets = i + offsets
#
#         sample_ptrs = (
#             waveform_ptr
#             + pid_channel * stride_channel
#             + (frame_start + sample_offsets) * stride_sample
#         )
#
#         values = tl.load(
#             sample_ptrs,
#             mask=sample_offsets < frame_size,
#             other=0.0,
#         )
#
#         sum_squared += values * values
#
#     sum_squared = tl.sum(sum_squared, axis=0)
#
#     rms = tl.sqrt(sum_squared / frame_size)
#
#     output_ptr = output_ptr + pid_channel * num_frames + pid_frame
#     tl.store(output_ptr, rms)
#
#
# def triton_rms(
#     waveform,
#     frame_size=2048,
#     block_size=512,
#     num_warps=4,
# ):
#     num_channels, num_samples = waveform.shape
#     num_frames = num_samples // frame_size
#
#     output = torch.empty(
#         (num_channels, num_frames),
#         device=waveform.device,
#         dtype=torch.float32,
#     )
#
#     rms_kernel[(num_channels, num_frames)](
#         waveform,
#         output,
#         waveform.stride(0),
#         waveform.stride(1),
#         frame_size,
#         num_frames,
#         BLOCK_SIZE=block_size,
#         num_warps=num_warps,
#     )
#
#     return output
import torch
import triton
import triton.language as tl


@triton.jit
def rms_kernel(
    waveform_ptr,
    output_ptr,
    stride_channel,
    stride_sample,
    frame_size,
    num_frames,
    BLOCK_SIZE: tl.constexpr,
):
    pid_channel = tl.program_id(0)
    pid_frame = tl.program_id(1)

    frame_start = pid_frame * frame_size
    offsets = tl.arange(0, BLOCK_SIZE)

    sum_squared = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for i in range(0, 2048, BLOCK_SIZE):
        sample_offsets = i + offsets

        ptrs = (
            waveform_ptr
            + pid_channel * stride_channel
            + (frame_start + sample_offsets) * stride_sample
        )

        values = tl.load(
            ptrs,
            mask=sample_offsets < frame_size,
            other=0.0,
        )

        sum_squared += values * values

    sum_squared = tl.sum(sum_squared, axis=0)

    rms = tl.sqrt(sum_squared / frame_size)

    output_ptr = output_ptr + pid_channel * num_frames + pid_frame
    tl.store(output_ptr, rms)


def triton_rms(
    waveform,
    frame_size=2048,
    block_size=1024,
    num_warps=4,
):
    num_channels, num_samples = waveform.shape
    num_frames = num_samples // frame_size

    output = torch.empty(
        (num_channels, num_frames),
        device=waveform.device,
        dtype=torch.float32,
    )

    rms_kernel[(num_channels, num_frames)](
        waveform,
        output,
        waveform.stride(0),
        waveform.stride(1),
        frame_size,
        num_frames,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return output