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
    BLOCK_SIZE: tl.constexpr
):
    pid_channel = tl.program_id(0)
    pid_frame = tl.program_id(1)

    frame_start = pid_frame * frame_size
    offsets = tl.arange(0, BLOCK_SIZE)

    sample_ptrs = waveform_ptr + (pid_channel * stride_channel) + (frame_start + offsets) * stride_sample
    x = tl.load(sample_ptrs, mask=offsets < frame_size, other=0.0)
    squared = x * x
    sum_squared = tl.sum(squared, axis=0)
    mean_squared = sum_squared / frame_size
    rms = tl.sqrt(mean_squared)
    output_ptrs = (output_ptr + pid_channel * num_frames + pid_frame)
    tl.store(output_ptrs, rms)


def triton_rms_one_frame(waveform):
    output = torch.empty(1, device=waveform.device, dtype=torch.float32)
    rms_kernel[(1,)](waveform, output, waveform.stride(1), BLOCK_SIZE=2048)
    return output


def triton_rms_channel(waveform, frame_size=2048):
    num_samples = waveform.shape[0]
    num_frames = num_samples // frame_size

    output = torch.empty(num_frames, device=waveform.device, dtype=torch.float32)
    rms_kernel[(num_frames,)](
        waveform,
        output,
        waveform.stride(0),
        frame_size,
        num_frames,
        BLOCK_SIZE=2048
    )
    return output


def triton_rms(waveform, frame_size=2048):
    num_channels, num_samples = waveform.shape
    num_frames = num_samples // frame_size

    output = torch.empty((num_channels, num_frames), device=waveform.device, dtype=torch.float32)
    rms_kernel[(num_channels, num_frames)](
        waveform,
        output,
        waveform.stride(0),
        waveform.stride(1),
        frame_size,
        num_frames,
        BLOCK_SIZE=2048,
        num_warps=4
    )
    return output