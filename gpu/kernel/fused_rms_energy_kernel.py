import triton
import triton.language as tl


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
