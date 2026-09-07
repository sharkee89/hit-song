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
    pid_frame = tl.program_id(0)

    # Inicijalizacija akumulatora za sume
    sum_weighted = 0.0
    sum_magnitude = 0.0

    # Petlja sigurno prelazi preko svih n_bins (1025 elemenata)
    # čak i kada je BLOCK_SIZE ograničen na hardverski maksimum (npr. 1024)
    for block_start in range(0, n_bins, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_bins

        # Indeksi za učitavanje
        frequency_offsets = offsets
        magnitude_offsets = pid_frame * n_bins + offsets

        # Učitavanje iz VRAM-a sa maskom
        frequencies = tl.load(frequencies_ptr + frequency_offsets, mask=mask, other=0.0)
        magnitude = tl.load(magnitude_ptr + magnitude_offsets, mask=mask, other=0.0)

        # Delimičan račun za ovaj blok
        weighted = frequencies * magnitude

        sum_weighted += tl.sum(weighted, axis=0)
        sum_magnitude += tl.sum(magnitude, axis=0)

    # Konačan izračun centroida za dati frejm
    centroid = sum_weighted / sum_magnitude
    tl.store(output_ptr + pid_frame, centroid)