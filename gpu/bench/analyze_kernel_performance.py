import torch


def analyze_kernel_performance(
    time_ms: float,
    batch_size: int,
    num_bins: int,
    out_dim: int,
    num_frames: int,
    gpu_max_bandwidth_gbs: float = 936.0,
    bytes_per_element: int = 4,  # FP32 = 4 B, FP16 = 2 B
):
    """Izračunava ostvareni Bandwidth, TFLOPS i efikasnost u odnosu na teorijski max."""
    time_sec = time_ms / 1000.0

    # 1. Izračunavanje prenetih bajtova
    bytes_read_filters = out_dim * num_bins * bytes_per_element
    bytes_read_spec = batch_size * num_bins * num_frames * bytes_per_element
    bytes_written_out = batch_size * out_dim * num_frames * bytes_per_element

    total_bytes = bytes_read_filters + bytes_read_spec + bytes_written_out
    achieved_gb_s = (total_bytes / 1e9) / time_sec

    # 2. Izračunavanje matematičkih operacija (2 * M * K * N FLOPs za GEMM)
    total_flops = 2 * batch_size * out_dim * num_bins * num_frames
    achieved_tflops = (total_flops / 1e12) / time_sec

    # Procenat iskorišćenosti teorijskog memorijskog opsega
    bandwidth_utilization = (achieved_gb_s / gpu_max_bandwidth_gbs) * 100

    return achieved_gb_s, achieved_tflops, bandwidth_utilization


# Ovo se izvršava SAMO ako pokreneš ovaj fajl direktno (python metrics.py)
# Kada uvoziš funkciju u bench_kernel_gpu.py, ovaj blok se preskače.
if __name__ == "__main__":
    dummy_ms = 1.5  # Test vrednost
    gb_s, tflops, util = analyze_kernel_performance(
        time_ms=dummy_ms,
        batch_size=10,
        num_bins=1025,
        out_dim=80,
        num_frames=10893,
        gpu_max_bandwidth_gbs=936.0,
    )
    print("--- TEST ANALYZE FUNCTION ---")
    print(f"Ostvareni Protok  : {gb_s:.2f} GB/s ({util:.1f}% teorijskog max)")
    print(f"Ostvareni Compute : {tflops:.3f} TFLOPS")