import sys
from pathlib import Path
import torch
import time
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from gpu.kernel.cuda.loader import run_custom_cuda_chroma


def benchmark():
    # Simuliramo ulazne podatke (npr. spektrogram oblika [broj_frejmova, broj_binova])
    # Testiramo sa 1 fajlom, pa sa 10 fajlova (veći batch)

    print("--- Pokretanje testa za 1 fajl ---")
    input_tensor_1 = torch.randn(1024, 512, device="cuda", dtype=torch.float32)

    # Zagrevanje (warmup)
    for _ in range(5):
        _ = run_custom_cuda_chroma(input_tensor_1)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(100):
        output_1 = run_custom_cuda_chroma(input_tensor_1)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    print(100_000 / (end_time - start_time), "iteracija u sekundi / prosečno vreme:",
          (end_time - start_time) / 100 * 1000, "ms")

    print("\n--- Pokretanje testa za 10+ fajlova (batch) ---")
    input_tensor_batch = torch.randn(1024 * 10, 512, device="cuda", dtype=torch.float32)

    for _ in range(5):
        _ = run_custom_cuda_chroma(input_tensor_batch)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(100):
        output_batch = run_custom_cuda_chroma(input_tensor_batch)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    print("Vreme za batch od 10+ fajlova:", (end_time - start_time) / 100 * 1000, "ms")


if __name__ == "__main__":
    benchmark()