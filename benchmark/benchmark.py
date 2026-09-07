import os
import sys
import time
import tracemalloc
import numpy as np
import torch

# Sređivanje putanja: dodajemo roditeljski folder i 'neural' folder u sys.path
# kako bi Python mogao da pronađe neural_orchestrator.py
current_dir = os.path.dirname(os.path.abspath(__file__))  # benchmark folder
parent_dir = os.path.dirname(current_dir)  # koren projekta (gde je ONNX)
neural_dir = os.path.join(parent_dir, 'neural')  # neural folder

if neural_dir not in sys.path:
    sys.path.append(neural_dir)

# Pokušavamo da uvezemo onnxruntime
try:
    import onnxruntime as ort
except ImportError:
    print("❌ Greška: Instaliraj onnxruntime koristeći: pip install onnxruntime")
    sys.exit(1)

# Uvoz modela iz neural_orchestrator.py koji je u susednom folderu
try:
    from neural_orchestrator import HitPredictionModel
except ImportError as e:
    print(f"❌ Greška pri uvozu HitPredictionModel: {e}")
    print("Proveri da li se fajl u 'neural' folderu tačno zove 'neural_orchestrator.py'")
    sys.exit(1)

# Konfigurisani parametri za benchmark
NUM_ITERATIONS = 50000
WARM_UP_RUNS = 1000
# Putanja do ONNX fajla koji se nalazi u korenu (jedan nivo iznad benchmark foldera)
ONNX_MODEL_PATH = os.path.join(parent_dir, "music_orchestrator.onnx")


def run_pytorch_benchmark(dummy_tensor):
    """Merenje latencije i memorije za izvorni PyTorch model."""
    print("\n--- [PyTorch] Inicijalizacija i merenje ---")

    # 1. Merenje memorije pre i nakon učitavanja modela
    tracemalloc.start()

    model = HitPredictionModel()
    model.eval()

    with torch.no_grad():
        model.linear.weight = torch.nn.Parameter(torch.tensor([[0.4, 0.5, 0.1, 0.2]]))
        model.linear.bias = torch.nn.Parameter(torch.tensor([-1.2]))  # <- OVDE JE BILA IZMENA

    _, pytorch_memory_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 2. Warm-up faza
    with torch.no_grad():
        for _ in range(WARM_UP_RUNS):
            _ = model(dummy_tensor)

    # 3. Merenje latencije
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(NUM_ITERATIONS):
            _ = model(dummy_tensor)
    end_time = time.perf_counter()

    pytorch_total_time = end_time - start_time
    pytorch_avg_latency = (pytorch_total_time / NUM_ITERATIONS) * 1000  # u ms

    print(f"PyTorch Peak Memory: {pytorch_memory_peak / 1024:.2f} KB")
    print(f"PyTorch Prosečna Latencija: {pytorch_avg_latency:.6f} ms")

    return pytorch_avg_latency, pytorch_memory_peak


def run_onnx_benchmark(dummy_numpy):
    """Merenje latencije i memorije za optimizovani ONNX Runtime graf."""
    print("\n--- [ONNX Runtime] Inicijalizacija i merenje ---")

    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ Greška: Fajl nije pronađen na putanji: {ONNX_MODEL_PATH}")
        print("Pokreni prvo neural_orchestrator.py da generišeš ONNX fajl u korenu projekta.")
        return None, None

    # 1. Merenje memorije za ONNX Runtime sesiju
    tracemalloc.start()

    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])

    _, onnx_memory_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    input_name = session.get_inputs()[0].name
    onnx_inputs = {input_name: dummy_numpy}

    # 2. Warm-up faza
    for _ in range(WARM_UP_RUNS):
        _ = session.run(None, onnx_inputs)

    # 3. Merenje latencije
    start_time = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        _ = session.run(None, onnx_inputs)
    end_time = time.perf_counter()

    onnx_total_time = end_time - start_time
    onnx_avg_latency = (onnx_total_time / NUM_ITERATIONS) * 1000  # u ms

    print(f"ONNX Peak Memory: {onnx_memory_peak / 1024:.2f} KB")
    print(f"ONNX Prosečna Latencija: {onnx_avg_latency:.6f} ms")

    return onnx_avg_latency, onnx_memory_peak


if __name__ == "__main__":
    print(f"=== Pokretanje Benchmarka ({NUM_ITERATIONS} iteracija, {WARM_UP_RUNS} warm-up) ===")
    print(f"Putanja do ONNX modela: {ONNX_MODEL_PATH}")

    # Generisanje identičnih "dummy" podataka (4 signala agenata)
    raw_input = [0.65, 0.88, 0.42, 0.71]

    torch_input = torch.tensor([raw_input], dtype=torch.float32)
    numpy_input = np.array([raw_input], dtype=np.float32)

    # Pokretanje testova
    py_latency, py_mem = run_pytorch_benchmark(torch_input)
    ox_latency, ox_mem = run_onnx_benchmark(numpy_input)

    # Generisanje komparativnog izveštaja
    if py_latency and ox_latency:
        print("\n" + "=" * 40)
        print("   AKADEMSKI IZVEŠTAJ (KOMPARATIVNA ANALIZA)")
        print("=" * 40)

        latency_improvement = ((py_latency - ox_latency) / py_latency) * 100
        mem_reduction = ((py_mem - ox_mem) / py_mem) * 100

        print(f"Ubrzanje izvršavanja (Latencija): {latency_improvement:.2f}%")
        print(f"Smanjenje memorijskog otiska:    {mem_reduction:.2f}%")
        print("=" * 40)