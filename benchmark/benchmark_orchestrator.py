import os
import sys
import time
import torch
import torch.nn as nn
import onnxruntime as ort
import psutil
import numpy as np

# Optimizacija niti za konzistentnost merenja
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


class HitPredictionModel(nn.Module):
    def __init__(self):
        super(HitPredictionModel, self).__init__()
        self.linear = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x) / 0.8)


def proveri_memoriju_mb():
    pid = os.getpid()
    process = psutil.Process(pid)
    return process.memory_info().rss / (1024 * 1024)


if __name__ == "__main__":
    print("🚀 Započinjemo pripremu benchmark testova za ISO/IEC 25010...")

    # 1. Inicijalizacija PyTorch modela sa specifičnim težinama orkestratora
    pytorch_model = HitPredictionModel()
    with torch.no_grad():
        pytorch_model.linear.weight = nn.Parameter(torch.tensor([[0.4, 0.5, 0.1, 0.2]]))
        pytorch_model.linear.bias = nn.Parameter(torch.tensor([-1.2]))
    pytorch_model.eval()

    # 2. Eksportovanje u ONNX format (Statički binarni graf)
    onnx_path = "hit_predictor.onnx"
    dummy_input_tensor = torch.randn(1, 4)
    torch.onnx.export(
        pytorch_model,
        dummy_input_tensor,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_signals'],
        output_names=['hit_probability'],
        dynamic_axes={'input_signals': {0: 'batch_size'}, 'hit_probability': {0: 'batch_size'}}
    )
    print(f"✅ Uspešno generisan statički binarni graf: {onnx_path}\n")

    # 3. Simulacija skupa podataka od 1000 pesama (4 signala po pesmi [0.0 - 1.0])
    BROJ_PESAMA = 1000
    test_data_np = np.random.rand(BROJ_PESAMA, 4).astype(np.float32)
    test_data_torch = torch.tensor(test_data_np)

    # ==========================================
    # --- WARM-UP FAZA (Izolovano od merenja) ---
    # ==========================================
    # Inicijalizujemo ONNX sesiju unapred da C++ runtime overhead ne uđe u RAM test
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    # Propuštamo po jedan podatak da oba frameworka alociraju svoje interne bafera
    with torch.no_grad():
        _ = pytorch_model(test_data_torch[0:1])
    _ = ort_session.run(None, {input_name: test_data_np[0:1]})

    # ==========================================
    # TEST 2: MEMORY FOOTPRINT BENCHMARK (Neto)
    # ==========================================
    print("📊 [TEST 2] Pokrećem Memory Footprint test (Neto alokacija)...")

    # PyTorch Neto Memorija za matričnu obradu
    mem_start_pt = proveri_memoriju_mb()
    with torch.no_grad():
        _ = pytorch_model(test_data_torch)
    mem_end_pt = proveri_memoriju_mb()
    ram_pytorch = max(0.0, mem_end_pt - mem_start_pt)

    # ONNX Neto Memorija za matričnu obradu
    mem_start_onnx = proveri_memoriju_mb()
    _ = ort_session.run(None, {input_name: test_data_np})
    mem_end_onnx = proveri_memoriju_mb()
    ram_onnx = max(0.0, mem_end_onnx - mem_start_onnx)

    # ==========================================
    # TEST 1 & 3: LATENCY & THROUGHPUT (STRESS)
    # ==========================================
    print("⏱️ [TEST 1 & 3] Pokrećem Latency i Throughput testove (Iterativno)...")

    # PyTorch vremenske performanse
    start_pt = time.perf_counter()
    with torch.no_grad():
        for i in range(BROJ_PESAMA):
            _ = pytorch_model(test_data_torch[i:i + 1])
    end_pt = time.perf_counter()
    ukupno_vreme_pt = end_pt - start_pt
    latencija_pt = (ukupno_vreme_pt / BROJ_PESAMA) * 1000
    throughput_pt = BROJ_PESAMA / ukupno_vreme_pt

    # ONNX vremenske performanse
    start_onnx = time.perf_counter()
    for i in range(BROJ_PESAMA):
        _ = ort_session.run(None, {input_name: test_data_np[i:i + 1]})
    end_onnx = time.perf_counter()
    ukupno_vreme_onnx = end_onnx - start_onnx
    latencija_onnx = (ukupno_vreme_onnx / BROJ_PESAMA) * 1000
    throughput_onnx = BROJ_PESAMA / ukupno_vreme_onnx

    # ==========================================
    # KONAČAN PRIKAZ REZULTATA ZA RAD
    # ==========================================
    print("\n" + "=" * 60)
    print("📈 KONAČNI REZULTATI PERFORMANSI (ISO/IEC 25010)")
    print("=" * 60)
    print(f"{'Metrika (Karakteristika kvaliteta)':<36} | {'PyTorch':<10} | {'ONNX Runtime':<12}")
    print("-" * 60)
    print(f"{'Avg Latency (Time Behaviour)':<36} | {latencija_pt:.4f} ms | {latencija_onnx:.4f} ms")
    print(f"{'RAM Footprint (Resource Util - Neto)':<36} | {ram_pytorch:.4f} MB | {ram_onnx:.4f} MB")
    print(f"{'Throughput (Capacity)':<36} | {throughput_pt:.1f} p/s  | {throughput_onnx:.1f} p/s")
    print("=" * 60)

    if latencija_onnx > 0:
        faktor_ubrzanja = latencija_pt / latencija_onnx
        print(
            f"💡 ONNX Runtime izvršava fuziju algoritama {faktor_ubrzanja:.1f}x brže od izvornog PyTorch framework-a.\n")

    # Čišćenje generisanog ONNX fajla nakon testa (opciono)
    if os.path.exists(onnx_path):
        os.remove(onnx_path)