import os
import sys
import time
import tempfile
import numpy as np
import torch
import matplotlib.pyplot as plt

# Inicijalizacija tvog modela
try:
    from neural_orchestrator import HitPredictionModel
except ImportError:
    import torch.nn as nn
    class HitPredictionModel(nn.Module):
        def __init__(self):
            super(HitPredictionModel, self).__init__()
            self.linear = nn.Linear(4, 1)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            return self.sigmoid(self.linear(x) / 0.8)

model = HitPredictionModel()
with torch.no_grad():
    model.linear.weight = torch.nn.Parameter(torch.tensor([[0.4, 0.5, 0.1, 0.2]]))
    model.linear.bias = torch.nn.Parameter(torch.tensor([-1.2]))
model.eval()

# =====================================================================
# KREIRANJE PRAVIH PRIVREMENIH FAJLOVA ZA HARDVERSKI TEST
# =====================================================================
temp_dir = tempfile.gettempdir()
mp3_path = os.path.join(temp_dir, "test_song.mp3")
wav_path = os.path.join(temp_dir, "test_song.wav")
txt_path = os.path.join(temp_dir, "test_doc.txt")

# Generišemo fajlove sa stvarnim bajtovima da prevarimo keš diska
with open(mp3_path, "wb") as f:
    f.write(b"ID3" + os.urandom(1024 * 50))  # Simulacija MP3 sa ID3 tagom (50 KB)
with open(wav_path, "wb") as f:
    f.write(b"RIFF" + os.urandom(1024 * 100)) # Simulacija WAV sa RIFF headerom (100 KB)
with open(txt_path, "wb") as f:
    f.write(b"Ovo je obican tekstualni dokument koji treba biti odbacen.")

# =====================================================================
# REALNA FUNKCIJA ZA VALIDACIJU (Bez simuliranog sleep-a)
# =====================================================================
def validate_user_input_hardware(file_path, user_params):
    # 1. Brza provera ekstenzije
    allowed_extensions = ['.mp3', '.wav', '.m4a', '.flac']
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in allowed_extensions:
        return False, []

    # 2. Stvarna hardverska I/O operacija - otvaranje i čitanje prvih bajtova (Header Verification)
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
            # Simulacija bazične inspekcije formata (npr. provera magičnih bajtova)
            if ext.lower() == '.mp3' and b"ID3" not in header:
                return False, []
            if ext.lower() == '.wav' and b"RIFF" not in header:
                return False, []
    except IOError:
        return False, []

    # 3. Sanitacija numeričkih opsega agenata
    sanitized_params = []
    has_corrections = False
    for param in user_params:
        val = max(0.0, min(1.0, float(param)))
        if val != param:
            has_corrections = True
        sanitized_params.append(val)

    # Ako je bilo korekcija, simuliramo dodatni sistemski log/upozorenje na disk
    if has_corrections:
        log_path = os.path.join(temp_dir, "validation_warnings.log")
        with open(log_path, "a") as log_file:
            log_file.write(f"{time.time()}: Parametri korigovani sa {user_params} na {sanitized_params}\n")

    return True, sanitized_params

# =====================================================================
# POKRETANJE MERENJA
# =====================================================================
print("🛡️ [ZAŠTITA OD GREŠAKA] Pokrećem stvarna merenja na hardveru...")
iterations = 2000

# Toplo pokretanje (Warm-up) da se stabilizuje procesor
for _ in range(100):
    _ = validate_user_input_hardware(mp3_path, [0.5, 0.5, 0.5, 0.5])

# Test 1: Validan MP3
start = time.perf_counter()
for _ in range(iterations):
    _ = validate_user_input_hardware(mp3_path, [0.5, 0.5, 0.5, 0.5])
lat_valid_mp3 = ((time.perf_counter() - start) / iterations) * 1000

# Test 2: Validan WAV
start = time.perf_counter()
for _ in range(iterations):
    _ = validate_user_input_hardware(wav_path, [0.5, 0.5, 0.5, 0.5])
lat_valid_wav = ((time.perf_counter() - start) / iterations) * 1000

# Test 3: Sanacija (Fajl je dobar, ali vrednosti idu van opsega -> piše se log na disk)
start = time.perf_counter()
for _ in range(iterations):
    _ = validate_user_input_hardware(mp3_path, [9.9, -5.0, 1.2, 0.5])
lat_out_of_bounds = ((time.perf_counter() - start) / iterations) * 1000

# Test 4: Invalidan format (Ekspresno odbacivanje bez otvaranja fajla)
start = time.perf_counter()
for _ in range(iterations):
    _ = validate_user_input_hardware(txt_path, [0.5, 0.5, 0.5, 0.5])
lat_invalid_format = ((time.perf_counter() - start) / iterations) * 1000

print(f"   -> Stvarna latencija za validan MP3:    {lat_valid_mp3:.4f} ms")
print(f"   -> Stvarna latencija za validan WAV:    {lat_valid_wav:.4f} ms")
print(f"   -> Stvarna latencija za sanaciju [9.9]: {lat_out_of_bounds:.4f} ms")
print(f"   -> Stvarna latencija za neispravan TXT: {lat_invalid_format:.4f} ms")

# Čišćenje test fajlova
for p in [mp3_path, wav_path, txt_path]:
    if os.path.exists(p): os.remove(p)

# =====================================================================
# GENERISANJE FINALNOG GRAFIKONA
# =====================================================================
fig, ax = plt.subplots(figsize=(7, 4))
fig.patch.set_facecolor('#faf8f5')
ax.set_facecolor('#faf8f5')

input_types = ['Valid MP3\n(Header Check)', 'Valid WAV\n(Stream Open)', 'Out of Bounds\n(Sanitacija + Disk Log)', 'Invalid Format\n(Fast Reject TXT)']
latencies_ms = [lat_valid_mp3, lat_valid_wav, lat_out_of_bounds, lat_invalid_format]
bar_colors = ['#2e7d32', '#1b5e20', '#e65100', '#c62828']

bars = ax.barh(input_types, latencies_ms, color=bar_colors, height=0.5, edgecolor='grey', alpha=0.9)
ax.set_title("Real-World User Error Protection Latency (Hardware Measurement)", fontsize=11, fontweight='bold', color='#2d3748', pad=15)
ax.set_xlabel("Time (milliseconds - ms)", fontsize=10, color='#2d3748')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar in bars:
    xval = bar.get_width()
    ax.text(xval + (max(latencies_ms) * 0.02), bar.get_y() + bar.get_height() / 2.0, f'{xval:.4f} ms',
             ha='left', va='center', fontsize=9, fontweight='bold', color='#2d3748')

plt.tight_layout()
plt.savefig('usability_hardware_benchmark.png', dpi=300, facecolor='#faf8f5')
plt.close()
print("\n📊 Novi grafikon zasnovan na realnom I/O radu hardvera je sačuvan kao 'usability_hardware_benchmark.png'!")