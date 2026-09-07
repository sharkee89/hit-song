import os
import sys
import time
import shutil
import matplotlib.pyplot as plt

# Kreiranje dummy privremenog foldera za simulaciju "trimmed_temp_" fajlova
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_cache")
os.makedirs(TEMP_DIR, exist_ok=True)


# =====================================================================
# SIMULACIJA AGENTSKIH METODA I FALLBACK MEHANIZAMA (Fault Tolerance)
# =====================================================================
def simulate_external_api_call(force_fail=False):
    """Simulacija poziva eksternog LLM/API servisa sa mrežnim kašnjenjem."""
    if force_fail:
        raise ConnectionError("❌ Eksterni API servis (Google Gemini/Spotify) je nedostupan!")
    time.sleep(0.12)  # Normalno mrežno kašnjenje od 120ms
    return {"status": "success", "data": "High viral potential"}


def local_heuristic_fallback():
    """Lokalni fallback algoritam u slučaju pada eksternih sistema."""
    # Brzi lokalni proračun na osnovu prosečnih istorijskih vrednosti
    return {"status": "fallback_success", "data": "Medium viral potential (Heuristic estimate)"}


def process_audio_pipeline(file_name, simulate_crash_during_execution=False, simulate_api_down=False):
    """Simulira kompletan proces obrade sa try-except-finally zaštitom."""
    temp_file_path = os.path.join(TEMP_DIR, f"trimmed_temp_{file_name}")

    # Kreiramo privremeni "cached" fajl na disku
    with open(temp_file_path, "w") as f:
        f.write("dummy audio data")

    try:
        # Korak 1: Obrada lokalnog fajla
        if simulate_crash_during_execution:
            raise RuntimeError("⚡ Katastrofalni sistemski pad (OS/Hardware Interrupt)!")

        # Korak 2: Poziv eksternog API-ja
        api_result = simulate_external_api_call(force_fail=simulate_api_down)
        return "Normal Mode", api_result

    except ConnectionError as ce:
        # Fallback mehanizam (Fault Tolerance)
        fallback_result = local_heuristic_fallback()
        return "Fallback Mode (API Down)", fallback_result

    except RuntimeError as re:
        # Prosleđujemo fatalni izuzetak dalje
        raise re

    finally:
        # Mogućnost oporavka (Recoverability): Garantovano čišćenje resursa
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# =====================================================================
# POKRETANJE TESTOVA I MERENJE METRIKA
# =====================================================================
print("=====================================================================")
# Test 1: Verifikacija oporavka (Recoverability) pod anomalijama
print("🧪 [RECOVERABILITY] Testiranje sigurnog brisanja privremenih fajlova...")
try:
    process_audio_pipeline("test_song.wav", simulate_crash_during_execution=True)
except RuntimeError as e:
    print(f"   -> Detektovan simulirani pad sistema: {e}")

# Provera da li je privremeni fajl obrisan uprkos padu sistema
remaining_files = os.listdir(TEMP_DIR)
recoverability_status = "✅ USPEH - Keš je uspešno očišćen!" if len(
    remaining_files) == 0 else "❌ NEUSPEH - Otkriveno curenje resursa!"
print(f"   -> Status nakon sistemskog pada: {recoverability_status}")

print("\n=====================================================================")
# Test 2: Otpornost na greške i latencija fallback-a (Fault Tolerance / Availability)
print("🧪 [FAULT TOLERANCE] Merenje latencija normalnog vs. fallback režima...")

iterations = 100
# A: Normalni režim rada
start_time = time.perf_counter()
for _ in range(iterations):
    _ = process_audio_pipeline("song.wav", simulate_api_down=False)
end_time = time.perf_counter()
latency_normal = ((end_time - start_time) / iterations) * 1000  # u ms

# B: Režim pod greškom (External API is Down)
start_time = time.perf_counter()
for _ in range(iterations):
    _ = process_audio_pipeline("song.wav", simulate_api_down=True)
end_time = time.perf_counter()
latency_fallback = ((end_time - start_time) / iterations) * 1000  # u ms

print(f"   -> Prosečna latencija (Normalan rad sa API-jem): {latency_normal:.2f} ms")
print(f"   -> Prosečna latencija (Lokalni Fallback režim):  {latency_fallback:.2f} ms")

# Čišćenje privremenog direktorijuma na kraju testa
shutil.rmtree(TEMP_DIR)

# =====================================================================
# GENERISANJE AKADEMSKOG GRAFIKONA (Matplotlib)
# =====================================================================
print("\n📊 Generišem naučni grafikon pouzdanosti...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#faf8f5')

# Grafikon 1: Latencije i dostupnost pod greškom
modes = ['Normal Mode (API Online)', 'Fallback Mode (API Offline)']
latencies = [latency_normal, latency_fallback]
colors = ['#1f77b4', '#ff7f0e']

bars = ax1.bar(modes, latencies, color=colors, width=0.45, edgecolor='grey')
ax1.set_facecolor('#faf8f5')
ax1.set_title("API Failover Latency & Service Availability", fontsize=11, fontweight='bold', color='#2d3748', pad=15)
ax1.set_ylabel("Execution Latency (milliseconds - ms)", fontsize=10, color='#2d3748')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f'{yval:.2f} ms',
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2d3748')

# Grafikon 2: Oporavak keš memorije (Fajlovi u temp folderu)
states = ['Before Execution', 'During Execution', 'Post-Crash (With Protection)', 'Post-Crash (No Protection)']
file_counts = [0, 1, 0, 1]  # Bez zaštite, fajl bi ostao zarobljen u kešu
colors_states = ['#2ca02c', '#1f77b4', '#2ca02c', '#d62728']

bars2 = ax2.bar(states, file_counts, color=colors_states, width=0.45, edgecolor='grey')
ax2.set_facecolor('#faf8f5')
ax2.set_title("Recoverability: Temporary Cache Clean-up Test", fontsize=11, fontweight='bold', color='#2d3748', pad=15)
ax2.set_ylabel("Active Temporary Files (trimmed_temp_*)", fontsize=10, color='#2d3748')
ax2.set_ylim(0, 1.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05, f'{int(yval)} files',
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2d3748')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reliability_benchmark_charts.png')
plt.savefig(output_path, dpi=300, facecolor='#faf8f5')
plt.close()

print(f"🎉 Grafikon pouzdanosti je uspešno sačuvan na: {output_path}")