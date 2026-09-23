from pathlib import Path

# Definisanje putanja sa apsolutnim rezonovanjem
CURRENT_DIR = Path(__file__).resolve().parent if '__file__' in locals() else Path.cwd()
PROJECT_ROOT_DIR = CURRENT_DIR.parent
EMBEDDINGS_DIR = PROJECT_ROOT_DIR / "data" / "processed" / "audio_embeddings"

print(f"--- FOLDER PATH CHECK ---")
print(f"Target Embeddings Directory: {EMBEDDINGS_DIR.resolve()}")
print(f"Does the directory exist? {EMBEDDINGS_DIR.exists()}")

if EMBEDDINGS_DIR.exists() and EMBEDDINGS_DIR.is_dir():
    # Prebrojavanje .npy fajlova u folderu
    npy_files = list(EMBEDDINGS_DIR.glob("*.npy"))
    print(f"Total .npy files found in directory: {len(npy_files)}")
else:
    print("WARNING: The directory does not exist or is not a valid folder!")

# Provera specifičnog fajla
track_id = "5iwz1NiezX7WWjnCgY5TH4"
specific_npy = EMBEDDINGS_DIR / f"{track_id}.npy"

print(f"\n--- SPECIFIC TRACK CHECK ---")
print(f"Checking file path: {specific_npy.resolve()}")
print(f"Does '5iwz1NiezX7WWjnCgY5TH4.npy' exist? {specific_npy.exists()}")
print(f"----------------------------")