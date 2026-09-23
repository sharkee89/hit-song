import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# Uvozimo tvog AudioAgent-a za ekstrakciju 2048 dimenzija
from audio.agent.audio_agent import AudioAgent

# Putanja do sačuvanih težina tvog modela
MODEL_WEIGHTS_PATH = PROJECT_ROOT_DIR / "audio" / "models" / "audio_quality_model.pt"


# 1. Definicija neuronske mreže koja prima 2048 dimenzija i vraća skalar
class AudioQualityScorer(nn.Module):
    def __init__(self, input_dim=2048):
        super(AudioQualityScorer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()  # Vraća vrednost između 0 i 1

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return self.sigmoid(x)


def load_trained_model(model_path: Path):
    """Inicijalizuje model i učitava trenirane težine."""
    model = AudioQualityScorer(input_dim=2048)

    if model_path.exists():
        try:
            # Učitavanje sačuvanih težina
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f"Uspešno učitane težine modela sa: {model_path}")
        except Exception as e:
            print(f"Upozorenje: Nije uspelo učitavanje težina ({e}), koriste se inicijalne težine.")
    else:
        print(f"Upozorenje: Fajl sa težinama ne postoji na {model_path}. Pokrećem sa inicijalnim stanjem.")

    model.eval()
    return model


def calibrate_score(raw_score: float, min_val: float = 0.25, max_val: float = 0.65) -> float:
    """
    Vrši Min-Max normalizaciju da razvuče uski opseg modela na 0.0 - 1.0.
    min_val i max_val prilagođavaš na osnovu empirijskih rezultata sa tvog validacionog skupa.
    """
    clipped = max(min_val, min(max_val, raw_score))
    normalized = (clipped - min_val) / (max_val - min_val)
    return float(normalized)


def main():
    # Uzimamo putanju do audio fajla iz .env fajla ili argumenta
    audio_source = os.getenv("AUDIO_FILE_PATH", "")

    if not audio_source or not os.path.exists(audio_source):
        print(f"GREŠKA: AUDIO_FILE_PATH nije definisan u .env ili fajl ne postoji: {audio_source}")
        print("Primer u .env fajlu: AUDIO_FILE_PATH=data/test_song.mp3")
        return

    print(f"\n[1/3] Pokrećem Audio Agent za ekstrakciju obeležja...")
    agent = AudioAgent()

    try:
        with open(audio_source, "rb") as f:
            import io
            audio_bytes = io.BytesIO(f.read())
            embedding = agent.extract_features(audio_bytes)
    except Exception as e:
        print(f"Greška pri ekstrakciji embedding-a: {e}")
        return

    query_emb = np.array(embedding, dtype=np.float32)
    print(f"Izvučen embedding oblika: {query_emb.shape} (Očekivano: 2048)")

    # Priprema tenzora za model
    x = torch.tensor(query_emb).unsqueeze(0)  # Oblik: [1, 2048]

    print(f"[2/3] Učitavam trenirani model za procenu kvaliteta...")
    model = load_trained_model(MODEL_WEIGHTS_PATH)

    print(f"[3/3] Računam i kalibrišem akustični skor...")
    with torch.no_grad():
        score_tensor = model(x)
        raw_score = score_tensor.item()

    # Primena Min-Max kalibracije
    acoustic_score = calibrate_score(raw_score, min_val=0.25, max_val=0.65)

    print("\n" + "=" * 45)
    print(f" REZULTAT AUDIO AGENTA (LAKMUS PAPIR)")
    print("=" * 45)
    print(f" Fajl: {Path(audio_source).name}")
    print(f" Sirovi skor modela:   {raw_score:.4f}")
    print(f" Kalibrisani skor:     {acoustic_score:.4f} (Raspon: 0.0 - 1.0)")

    if acoustic_score > 0.6:
        print(" Status: Visok standard produkcije / Komercijalni profil.")
    elif acoustic_score > 0.4:
        print(" Status: Prosečna produkcija / Granični kvalitet.")
    else:
        print(" Status: Amaterski snimak / Nizak akustični kvalitet.")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()