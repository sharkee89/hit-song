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
MODEL_WEIGHTS_PATH = PROJECT_ROOT_DIR / "models" / "audio_agent_predictor.pth"


# 1. Arhitektura MORA biti identična onoj u skripti za treniranje!
class AudioAgentPredictor(nn.Module):
    def __init__(self, input_dim=2048):
        super(AudioAgentPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()  # Vraća vrednost između 0 i 1
        )

    def forward(self, x):
        return self.network(x)


def load_trained_model(model_path: Path):
    """Inicijalizuje model i učitava trenirane težine."""
    model = AudioAgentPredictor(input_dim=2048)

    if model_path.exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f"Uspešno učitane težine modela sa: {model_path}")
        except Exception as e:
            print(f"Upozorenje: Nije uspelo učitavanje težina ({e}), koriste se inicijalne težine.")
    else:
        print(f"Upozorenje: Fajl sa težinama ne postoji na {model_path}. Pokrećem sa inicijalnim stanjem.")

    model.eval()
    return model


def calibrate_score(raw_score: float, min_val: float = 0.25, max_val: float = 0.65) -> float:
    """Vrši Min-Max normalizaciju da razvuče opseg modela na 0.0 - 1.0."""
    clipped = max(min_val, min(max_val, raw_score))
    normalized = (clipped - min_val) / (max_val - min_val)
    return float(normalized)


def main():
    audio_source = os.getenv("AUDIO_FILE_PATH", "")

    if not audio_source or not os.path.exists(audio_source):
        print(f"GREŠKA: AUDIO_FILE_PATH nije definisan u .env ili fajl ne postoji: {audio_source}")
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
    x = torch.tensor(query_emb).unsqueeze(0)  # Oblik: [1, 2048]

    print(f"[2/3] Učitavam trenirani model za procenu kvaliteta...")
    model = load_trained_model(MODEL_WEIGHTS_PATH)

    print(f"[3/3] Računam i kalibrišem akustični skor...")
    with torch.no_grad():
        score_tensor = model(x)
        raw_score = score_tensor.item()

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