import json
import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

audio_env_path = os.getenv("AUDIO_FILE_PATH", "")
AUDIO_FILE = Path(audio_env_path) if audio_env_path else PROJECT_ROOT / "test" / "sample.mp3"


def main():
    if not AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_FILE}\n"
            "Proveri da li je AUDIO_FILE_PATH dobro definisan u .env fajlu."
        )

    input_jsonl = PROJECT_ROOT / "test" / "input.jsonl"
    input_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(input_jsonl, "w", encoding="utf-8") as f:
        json.dump(
            {
                "path": str(AUDIO_FILE.resolve())
            },
            f,
        )
        f.write("\n")

    print(f"Analiziram: {AUDIO_FILE.name}...")

    result = subprocess.run(
        [
            "audio-aes",
            str(input_jsonl),
            "--batch-size",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("AudioBox-Aesthetics nije uspeo.")

    # Nalazimo liniju koja počinje sa JSON objektom '{'
    data = None
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if data:
        print("\n" + "=" * 45)
        print(" REZULTAT AUDIOBOX-AESTHETICS ANALIZE")
        print("=" * 45)
        print(f" Fajl: {AUDIO_FILE.name}")
        print(f" Production Quality (PQ):     {data.get('PQ'):.2f} / 10")
        print(f" Production Complexity (PC):  {data.get('PC'):.2f} / 10")
        print(f" Content Enjoyment (CE):      {data.get('CE'):.2f} / 10")
        print(f" Content Usefulness (CU):     {data.get('CU'):.2f} / 10")
        print("=" * 45 + "\n")
    else:
        print("RAW OUTPUT:", result.stdout)
        print("Nisam uspeo da parsiram JSON rezultat.")


if __name__ == "__main__":
    main()