# Hit Song Predictor (Flutter)

Flutter front-end for the hit-song research pipeline. The user enters an artist name and
uploads an MP3/WAV file; the app calls the Python API (`../api/server.py`), which reuses the
existing project code:

| Phase | What runs | Source |
|---|---|---|
| 1 · ML (≈5 s) | Peak 30 s window → Librosa (31) + Audiobox Aesthetics (PQ, PC, CE, CU) → TalentAgent → XGBoost + TreeSHAP | `audio/utils/audio_utils.py`, `inference/inference_hit.py` |
| 2 · LLM (≈30–60 s) | Gemini + GPT-audio listen to the file, DeepSeek builds the consensus | `inference/inference_hit.get_ai_analysis` |

## Run

1. Start the API from the project root (PowerShell):

   ```powershell
   .\run_api.ps1
   ```

   Wait for `[api] Models warmed up and ready.` (Audiobox model loads once).

2. Start the app:

   ```powershell
   cd flutter_app
   flutter run -d chrome
   ```

   To point the app at another machine, add `--dart-define=API_URL=http://HOST:8000`.

## Notes

- API keys come from the project `.env` (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`).
- The API trusts the Windows certificate store (`truststore`), which is required while
  Avast Web Shield intercepts HTTPS on this machine.
- Windows desktop builds need the Visual Studio "Desktop development with C++" workload;
  web (Chrome) works out of the box.
