import json
import os
import sys
import torch
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audio.ai.ai_engine import AIEngine

class AudioAgent:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            try:
                from audio.engine.gpu_engine import GPUAudioEngine
                from audio.kernels.rms_kernel import triton_rms
                self.engine = GPUAudioEngine()
            except ImportError as e:
                print(f"⚠️ CUDA is unavailable, error during import gpu engine ({e}). Switching to CPU.")
                from audio.engine.cpu_engine import CPUAudioEngine
                self.engine = CPUAudioEngine()
                self.device = "cpu"
        else:
            from audio.engine.cpu_engine import CPUAudioEngine
            self.engine = CPUAudioEngine()

        print(f"[AudioAgent] Device: {self.device}")

    def process_audio_file(self, file_path: str) -> str:
        return self.engine.process_audio_file(file_path)

    def get_ai_analysis(self, audio_data):
        ai_engine = AIEngine()
        return ai_engine.get_audio_detail_analysis(audio_data)

    def get_data(self, audio_data, audio_detail_analysis):
        audio_data_dict = json.loads(audio_data)
        audio_data_dict["ai_analysis"] = audio_detail_analysis
        return json.dumps(audio_data_dict, indent=4)

    def process(self, file_path: str) -> str:
        audio_data = self.process_audio_file(file_path)
        audio_detail_analysis = self.get_ai_analysis(audio_data)
        return self.get_data(audio_data, audio_detail_analysis)

if __name__ == "__main__":
    load_dotenv()
    file_path = os.getenv("AUDIO_FILE_PATH", "")
    agent = AudioAgent()
    if file_path:
        print(agent.process(file_path))
    else:
        print("❌ AUDIO_FILE_PATH nije definisan u .env fajlu.")
