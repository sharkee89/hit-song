











import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audio.engine.cpu_engine import CPUAudioEngine
from audio.engine.gpu_engine import GPUAudioEngine

class AudioAgent:

    def __init__(self):
        if torch.cuda.is_available():
            self.engine = GPUAudioEngine()
            self.device = "cuda"
        else:
            self.engine = CPUAudioEngine()
            self.device = "cpu"
        print(f"[AudioAgent] Device: {self.device}")

    def process_track(self, file_path: str) -> dict:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")

        waveform, sample_rate, bench = self.engine.load(file_path)

        channels = waveform.shape[0]
        samples = waveform.shape[1]
        duration = samples / sample_rate

        print("shape:", waveform.shape)
        print("stride:", waveform.stride())
        print("dtype:", waveform.dtype)
        print("device:", waveform.device)
        print("contiguous:", waveform.is_contiguous())

        return {
            "agent": "audio",
            "status": "completed",
            "device": self.device,
            "bench": bench,
            "audio": {
                "sample_rate": sample_rate,
                "channels": channels,
                "samples": samples,
                "duration_sec": round(duration, 2)
            }
        }

if __name__ == "__main__":
    file_path = r"C:\Users\Sharkee\Downloads\Aylex_-_Live_It_(freetouse.com).mp3"
    agent = AudioAgent()
    result = agent.process_track(file_path)
    print(result)
