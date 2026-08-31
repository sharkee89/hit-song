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


class AudioAgent:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            try:
                from audio.engine.gpu_audio import GPUAudioEngine
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

    def process_audio_file(self, file_path: str) -> dict:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")

        waveform, sample_rate, data, bench = self.engine.load(file_path)

        # Svođenje na mono kanal pre traženja pika da indeks odgovara vremenskoj liniji jednog kanala
        if data.ndim > 1:
            rms_mono = data.mean(dim=0)
        else:
            rms_mono = data

        if rms_mono.numel() > 0:
            peak_val, peak_index = torch.max(rms_mono, dim=0)
            peak_rms = peak_val.item()
            frame_size = 2048
            peak_second = round((peak_index.item() * frame_size) / sample_rate, 2)
        else:
            peak_rms = 0.0
            peak_second = 0.0

        rms_flat = data.flatten()
        rms_frames_count = rms_flat.numel()
        average_rms = rms_flat.mean().item() if rms_frames_count > 0 else 0.0
        peak_rms = rms_flat.max().item() if rms_frames_count > 0 else 0.0

        channels = waveform.shape[0]
        samples = waveform.shape[1]
        duration = samples / sample_rate

        return {
            "agent": "audio",
            "status": "completed",
            "device": self.device,
            "bench": bench,
            "audio": {
                "sample_rate": sample_rate,
                "channels": channels,
                "samples": samples,
                "duration_sec": round(duration, 2),
                "rms_metrics": {
                    "rms_frames_count": rms_frames_count,
                    "average_rms": round(average_rms, 4),
                    "peak_rms": round(peak_rms, 4),
                    "peak_second": peak_second,
                    "energy_profile_sample": [round(val, 4) for val in rms_flat[:100].tolist()]
                }
            }
        }


if __name__ == "__main__":
    load_dotenv()
    file_path = os.getenv("AUDIO_FILE_PATH", "")
    agent = AudioAgent()
    result = agent.process_audio_file(file_path)
    print(result)