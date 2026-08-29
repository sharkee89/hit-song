import time
import torch
import torchaudio


class GPUAudioEngine:

    def __init__(self):
        self.device = torch.device("cuda")

    def load(self, file_path: str):
        start_time = time.perf_counter()
        waveform, sample_rate = torchaudio.load(file_path)

        decode_time = time.perf_counter()
        waveform = waveform.to(self.device)
        torch.cuda.synchronize(self.device)

        transfer_time = time.perf_counter()

        return waveform, sample_rate, {
            "decode_time_ms": round(
                (decode_time - start_time) * 1000, 2
            ),
            "cpu_to_gpu_time_ms": round(
                (transfer_time - decode_time) * 1000, 2
            ),
            "total_load_time_ms": round(
                (transfer_time - start_time) * 1000, 2
            ),
            "dtype": str(waveform.dtype),
            "device": str(waveform.device),
            "shape": list(waveform.shape),
            "contiguous": waveform.is_contiguous()
        }
