import torch
import torchaudio


class CPUAudioEngine:

    def __init__(self):
        self.device = torch.device("cpu")

    def load(self, file_path: str):
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate
