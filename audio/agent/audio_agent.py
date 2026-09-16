import io
import json
import os
import sys
import librosa
import numpy as np
import soundfile as sf
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from dotenv import load_dotenv

# Ensure root path resolution
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audio.ai.ai_engine import AIEngine


class AudioAgent:

    def __init__(self, device: str = None):
        # 1. Device and Engine setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda":
            try:
                from audio.engine.gpu_engine import GPUAudioEngine
                from audio.kernels.rms_kernel import triton_rms
                self.engine = GPUAudioEngine()
            except ImportError as e:
                print(f"⚠️ CUDA is unavailable or engine missing ({e}). Switching to CPU.")
                from audio.engine.cpu_engine import CPUAudioEngine
                self.engine = CPUAudioEngine()
                self.device = "cpu"
        else:
            from audio.engine.cpu_engine import CPUAudioEngine
            self.engine = CPUAudioEngine()

        print(f"[AudioAgent] Execution Device: {self.device}")

        # 2. ResNet-50 Embedding Extraction Model Setup
        weights = ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        self.embedding_model = torch.nn.Sequential(*(list(resnet.children())[:-1])).to(self.device)
        self.embedding_model.eval()

    def _find_best_offset_samples(self, y: np.ndarray, sr: int = 22050, target_duration: float = 30.0) -> int:
        """
        Calculates starting sample index corresponding to peak RMS energy density.
        """
        target_samples = int(sr * target_duration)
        if len(y) <= target_samples:
            return 0

        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        window_frames = int(target_samples / hop_length)

        if len(rms) <= window_frames:
            return 0

        energy_profile = np.convolve(rms, np.ones(window_frames), mode="valid")
        max_frame_idx = int(np.argmax(energy_profile))
        start_sample = max_frame_idx * hop_length

        return min(start_sample, len(y) - target_samples)

    def extract_features(self, source: str | io.BytesIO) -> list:
        """
        Extracts single 2048-dimensional ResNet-50 embedding vector from local path or byte buffer
        using the optimal 30s peak RMS energy window.
        """
        sr = 22050
        target_duration = 30.0
        target_samples = int(sr * target_duration)

        try:
            y, _ = librosa.load(source, sr=sr, mono=True)
        except Exception as err:
            raise ValueError(f"Failed to decode audio source: {err}")

        if len(y) == 0:
            raise ValueError("Loaded audio signal is empty.")

        if len(y) > target_samples:
            start_sample = self._find_best_offset_samples(y, sr=sr, target_duration=target_duration)
            y = y[start_sample: start_sample + target_samples]
        else:
            y = np.pad(y, (0, target_samples - len(y)))

        return self._compute_embedding_from_signal(y, sr)

    def extract_sliding_features(self, source: str | io.BytesIO, window_duration: float = 30.0, step_duration: float = 10.0) -> list[list]:
        """
        Extracts multiple 2048-dimensional ResNet-50 embedding vectors across the entire audio file
        using a sliding window approach.
        """
        sr = 22050
        try:
            y, _ = librosa.load(source, sr=sr, mono=True)
        except Exception as err:
            raise ValueError(f"Failed to decode audio source: {err}")

        if len(y) == 0:
            raise ValueError("Loaded audio signal is empty.")

        window_samples = int(sr * window_duration)
        step_samples = int(sr * step_duration)

        # If audio is shorter than window, fall back to standard single extraction
        if len(y) <= window_samples:
            return [self.extract_features(source)]

        embeddings = []
        for start_sample in range(0, len(y) - window_samples + 1, step_samples):
            segment = y[start_sample:start_sample + window_samples]
            emb = self._compute_embedding_from_signal(segment, sr)
            embeddings.append(emb)

        # Handle tail end if not cleanly divisible
        if (len(y) - window_samples) % step_samples != 0:
            segment = y[-window_samples:]
            emb = self._compute_embedding_from_signal(segment, sr)
            embeddings.append(emb)

        return embeddings

    def _compute_embedding_from_signal(self, y: np.ndarray, sr: int) -> list:
        """Internal helper to convert a 30s audio signal numpy array into a ResNet-50 embedding vector."""
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512
        )
        log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)

        log_mel_scaled = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)

        mel_rgb = np.stack([log_mel_scaled] * 3, axis=-1)
        mel_rgb = np.transpose(mel_rgb, (2, 0, 1))

        tensor_input = torch.tensor(mel_rgb, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.embedding_model(tensor_input)
            features = features.squeeze().cpu().numpy().tolist()

        return features

    def process_audio_file(self, file_path: str) -> str:
        """Processes audio through low-level C++/GPU/CPU engine for raw audio features."""
        return self.engine.process_audio_file(file_path)

    def get_ai_analysis(self, audio_data):
        """Passes extracted features to LLM / AIEngine for market analysis."""
        ai_engine = AIEngine()
        return ai_engine.get_audio_detail_analysis(audio_data)

    def get_data(self, audio_data: str, audio_detail_analysis, resnet_vector: list) -> str:
        """Combines raw feature extraction, AI analysis, and deep ResNet embedding vector into JSON."""
        audio_data_dict = json.loads(audio_data)
        audio_data_dict["ai_analysis"] = audio_detail_analysis
        audio_data_dict["resnet_embedding"] = resnet_vector
        return json.dumps(audio_data_dict, indent=4)

    def process(self, file_path: str) -> str:
        """
        Full end-to-end execution pipeline combining audio DSP, deep embeddings, and LLM market report.
        """
        audio_data = self.process_audio_file(file_path)
        audio_detail_analysis = self.get_ai_analysis(audio_data)
        resnet_vector = self.extract_features(file_path)
        return self.get_data(audio_data, audio_detail_analysis, resnet_vector)


if __name__ == "__main__":
    load_dotenv()
    file_path = os.getenv("AUDIO_FILE_PATH", "")
    agent = AudioAgent()

    if file_path and os.path.exists(file_path):
        print("🚀 Starting full AudioAgent pipeline processing...")
        result_json = agent.process(file_path)
        data = json.loads(result_json)
        print("✅ Pipeline executed successfully!")
        print(data)
    else:
        print("❌ AUDIO_FILE_PATH isn't defined or file doesn't exist in .env.")