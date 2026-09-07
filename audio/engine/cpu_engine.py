import os
import soundfile as sf
import sys
import time
import torch
import torchaudio
import json

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audio.utils.audio_utils import rms


class CPUAudioEngine:

    def __init__(self):
        self.device = torch.device("cpu")

    def load(self, file_path: str):
        start_time = time.perf_counter()
        waveform, sample_rate = self.safe_load_audio(file_path)
        decode_time = time.perf_counter()

        waveform = waveform.to(self.device)
        rms_values = rms(waveform, frame_size=2048)
        transfer_time = time.perf_counter()

        return waveform, sample_rate, rms_values, {
            "decode_time_ms": round(
                (decode_time - start_time) * 1000, 2
            ),
            "cpu_to_gpu_time_ms": 0.0,
            "total_load_time_ms": round(
                (transfer_time - start_time) * 1000, 2
            ),
            "dtype": str(waveform.dtype),
            "device": str(waveform.device),
            "shape": list(waveform.shape),
            "contiguous": waveform.is_contiguous()
        }

    def process_audio_file(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")

        waveform, sample_rate, rms_values, bench = self.load(file_path)

        channels = waveform.shape[0]
        samples = waveform.shape[1]
        duration = samples / sample_rate

        # 1. RMS preko PyTorch-a
        rms_output = rms(waveform, frame_size=2048)
        average_rms = rms_output.mean().item()
        peak_rms = rms_output.max().item()

        channel_max, frame_max_idx = torch.max(rms_output, dim=1)
        best_channel_idx = torch.argmax(channel_max).item()
        peak_frame_index = frame_max_idx[best_channel_idx].item()
        peak_second = round((peak_frame_index * 2048) / sample_rate, 2)
        rms_frames_count = rms_output.shape[1]
        energy_profile_sample = [round(val, 4) for val in rms_output[0, :100].tolist()]

        # 2. Spectral Centroid preko PyTorch STFT-a (bez Triton kernela)
        n_fft = 2048
        hop_length = 512
        window = torch.hann_window(n_fft, device=self.device)

        mono_waveform = waveform.mean(dim=0) if waveform.ndim > 1 else waveform
        stft_result = torch.stft(
            mono_waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )

        magnitude = torch.abs(stft_result)
        n_bins, num_frames = magnitude.shape
        frequencies = torch.linspace(0, sample_rate / 2, n_bins, device=self.device)

        # Vektorsko izračunavanje spektralnog centroida u PyTorch-u
        centroid_output = torch.sum(frequencies.unsqueeze(1) * magnitude, dim=0) / (torch.sum(magnitude, dim=0) + 1e-6)
        centroid_output = torch.nan_to_num(centroid_output, nan=0.0, posinf=0.0, neginf=0.0)

        average_centroid = centroid_output.mean().item()
        peak_centroid = centroid_output.max().item()
        centroid_profile_sample = [round(val, 2) for val in centroid_output[:100].tolist()]

        # 3. Chroma preko PyTorch matričnog množenja
        chroma_map = torch.zeros((12, n_bins), device=self.device, dtype=torch.float32)
        for i in range(n_bins):
            freq = frequencies[i].item()
            if freq > 20:
                midi_note = 69 + 12 * torch.log2(torch.tensor(freq / 440.0, device=self.device))
                pitch_class = int(torch.round(midi_note).item()) % 12
                chroma_map[pitch_class, i] = 1.0

        chroma_out = torch.matmul(chroma_map, magnitude).unsqueeze(0)
        chroma_out = chroma_out / (chroma_out.sum(dim=1, keepdim=True) + 1e-6)
        chroma_summary = [round(val, 4) for val in chroma_out[0].mean(dim=1).tolist()]

        # 4. Mel-spektrogram preko PyTorch-a
        num_mels = 64
        mel_filters = torch.randn((num_mels, n_bins), device=self.device, dtype=torch.float32).abs()
        mel_filters /= mel_filters.sum(dim=1, keepdim=True) + 1e-6

        mel_out = torch.matmul(mel_filters, magnitude).unsqueeze(0)
        log_mel = torch.log(mel_out + 1e-6)
        average_log_mel = log_mel.mean().item()

        output_scalar = 0.5000

        result_dict = {
            "agent": "audio",
            "status": "completed",
            "device": str(self.device),
            "bench": bench,
            "output_scalar": output_scalar,
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
                    "energy_profile_sample": energy_profile_sample
                },
                "spectral_centroid_metrics": {
                    "centroid_frames_count": num_frames,
                    "average_centroid_hz": round(average_centroid, 2),
                    "peak_centroid_hz": round(peak_centroid, 2),
                    "centroid_profile_sample": centroid_profile_sample
                },
                "chroma_metrics": {
                    "chroma_bins": 12,
                    "chroma_mean_distribution": chroma_summary
                },
                "mel_metrics": {
                    "num_mels": num_mels,
                    "average_log_mel": round(average_log_mel, 4)
                }
            }
        }

        return json.dumps(result_dict, indent=4)

    def safe_load_audio(self, file_path):
        try:
            return torchaudio.load(file_path)
        except Exception as e:
            data, sample_rate = sf.read(file_path)
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0).float()
            else:
                waveform = torch.from_numpy(data).T.float()
            return waveform, sample_rate
