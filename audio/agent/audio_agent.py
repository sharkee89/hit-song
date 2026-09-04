import os
import sys
import json
import torch
import triton
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audio.kernels.rms_kernel import triton_rms
from audio.kernels.spectral_centroid_kernel import spectral_centroid_kernel
from audio.kernels.triton_chroma_kernel import triton_chroma_kernel
from audio.kernels.triton_mel_kernel import triton_mel_kernel


class AudioAgent:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            try:
                from audio.engine.gpu_engine import GPUAudioEngine
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
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")

        waveform, sample_rate, data, bench = self.engine.load(file_path)

        channels = waveform.shape[0]
        samples = waveform.shape[1]
        duration = samples / sample_rate

        # 1. Izračunavanje RMS preko Triton kernela
        rms_output = triton_rms(waveform, frame_size=2048)
        average_rms = rms_output.mean().item()
        peak_rms = rms_output.max().item()

        channel_max, frame_max_idx = torch.max(rms_output, dim=1)
        best_channel_idx = torch.argmax(channel_max).item()
        peak_frame_index = frame_max_idx[best_channel_idx].item()
        peak_second = round((peak_frame_index * 2048) / sample_rate, 2)
        rms_frames_count = rms_output.shape[1]
        energy_profile_sample = [round(val, 4) for val in rms_output[0, :100].tolist()]

        # 2. Izračunavanje Spectral Centroid preko Triton kernela
        n_fft = 2048
        hop_length = 512
        window = torch.hann_window(n_fft, device=waveform.device)

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
        frequencies = torch.linspace(0, sample_rate / 2, n_bins, device=waveform.device)

        centroid_output = torch.empty(num_frames, device=waveform.device, dtype=torch.float32)
        spectral_centroid_kernel[(num_frames,)](
            frequencies,
            magnitude,
            centroid_output,
            n_bins,
            BLOCK_SIZE=min(1024, n_bins)
        )
        centroid_output = torch.nan_to_num(centroid_output, nan=0.0, posinf=0.0, neginf=0.0)
        average_centroid = centroid_output.mean().item()
        peak_centroid = centroid_output.max().item()
        centroid_profile_sample = [round(val, 2) for val in centroid_output[:100].tolist()]

        # 3. Izračunavanje Chroma preko Triton kernela
        # Kreiranje hromatske mape (12 tonova x n_bins)
        chroma_map = torch.zeros((12, n_bins), device=waveform.device, dtype=torch.float32)
        for i in range(n_bins):
            freq = frequencies[i].item()
            if freq > 20:
                midi_note = 69 + 12 * torch.log2(torch.tensor(freq / 440.0))
                pitch_class = int(torch.round(midi_note).item()) % 12
                chroma_map[pitch_class, i] = 1.0

        chroma_out = torch.zeros((1, 12, num_frames), device=waveform.device, dtype=torch.float32)
        grid_chroma = lambda meta: (triton.cdiv(num_frames, meta['BLOCK_SIZE_N']), 1)
        triton_chroma_kernel[grid_chroma](
            magnitude.unsqueeze(0),
            chroma_map,
            chroma_out,
            num_bins=n_bins,
            num_frames=num_frames,
            stride_spec_batch=magnitude.numel(),
            stride_spec_bin=magnitude.stride(0),
            stride_spec_frame=magnitude.stride(1),
            stride_out_batch=chroma_out.stride(0),
            stride_out_chr=chroma_out.stride(1),
            stride_out_frame=chroma_out.stride(2)
        )
        chroma_summary = [round(val, 4) for val in chroma_out[0].mean(dim=1).tolist()]

        # 4. Izračunavanje Mel-spektrograma preko Triton kernela
        num_mels = 64
        mel_filters = torch.randn((num_mels, n_bins), device=waveform.device, dtype=torch.float32).abs()
        mel_filters /= mel_filters.sum(dim=1, keepdim=True) + 1e-6

        mel_out = torch.zeros((1, num_mels, num_frames), device=waveform.device, dtype=torch.float32)
        grid_mel = lambda meta: (
            triton.cdiv(num_mels, meta['BLOCK_SIZE_M']),
            triton.cdiv(num_frames, meta['BLOCK_SIZE_N']),
            1
        )
        triton_mel_kernel[grid_mel](
            magnitude.unsqueeze(0),
            mel_filters,
            mel_out,
            num_bins=n_bins,
            num_mels=num_mels,
            num_frames=num_frames,
            stride_spec_batch=magnitude.numel(),
            stride_spec_bin=magnitude.stride(0),
            stride_spec_frame=magnitude.stride(1),
            stride_out_batch=mel_out.stride(0),
            stride_out_mel=mel_out.stride(1),
            stride_out_frame=mel_out.stride(2)
        )
        average_log_mel = mel_out.mean().item()

        # Placeholder za izlazni skalar koji će kasnije definisati neuronska mreža
        output_scalar = 0.5000

        result_dict = {
            "agent": "audio",
            "status": "completed",
            "device": self.device,
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


if __name__ == "__main__":
    load_dotenv()
    file_path = os.getenv("AUDIO_FILE_PATH", "")
    agent = AudioAgent()
    if file_path:
        json_result = agent.process_audio_file(file_path)
        print(json_result)
    else:
        print("❌ AUDIO_FILE_PATH nije definisan u .env fajlu.")
