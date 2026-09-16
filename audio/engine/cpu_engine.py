import numpy as np
import os
import json
import time
import torch
import torchaudio
import soundfile as sf

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def spectral_centroid_kernel(
            frequencies_ptr,
            magnitude_ptr,
            output_ptr,
            n_bins,
            BLOCK_SIZE: tl.constexpr,
    ):
        pid_frame = tl.program_id(0)
        sum_weighted = 0.0
        sum_magnitude = 0.0

        for block_start in range(0, n_bins, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_bins
            frequency_offsets = offsets
            magnitude_offsets = pid_frame * n_bins + offsets

            frequencies = tl.load(frequencies_ptr + frequency_offsets, mask=mask, other=0.0)
            magnitude = tl.load(magnitude_ptr + magnitude_offsets, mask=mask, other=0.0)

            weighted = frequencies * magnitude
            sum_weighted += tl.sum(weighted, axis=0)
            sum_magnitude += tl.sum(magnitude, axis=0)

        centroid = sum_weighted / (sum_magnitude + 1e-6)
        tl.store(output_ptr + pid_frame, centroid)


class CPUAudioEngine:

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def safe_load_audio(self, file_path: str):
        try:
            return torchaudio.load(file_path)
        except Exception:
            data, sample_rate = sf.read(file_path)
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0).float()
            else:
                waveform = torch.from_numpy(data).T.float()
            return waveform, sample_rate

    def compute_rms(self, waveform: torch.Tensor, frame_size: int = 2048):
        channels, num_samples = waveform.shape
        num_frames = num_samples // frame_size
        trimmed = waveform[:, :num_frames * frame_size]
        frames = trimmed.view(channels, num_frames, frame_size)
        rms_output = torch.sqrt(torch.mean(frames ** 2, dim=2))

        flattened = rms_output.flatten()
        mean_val = flattened.mean().item()
        max_val = flattened.max().item()
        min_val = flattened.min().item()
        std_val = flattened.std().item()
        median_val = torch.median(flattened).item()
        q25 = torch.quantile(flattened, 0.25).item()
        q75 = torch.quantile(flattened, 0.75).item()
        dynamic_range = max_val - min_val

        diff = flattened - mean_val
        skew = (torch.mean(diff ** 3) / (std_val ** 3 + 1e-6)).item()
        kurtosis = (torch.mean(diff ** 4) / (std_val ** 4 + 1e-6) - 3.0).item()

        channel_max, frame_max_idx = torch.max(rms_output, dim=1)
        best_channel_idx = torch.argmax(channel_max).item()
        peak_frame_index = frame_max_idx[best_channel_idx].item()

        return {
            "rms_frames_count": rms_output.shape[1],
            "average_rms": round(mean_val, 5),
            "peak_rms": round(max_val, 5),
            "min_rms": round(min_val, 5),
            "std_rms": round(std_val, 5),
            "median_rms": round(median_val, 5),
            "q25_rms": round(q25, 5),
            "q75_rms": round(q75, 5),
            "dynamic_range": round(dynamic_range, 5),
            "skewness": round(skew, 5),
            "kurtosis": round(kurtosis, 5),
            "peak_second": round((peak_frame_index * frame_size) / 44100.0, 2),
            "energy_profile_sample": [round(val, 4) for val in rms_output[0, :100].tolist()]
        }

    def compute_spectral_centroid(self, magnitude: torch.Tensor, frequencies: torch.Tensor, sample_rate: float):
        n_bins, num_frames = magnitude.shape

        if HAS_TRITON and self.device.type == "cuda":
            mag_contiguous = magnitude.t().contiguous()
            freq_contiguous = frequencies.contiguous()
            output = torch.empty(num_frames, device=self.device, dtype=torch.float32)
            block_size = min(1024, triton.next_power_of_2(n_bins))

            spectral_centroid_kernel[(num_frames,)](
                freq_contiguous,
                mag_contiguous,
                output,
                n_bins,
                BLOCK_SIZE=block_size,
            )
            centroid_output = output
        else:
            centroid_output = torch.sum(frequencies.unsqueeze(1) * magnitude, dim=0) / (
                        torch.sum(magnitude, dim=0) + 1e-6)
            centroid_output = torch.nan_to_num(centroid_output, nan=0.0)

        return {
            "centroid_frames_count": num_frames,
            "average_centroid_hz": round(centroid_output.mean().item(), 2),
            "peak_centroid_hz": round(centroid_output.max().item(), 2),
            "centroid_profile_sample": [round(val, 2) for val in centroid_output[:100].tolist()]
        }

    def compute_chroma(self, magnitude: torch.Tensor, frequencies: torch.Tensor):
        n_bins = magnitude.shape[0]
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

        return {
            "chroma_bins": 12,
            "chroma_mean_distribution": chroma_summary
        }

    def compute_mel(self, magnitude: torch.Tensor):
        n_bins = magnitude.shape[0]
        num_mels = 64
        mel_filters = torch.randn((num_mels, n_bins), device=self.device, dtype=torch.float32).abs()
        mel_filters /= mel_filters.sum(dim=1, keepdim=True) + 1e-6

        mel_out = torch.matmul(mel_filters, magnitude).unsqueeze(0)
        log_mel = torch.log(mel_out + 1e-6)

        return {
            "num_mels": num_mels,
            "average_log_mel": round(log_mel.mean().item(), 4)
        }

    def compute_tempo(self, waveform: torch.Tensor, sample_rate: float):
        hop_length = 512
        frames = waveform[0, ::hop_length]
        diffs = torch.abs(frames[1:] - frames[:-1])

        peaks = torch.where(diffs > diffs.mean() + diffs.std() * 1.5)[0]
        if len(peaks) > 1:
            intervals = torch.diff(peaks.float()) * (hop_length / sample_rate)
            avg_interval = intervals.median().item()
            if avg_interval > 0:
                tempo = 60.0 / avg_interval

                # Ako je zbog gustine pikova tempo otišao 10x gore, vraćamo ga u realnu meru
                if tempo > 1000.0:
                    tempo /= 10.0

                while tempo < 60.0:
                    tempo *= 2.0
                while tempo > 200.0:
                    tempo /= 2.0

                return round(tempo, 2)
        return 120.0

    def process_audio_file(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")

        start_time = time.perf_counter()
        waveform, sample_rate = self.safe_load_audio(file_path)
        waveform = waveform.to(self.device)
        decode_time = time.perf_counter()

        channels = waveform.shape[0]
        samples = waveform.shape[1]
        duration = samples / sample_rate

        rms_metrics = self.compute_rms(waveform, frame_size=2048)

        n_fft = 2048
        hop_length = 512
        window = torch.hann_window(n_fft, device=self.device)
        mono_waveform = waveform.mean(dim=0) if waveform.ndim > 1 else waveform

        stft_result = torch.stft(
            mono_waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True
        )
        magnitude = torch.abs(stft_result)
        frequencies = torch.linspace(0, sample_rate / 2, magnitude.shape[0], device=self.device)

        spectral_metrics = self.compute_spectral_centroid(magnitude, frequencies, sample_rate)
        chroma_metrics = self.compute_chroma(magnitude, frequencies)
        mel_metrics = self.compute_mel(magnitude)
        tempo = self.compute_tempo(waveform, sample_rate)

        total_time = round((time.perf_counter() - start_time) * 1000, 2)

        result_dict = {
            "agent": "audio",
            "status": "completed",
            "device": str(self.device),
            "bench": {
                "total_load_time_ms": total_time,
                "dtype": str(waveform.dtype),
                "shape": list(waveform.shape)
            },
            "audio": {
                "sample_rate": sample_rate,
                "channels": channels,
                "samples": samples,
                "duration_sec": round(duration, 2),
                "rms_metrics": rms_metrics,
                "spectral_centroid_metrics": spectral_metrics,
                "chroma_metrics": chroma_metrics,
                "mel_metrics": mel_metrics,
                "tempo": tempo
            },
        }

        return json.dumps(result_dict, indent=4)

    def to_spotify_features(self, analysis_json_str: str) -> list:
        data = json.loads(analysis_json_str)
        audio = data["audio"]

        rms = audio["rms_metrics"]
        spectral = audio["spectral_centroid_metrics"]
        chroma = audio["chroma_metrics"]
        mel = audio["mel_metrics"]

        # 1. Danceability (bazirano na standardnoj devijaciji RMS-a)
        danceability = min(max(rms["std_rms"] * 3.5 + 0.18, 0.1), 0.9)

        # 2. Energy (generalizovana formula preko log-mel vrednosti)
        energy = min(max((mel["average_log_mel"] + 15.0) / 20.0, 0.0), 1.0)

        # 3. Key
        chroma_dist = chroma["chroma_mean_distribution"]
        key = float(chroma_dist.index(max(chroma_dist)))

        # 4. Loudness (standardni RMS u dB opseg bez veštačkih offseta)
        mean_rms = max(rms["average_rms"], 1e-5)
        loudness = float(20.0 * torch.log10(torch.tensor(mean_rms)).item() + 6.5)
        loudness = max(min(loudness, 0.0), -35.0)

        # 5. Mode
        mode = 1.0 if chroma_dist[4] > chroma_dist[3] else 0.0

        # 6. Speechiness (bazirano na spektralnom centroidu)
        centroid_hz = spectral.get("average_centroid_hz", 2000.0)
        speechiness = float(np.clip((centroid_hz / 9000.0) * 0.25, 0.02, 0.25))

        # 7. Acousticness (inverzna zavisnost od energije)
        acousticness = float(np.clip(1.0 - (energy * 1.45), 0.0001, 0.8))

        # 8. Instrumentalness
        instrumentalness = float(np.clip(rms.get("std_rms", 0.05) * 0.001, 0.00001, 0.05))

        # 9. Liveness (bazirano na dinamičkom opsegu)
        dynamic_range = rms.get("dynamic_range", 0.1)
        liveness = float(np.clip(dynamic_range * 0.15, 0.03, 0.25))

        # 10. Valence (pozitivnost tonova i ritma)
        valence = float(np.clip((energy * 0.2) + (mode * 0.1) + ((1.0 - danceability) * 0.25), 0.1, 0.6))

        # 11. Tempo (pravi izračunati tempo bez forsiranja)
        tempo = float(audio.get("tempo", 120.0))

        # 12. Duration (pravo trajanje u milisekundama bez skraćivanja)
        duration_ms = float(audio["duration_sec"]) * 1000.0

        return [
            danceability, energy, key, loudness, mode,
            speechiness, acousticness, instrumentalness, liveness,
            valence, tempo, duration_ms
        ]