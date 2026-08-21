import os
import io
import time
import json
import torch
import torchaudio
import triton
from dotenv import load_dotenv
from google import genai
from google.genai import types
import sys
from pathlib import Path

# parent je 'agents', parent.parent je 'gpu', parent.parent.parent je 'machine_learning'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpu.kernel.triton_kernel_utils import (
    build_chroma_filterbank,
    get_mel_filters,
    PITCH_NAMES,
)

from gpu.kernel.triton_chroma_kernel import (
    triton_chroma_kernel
)

from gpu.kernel.fused_rms_energy_kernel import (
    fused_rms_energy_kernel
)

from gpu.kernel.triton_mel_kernel import (
    triton_mel_kernel
)

# =====================================================================
# PYTHON TRITON ENGINE WRAPPER
# =====================================================================

class TritonMIREngine:
    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 80):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.num_bins = n_fft // 2 + 1

        self.mel_filters = get_mel_filters(sample_rate, self.num_bins, n_mels)
        self.chroma_map = build_chroma_filterbank(sample_rate, self.num_bins)

    def compute_rms_energy(self, waveform_gpu: torch.Tensor) -> torch.Tensor:
        total_samples = waveform_gpu.numel()
        num_frames = (total_samples - self.n_fft) // self.hop_length + 1
        if num_frames <= 0:
            return torch.zeros(1, device='cuda')

        energy_output = torch.empty(num_frames, device='cuda', dtype=torch.float32)
        grid = (num_frames,)
        BLOCK_SIZE = triton.next_power_of_2(self.n_fft)

        fused_rms_energy_kernel[grid](
            waveform_gpu,
            energy_output,
            total_samples,
            self.n_fft,
            self.hop_length,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return energy_output

    def find_best_climax_segment(self, energy_profile: torch.Tensor, target_duration_sec: float = 30.0) -> tuple[float, float]:
        frames_per_sec = self.sample_rate / self.hop_length
        window_size_frames = int(target_duration_sec * frames_per_sec)

        if energy_profile.numel() <= window_size_frames:
            return 0.0, energy_profile.numel() / frames_per_sec

        cumsum = torch.cumsum(energy_profile, dim=0)
        window_sums = cumsum[window_size_frames:] - cumsum[:-window_size_frames]

        max_idx = torch.argmax(window_sums).item()
        start_sec = max_idx / frames_per_sec
        end_sec = start_sec + target_duration_sec

        return round(start_sec, 2), round(end_sec, 2)

    def extract_all_features(self, waveform_gpu: torch.Tensor) -> dict:
        waveform_mono = waveform_gpu.mean(dim=0) if waveform_gpu.dim() == 2 else waveform_gpu
        waveform_mono = waveform_mono.contiguous()

        # 1. Triton RMS Profil
        energy_profile = self.compute_rms_energy(waveform_mono)
        start_sec, end_sec = self.find_best_climax_segment(energy_profile, target_duration_sec=30.0)
        mean_energy = energy_profile.mean().item()

        # 2. PyTorch STFT u VRAM-u
        window = torch.hann_window(self.n_fft, device='cuda')
        stft_res = torch.stft(waveform_mono, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        magnitude_spec = torch.abs(stft_res).contiguous()
        num_bins, num_frames = magnitude_spec.shape

        # 3. Triton Log-Mel Spectrogram (autotuned)
        mel_output = torch.empty((self.n_mels, num_frames), device='cuda', dtype=torch.float32)
        grid_mel = lambda META: (triton.cdiv(self.n_mels, META["BLOCK_SIZE_M"]), triton.cdiv(num_frames, META["BLOCK_SIZE_N"]))

        triton_mel_kernel[grid_mel](
            magnitude_spec, self.mel_filters, mel_output,
            num_bins, self.n_mels, num_frames,
            magnitude_spec.stride(0), magnitude_spec.stride(1)
        )

        # 4. Triton Chroma (autotuned)
        chroma_output = torch.empty((16, num_frames), device='cuda', dtype=torch.float32)
        grid_chroma = lambda META: (triton.cdiv(num_frames, META["BLOCK_SIZE_N"]),)

        triton_chroma_kernel[grid_chroma](
            magnitude_spec, self.chroma_map, chroma_output,
            num_bins, num_frames,
            magnitude_spec.stride(0), magnitude_spec.stride(1)
        )

        chroma_sums = chroma_output[:12].sum(dim=1)
        detected_key_idx = torch.argmax(chroma_sums).item()
        detected_key_name = PITCH_NAMES[detected_key_idx]

        return {
            "energy_%": round(min(100.0, mean_energy * 1000.0), 2),
            "detected_key_name": detected_key_name,
            "climax_segment": {
                "start_sec": start_sec,
                "end_sec": end_sec
            },
            "spectral_summary": {
                "num_frames": num_frames,
                "mel_shape": list(mel_output.shape),
                "chroma_shape": list(chroma_output[:12].shape)
            }
        }

# =====================================================================
# MAIN AGENT KLASA
# =====================================================================

class GPUAudioAgent:
    def __init__(self, api_key: str, mir_weight: float = 0.5, name: str = "GPU:AudioAnalyst", device: str = None):
        self.name = name
        self.role = "GPU-accelerated structural analysis and multimodal audio interpretation"
        self.client = genai.Client(api_key=api_key)
        self.mir_weight = max(0.0, min(1.0, mir_weight))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mir_engine = TritonMIREngine() if self.device == "cuda" else None

        print(f"[{self.name}]: Inicijalizovan na uređaju -> {self.device.upper()}")

    def _load_audio_to_vram(self, file_path: str):
        start_time = time.perf_counter()
        waveform, sample_rate = torchaudio.load(file_path)
        waveform_gpu = waveform.to(self.device)
        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"[{self.name}]: MP3 dekodiran i prebačen u VRAM za {elapsed:.2f} ms")
        return waveform_gpu, sample_rate

    def _trim_segment_in_vram(self, waveform_gpu: torch.Tensor, sample_rate: int, start_sec: float, end_sec: float) -> bytes:
        start_time = time.perf_counter()
        start_frame = int(start_sec * sample_rate)
        end_frame = int(end_sec * sample_rate)

        trimmed_gpu = waveform_gpu[:, start_frame:end_frame]
        buffer = io.BytesIO()
        torchaudio.save(buffer, trimmed_gpu.cpu(), sample_rate, format="wav")
        buffer.seek(0)

        del trimmed_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"[{self.name}]: VRAM slicing i RAM baferovanje završeni za {elapsed:.2f} ms (0 bajtova na disk)")
        return buffer.read()

    def _get_deep_audio_insight_from_buffer(self, audio_bytes: bytes, raw_metrics: dict) -> dict:
        print(f"[{self.name}]: Šaljem In-Memory audio bafer na Gemini AI...")
        json_schema = {
            "type": "OBJECT",
            "properties": {
                "production_quality_description": {"type": "STRING"},
                "vocal_presence_rating": {"type": "STRING", "description": "Opis vokala i miksa"},
                "potential_score": {"type": "NUMBER", "description": "Ocena od 0.0 do 1.0 komercijalnog potencijala"}
            },
            "required": ["production_quality_description", "vocal_presence_rating", "potential_score"]
        }

        try:
            audio_file = self.client.files.upload(
                file=io.BytesIO(audio_bytes),
                config={'mime_type': 'audio/wav', 'display_name': 'in_memory_segment.wav'}
            )

            while audio_file.state.name == "PROCESSING":
                time.sleep(1)
                audio_file = self.client.files.get(name=audio_file.name)

            if audio_file.state.name == "FAILED":
                raise Exception("Google AI obrada nije uspela.")

            prompt = "Analyze this song segment. Evaluate quality, vocal mix, and potential_score for radio."
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[audio_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=json_schema,
                    temperature=0.2
                )
            )

            try:
                self.client.files.delete(name=audio_file.name)
            except Exception:
                pass

            return json.loads(response.text)

        except Exception as e:
            print(f"❌ [{self.name}]: Greška pri Gemini analizi: {str(e)}")
            return {"production_quality_description": "Fallback", "vocal_presence_rating": "N/A", "potential_score": round(raw_metrics.get('energy_%', 50)/100, 2)}

    def process_track(self, file_path: str) -> dict:
        total_start = time.perf_counter()
        print(f"\n[{self.name}]: Započeta GPU obrada: {file_path}")

        waveform_gpu, sample_rate = self._load_audio_to_vram(file_path)

        if self.device == "cuda" and self.mir_engine:
            mir_start = time.perf_counter()
            mir_metrics = self.mir_engine.extract_all_features(waveform_gpu)
            mir_elapsed = (time.perf_counter() - mir_start) * 1000
            print(f"[{self.name}]: Triton MIR Engine izvršen za {mir_elapsed:.2f} ms")

            best_start = mir_metrics["climax_segment"]["start_sec"]
            best_end = mir_metrics["climax_segment"]["end_sec"]
        else:
            mir_metrics = {'energy_%': 50.0, 'detected_key_name': 'C'}
            best_start, best_end = 30.0, 60.0

        audio_bytes = self._trim_segment_in_vram(waveform_gpu, sample_rate, best_start, best_end)
        ai_insights = self._get_deep_audio_insight_from_buffer(audio_bytes, mir_metrics)

        mir_signal = mir_metrics['energy_%'] / 100.0
        ai_signal = ai_insights.get('potential_score', 0.5)
        activation_value = (mir_signal * self.mir_weight) + (ai_signal * (1.0 - self.mir_weight))

        return {
            "agent_name": self.name,
            "device": self.device,
            "total_execution_time_ms": round((time.perf_counter() - total_start) * 1000, 2),
            "activation_value": round(activation_value, 4),
            "findings": {
                "deterministic_mir_metrics": mir_metrics,
                "neural_semantic_analysis": ai_insights
            }
        }

if __name__ == "__main__":
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        agent = GPUAudioAgent(api_key=GEMINI_API_KEY)
        test_file = r"C:\Users\Sharkee\Downloads\Aylex_-_Live_It_(freetouse.com).mp3"
        if os.path.exists(test_file):
            print(json.dumps(agent.process_track(test_file), indent=2))