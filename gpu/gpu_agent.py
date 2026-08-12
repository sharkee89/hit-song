import os
import io
import time
import json
import math
import torch
import torchaudio
import torchaudio.functional as F
import triton
import triton.language as tl
from dotenv import load_dotenv
from google import genai
from google.genai import types


# =====================================================================
# FAZA 2: TRITON KERNELI (RMS, Mel, Chroma)
# =====================================================================

@triton.jit
def _fused_rms_energy_kernel(
        audio_ptr,  # Pointer na mono audio [total_samples]
        energy_out_ptr,  # Pointer na izlazni RMS profil [num_frames]
        total_samples,
        frame_length,
        hop_length,
        BLOCK_SIZE: tl.constexpr
):
    """Izračunava RMS energiju po frejmovima u jednom GPU prolazu."""
    frame_idx = tl.program_id(0)
    start_sample = frame_idx * hop_length

    offsets = tl.arange(0, BLOCK_SIZE)
    current_samples = start_sample + offsets

    mask = (offsets < frame_length) & (current_samples < total_samples)
    audio_samples = tl.load(audio_ptr + current_samples, mask=mask, other=0.0)

    # Vectorized FMA (Squared sum unutar SRAM-a)
    squared_samples = audio_samples * audio_samples
    sum_squares = tl.sum(tl.where(mask, squared_samples, 0.0), axis=0)

    rms = tl.sqrt(sum_squares / frame_length)
    tl.store(energy_out_ptr + frame_idx, rms)


@triton.jit
def _fused_mel_projection_kernel(
        spec_ptr,  # Magnitude spec [num_bins, num_frames]
        mel_filters_ptr,  # Filterbank matrix [num_mels, num_bins]
        mel_out_ptr,  # Izlazni Log-Mel spec [num_mels, num_frames]
        num_bins,
        num_mels,
        num_frames,
        stride_bin,
        stride_frame,
        BLOCK_SIZE_BIN: tl.constexpr
):
    """Projektuje STFT binove u Mel skalu sa log-scalingom u SRAM-u."""
    frame_idx = tl.program_id(0)
    mel_idx = tl.program_id(1)

    bin_offsets = tl.arange(0, BLOCK_SIZE_BIN)
    bin_mask = bin_offsets < num_bins

    # Dvo-dimenzionalno adresiranje uz poštovanje stride-ova
    spec_offsets = bin_offsets * stride_bin + frame_idx * stride_frame
    spec_col = tl.load(spec_ptr + spec_offsets, mask=bin_mask, other=0.0)

    filter_offsets = mel_idx * num_bins + bin_offsets
    mel_row = tl.load(mel_filters_ptr + filter_offsets, mask=bin_mask, other=0.0)

    # Fused Dot-Product & Log Compression
    mel_val = tl.sum(tl.where(bin_mask, spec_col * mel_row, 0.0), axis=0)
    log_mel = tl.log(mel_val + 1e-6)

    out_offset = mel_idx * num_frames + frame_idx
    tl.store(mel_out_ptr + out_offset, log_mel)


@triton.jit
def _fused_chroma_projection_kernel(
        spec_ptr,  # Magnitude spec [num_bins, num_frames]
        chroma_map_ptr,  # Matrix [12, num_bins] koja binove preslikava u 12 tonova
        chroma_out_ptr,  # Izlazni Chroma profil [12, num_frames]
        num_bins,
        num_frames,
        stride_bin,
        stride_frame,
        BLOCK_SIZE_BIN: tl.constexpr
):
    """Izdvaja 12-tonski Chroma profil (C, C#, D... B) iz spektrograma."""
    frame_idx = tl.program_id(0)
    chroma_idx = tl.program_id(1)

    bin_offsets = tl.arange(0, BLOCK_SIZE_BIN)
    bin_mask = bin_offsets < num_bins

    spec_offsets = bin_offsets * stride_bin + frame_idx * stride_frame
    spec_col = tl.load(spec_ptr + spec_offsets, mask=bin_mask, other=0.0)

    map_offsets = chroma_idx * num_bins + bin_offsets
    chroma_row = tl.load(chroma_map_ptr + map_offsets, mask=bin_mask, other=0.0)

    chroma_val = tl.sum(tl.where(bin_mask, spec_col * chroma_row, 0.0), axis=0)

    out_offset = chroma_idx * num_frames + frame_idx
    tl.store(chroma_out_ptr + out_offset, chroma_val)


# =====================================================================
# FAZA 2: PYTHON TRITON ENGINE WRAPPER
# =====================================================================

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class TritonMIREngine:
    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 80):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.num_bins = n_fft // 2 + 1

        # Pre-computovani Mel i Chroma filteri na GPU
        self.mel_filters = F.melscale_fbanks(
            n_freqs=self.num_bins,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm='slaney'
        ).to('cuda').T.contiguous()  # Shape: [n_mels, num_bins]

        self.chroma_map = self._build_chroma_filterbank().to('cuda').contiguous()  # Shape: [12, num_bins]

    def _build_chroma_filterbank(self) -> torch.Tensor:
        """Gradi matricu preslikavanja FFT binova u 12 polu-tonskih klasa."""
        chroma_map = torch.zeros((12, self.num_bins), dtype=torch.float32)
        fft_freqs = torch.linspace(0, self.sample_rate / 2.0, self.num_bins)

        for i, freq in enumerate(fft_freqs):
            if freq > 0:
                midi_pitch = 69 + 12 * math.log2(freq / 440.0)
                pitch_class = int(round(midi_pitch)) % 12
                chroma_map[pitch_class, i] = 1.0

        chroma_map = chroma_map / (chroma_map.sum(dim=1, keepdim=True) + 1e-6)
        return chroma_map

    def compute_rms_energy(self, waveform_gpu: torch.Tensor) -> torch.Tensor:
        """Računa energetski profil po frejmovima."""
        total_samples = waveform_gpu.numel()
        num_frames = (total_samples - self.n_fft) // self.hop_length + 1
        if num_frames <= 0:
            return torch.zeros(1, device='cuda')

        energy_output = torch.empty(num_frames, device='cuda', dtype=torch.float32)
        grid = (num_frames,)
        BLOCK_SIZE = triton.next_power_of_2(self.n_fft)

        _fused_rms_energy_kernel[grid](
            waveform_gpu,
            energy_output,
            total_samples,
            self.n_fft,
            self.hop_length,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return energy_output

    def find_best_climax_segment(self, energy_profile: torch.Tensor, target_duration_sec: float = 30.0) -> tuple[
        float, float]:
        """Pronalazi vremenski prozor sa najvećom prosečnom RMS energijom (refren)."""
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
        """Glavni ulaz: Eksportuje celokupan MIR profil i Climax interval."""
        if waveform_gpu.dim() == 2:
            waveform_mono = waveform_gpu.mean(dim=0)
        else:
            waveform_mono = waveform_gpu

        waveform_mono = waveform_mono.contiguous()

        # 1. Triton RMS Profil i Refren
        energy_profile = self.compute_rms_energy(waveform_mono)
        start_sec, end_sec = self.find_best_climax_segment(energy_profile, target_duration_sec=30.0)
        mean_energy = energy_profile.mean().item()

        # 2. PyTorch STFT u VRAM-u
        window = torch.hann_window(self.n_fft, device='cuda')
        stft_res = torch.stft(
            waveform_mono,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )
        magnitude_spec = torch.abs(stft_res).contiguous()
        num_bins, num_frames = magnitude_spec.shape

        # 3. Triton Log-Mel Spectrogram
        mel_output = torch.empty((self.n_mels, num_frames), device='cuda', dtype=torch.float32)
        grid_mel = (num_frames, self.n_mels)
        BLOCK_SIZE_BIN = triton.next_power_of_2(num_bins)

        _fused_mel_projection_kernel[grid_mel](
            magnitude_spec,
            self.mel_filters,
            mel_output,
            num_bins,
            self.n_mels,
            num_frames,
            magnitude_spec.stride(0),
            magnitude_spec.stride(1),
            BLOCK_SIZE_BIN=BLOCK_SIZE_BIN
        )

        # 4. Triton Chroma & Key Detection
        chroma_output = torch.empty((12, num_frames), device='cuda', dtype=torch.float32)
        grid_chroma = (num_frames, 12)

        _fused_chroma_projection_kernel[grid_chroma](
            magnitude_spec,
            self.chroma_map,
            chroma_output,
            num_bins,
            num_frames,
            magnitude_spec.stride(0),
            magnitude_spec.stride(1),
            BLOCK_SIZE_BIN=BLOCK_SIZE_BIN
        )

        chroma_sums = chroma_output.sum(dim=1)
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
                "chroma_shape": list(chroma_output.shape)
            }
        }


# =====================================================================
# MAIN AGENT KLASA (Faza 1, 3 i Integracija)
# =====================================================================

class GPUAudioAgent:
    def __init__(self, api_key: str, mir_weight: float = 0.5, name: str = "GPU:AudioAnalyst", device: str = None):
        self.name = name
        self.role = "GPU-accelerated structural analysis and multimodal audio interpretation"
        self.client = genai.Client(api_key=api_key)
        self.mir_weight = max(0.0, min(1.0, mir_weight))

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda":
            self.mir_engine = TritonMIREngine()
        else:
            self.mir_engine = None

        print(f"[{self.name}]: Inicijalizovan na uređaju -> {self.device.upper()}")

    def _load_audio_to_vram(self, file_path: str):
        """FAZA 1: Učitava MP3 direktno u RAM, pa šalje u GPU VRAM kao float32 Tensor."""
        start_time = time.perf_counter()
        waveform, sample_rate = torchaudio.load(file_path)
        waveform_gpu = waveform.to(self.device)

        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"[{self.name}]: MP3 dekodiran i prebačen u VRAM za {elapsed:.2f} ms")

        return waveform_gpu, sample_rate

    def _trim_segment_in_vram(self, waveform_gpu: torch.Tensor, sample_rate: int, start_sec: float,
                              end_sec: float) -> bytes:
        """FAZA 1 & 3: Sečenje u VRAM-u i pakovanje u In-Memory WAV bafer."""
        start_time = time.perf_counter()

        start_frame = int(start_sec * sample_rate)
        end_frame = int(end_sec * sample_rate)

        trimmed_gpu = waveform_gpu[:, start_frame:end_frame]
        trimmed_cpu = trimmed_gpu.cpu()

        buffer = io.BytesIO()
        torchaudio.save(buffer, trimmed_cpu, sample_rate, format="wav")
        buffer.seek(0)

        # Čišćenje VRAM referenci
        del trimmed_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"[{self.name}]: VRAM slicing i RAM baferovanje završeni za {elapsed:.2f} ms (0 bajtova na disk)")

        return buffer.read()

    def _get_deep_audio_insight_from_buffer(self, audio_bytes: bytes, raw_metrics: dict) -> dict:
        """FAZA 3: Šalje In-Memory audio bafer direktno na Gemini API."""
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

        max_attempts = 3
        initial_delay = 12.0

        for attempt in range(max_attempts):
            try:
                audio_file = self.client.files.upload(
                    file=io.BytesIO(audio_bytes),
                    config={
                        'mime_type': 'audio/wav',
                        'display_name': 'in_memory_segment.wav'
                    }
                )

                while audio_file.state.name == "PROCESSING":
                    time.sleep(1)
                    audio_file = self.client.files.get(name=audio_file.name)

                if audio_file.state.name == "FAILED":
                    raise Exception("Google AI nije uspeo da procesira in-memory audio bafer.")

                prompt = """
                Analyze this song segment. Evaluate the overall production quality and the vocal mix/presence.
                Provide your final evaluation through the requested schema. 
                The 'potential_score' must reflect how well this tracks aligns with modern commercial radio standards.
                """

                # ISPRAVLJENO: gemini-2.5-flash
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
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_attempts - 1:
                        print(f"⚠️ [{self.name}]: 429 Rate Limit. Pauza {initial_delay}s...")
                        time.sleep(initial_delay)
                        continue

                print(f"❌ [{self.name}]: Greška pri Gemini analizi: {str(e)}")
                return self._generate_heuristic_fallback(raw_metrics)

        return self._generate_heuristic_fallback(raw_metrics)

    def _generate_heuristic_fallback(self, raw_metrics: dict) -> dict:
        energy = raw_metrics.get('energy_%', 50) / 100
        return {
            "production_quality_description": "GPU Fallback mode. Analysis derived from local VRAM metrics.",
            "vocal_presence_rating": "Undetected (Local Fallback)",
            "potential_score": round(energy, 2)
        }

    def process_track(self, file_path: str) -> dict:
        total_start = time.perf_counter()
        print(f"\n[{self.name}]: Započeta GPU obrada: {file_path}")

        # 1. Učitavanje u VRAM
        waveform_gpu, sample_rate = self._load_audio_to_vram(file_path)

        # 2. FAZA 2: Triton Custom MIR Engine obrada
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

        # 3. Sečenje najjačeg refrena u VRAM-u
        audio_bytes = self._trim_segment_in_vram(
            waveform_gpu, sample_rate, best_start, best_end
        )

        # 4. Slanje na Gemini API
        ai_insights = self._get_deep_audio_insight_from_buffer(audio_bytes, mir_metrics)

        # 5. Konačni izračun
        mir_signal = mir_metrics['energy_%'] / 100.0
        ai_signal = ai_insights.get('potential_score', 0.5)
        activation_value = (mir_signal * self.mir_weight) + (ai_signal * (1.0 - self.mir_weight))

        total_elapsed = (time.perf_counter() - total_start) * 1000

        return {
            "agent_name": self.name,
            "device": self.device,
            "total_execution_time_ms": round(total_elapsed, 2),
            "activation_value": round(activation_value, 4),
            "findings": {
                "deterministic_mir_metrics": mir_metrics,
                "neural_semantic_analysis": ai_insights
            }
        }


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    load_dotenv()

    # Bezbedno očitavanje ključa iz .env
    GEMINI_API_KEY = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("❌ Nije pronađen API ključ u .env fajlu!")
    else:
        agent = GPUAudioAgent(api_key=GEMINI_API_KEY)
        test_file = r"C:\Users\Sharkee\Downloads\Aylex_-_Live_It_(freetouse.com).mp3"

        if os.path.exists(test_file):
            result = agent.process_track(test_file)
            print("\n--- KONAČNI REZULTAT ---")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Test fajl ne postoji na putanji: {test_file}")