import io
import json
import os
import sys

import librosa
import numpy as np
import torch
from dotenv import load_dotenv


# Ensure root path resolution
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class ViralAgent:

    def __init__(self, device: str = None):
        # ---------------------------------------------------------
        # 1. Device setup
        # ---------------------------------------------------------
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.sample_rate = 22050
        self.analysis_duration = 30.0
        self.target_samples = int(
            self.sample_rate * self.analysis_duration
        )

        self.hop_length = 512
        self.n_fft = 2048
        self.n_mels = 128

        self.feature_names = [
            "tempo_normalized",
            "beat_strength",
            "onset_density",
            "beat_regularity",
            "energy_mean",
            "energy_variance",
            "energy_contrast",
            "spectral_brightness",
            "spectral_contrast",
            "spectral_rolloff",
            "zero_crossing_rate",
            "harmonic_ratio",
            "percussive_ratio",
            "loopability_score",
            "hook_strength",
            "hook_early_presence",
        ]

        print(f"[ViralAgent] Execution Device: {self.device}")

    # -------------------------------------------------------------
    # 2. Audio loading
    # -------------------------------------------------------------
    def _load_audio(self, source: str | io.BytesIO) -> np.ndarray:
        """
        Loads an audio source as mono audio with a fixed 30-second
        duration, matching the general AudioAgent input format.
        """

        try:
            y, _ = librosa.load(
                source,
                sr=self.sample_rate,
                mono=True,
            )
        except Exception as err:
            raise ValueError(
                f"Failed to decode audio source: {err}"
            ) from err

        if len(y) == 0:
            raise ValueError("Loaded audio signal is empty.")

        if len(y) > self.target_samples:
            y = y[:self.target_samples]
        else:
            y = np.pad(
                y,
                (0, self.target_samples - len(y)),
                mode="constant",
            )

        # Normalize amplitude
        peak = np.max(np.abs(y))

        if peak > 0:
            y = y / peak

        return y.astype(np.float32)

    # -------------------------------------------------------------
    # 3. Safe numerical helpers
    # -------------------------------------------------------------
    @staticmethod
    def _safe_mean(values: np.ndarray) -> float:
        value = float(np.mean(values))

        if not np.isfinite(value):
            return 0.0

        return value

    @staticmethod
    def _safe_std(values: np.ndarray) -> float:
        value = float(np.std(values))

        if not np.isfinite(value):
            return 0.0

        return value

    @staticmethod
    def _normalize(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        if maximum <= minimum:
            return 0.0

        normalized = (value - minimum) / (
            maximum - minimum
        )

        return float(np.clip(normalized, 0.0, 1.0))

    # -------------------------------------------------------------
    # 4. Audio feature extraction
    # -------------------------------------------------------------
    def _extract_features(
        self,
        y: np.ndarray,
    ) -> dict[str, float]:

        sr = self.sample_rate
        hop_length = self.hop_length

        # ---------------------------------------------------------
        # STFT and magnitude
        # ---------------------------------------------------------
        stft = librosa.stft(
            y,
            n_fft=self.n_fft,
            hop_length=hop_length,
        )

        magnitude = np.abs(stft)

        # ---------------------------------------------------------
        # Tempo and onset features
        # ---------------------------------------------------------
        tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
            hop_length=hop_length,
        )

        tempo = float(
            np.asarray(tempo).reshape(-1)[0]
        )

        onset_strength = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=hop_length,
        )

        beat_strength = self._safe_mean(onset_strength)
        onset_density = self._safe_std(onset_strength)

        # ---------------------------------------------------------
        # Beat regularity
        # ---------------------------------------------------------
        if len(beat_frames) > 1:
            beat_times = librosa.frames_to_time(
                beat_frames,
                sr=sr,
                hop_length=hop_length,
            )

            beat_intervals = np.diff(beat_times)

            if len(beat_intervals) > 0:
                interval_std = self._safe_std(
                    beat_intervals
                )

                beat_regularity = 1.0 / (
                    1.0 + interval_std
                )
            else:
                beat_regularity = 0.0
        else:
            beat_regularity = 0.0

        # ---------------------------------------------------------
        # Energy and dynamics
        # ---------------------------------------------------------
        rms = librosa.feature.rms(
            S=magnitude,
            hop_length=hop_length,
        )[0]

        energy_mean = self._safe_mean(rms)
        energy_variance = self._safe_std(rms)

        if len(rms) > 1:
            first_third_end = max(1, len(rms) // 3)

            first_part = rms[:first_third_end]
            last_part = rms[-first_third_end:]

            energy_contrast = abs(
                self._safe_mean(last_part)
                - self._safe_mean(first_part)
            )
        else:
            energy_contrast = 0.0

        # ---------------------------------------------------------
        # Spectral features
        # ---------------------------------------------------------
        spectral_centroid = librosa.feature.spectral_centroid(
            S=magnitude,
            sr=sr,
            hop_length=hop_length,
        )[0]

        spectral_rolloff = librosa.feature.spectral_rolloff(
            S=magnitude,
            sr=sr,
            hop_length=hop_length,
            roll_percent=0.85,
        )[0]

        spectral_contrast = librosa.feature.spectral_contrast(
            S=magnitude,
            sr=sr,
            hop_length=hop_length,
        )

        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y,
            hop_length=hop_length,
        )[0]

        spectral_brightness = self._safe_mean(
            spectral_centroid
        )

        spectral_contrast_value = self._safe_mean(
            spectral_contrast
        )

        spectral_rolloff_value = self._safe_mean(
            spectral_rolloff
        )

        zcr_value = self._safe_mean(
            zero_crossing_rate
        )

        # ---------------------------------------------------------
        # Harmonic/percussive separation
        # ---------------------------------------------------------
        harmonic, percussive = librosa.effects.hpss(y)

        harmonic_energy = float(
            np.mean(harmonic ** 2)
        )

        percussive_energy = float(
            np.mean(percussive ** 2)
        )

        total_energy = (
            harmonic_energy + percussive_energy
        )

        if total_energy > 0:
            harmonic_ratio = (
                harmonic_energy / total_energy
            )

            percussive_ratio = (
                percussive_energy / total_energy
            )
        else:
            harmonic_ratio = 0.0
            percussive_ratio = 0.0

        # ---------------------------------------------------------
        # Loopability proxy
        # ---------------------------------------------------------
        segment_length = int(5.0 * sr)

        if len(y) >= 2 * segment_length:
            first_segment = y[:segment_length]
            last_segment = y[-segment_length:]

            correlation = np.corrcoef(
                first_segment,
                last_segment,
            )[0, 1]

            if np.isfinite(correlation):
                segment_similarity = float(
                    np.clip(
                        (correlation + 1.0) / 2.0,
                        0.0,
                        1.0,
                    )
                )
            else:
                segment_similarity = 0.0
        else:
            segment_similarity = 0.0

        # ---------------------------------------------------------
        # Hook-related proxy features
        # ---------------------------------------------------------
        # This is an approximation based on onset strength and RMS.
        min_length = min(
            len(onset_strength),
            len(rms),
        )

        if min_length > 0:
            normalized_rms = rms[:min_length] / (
                np.max(rms[:min_length]) + 1e-8
            )

            combined_hook_signal = (
                onset_strength[:min_length]
                * normalized_rms
            )

            hook_index = int(
                np.argmax(combined_hook_signal)
            )

            max_hook_value = np.max(
                combined_hook_signal
            )

            if max_hook_value > 0:
                hook_strength = float(
                    combined_hook_signal[hook_index]
                    / (max_hook_value + 1e-8)
                )
            else:
                hook_strength = 0.0

            hook_time = float(
                librosa.frames_to_time(
                    hook_index,
                    sr=sr,
                    hop_length=hop_length,
                )
            )

            hook_early_presence = float(
                np.exp(-hook_time / 10.0)
            )
        else:
            hook_strength = 0.0
            hook_early_presence = 0.0

        # ---------------------------------------------------------
        # Return normalized feature dictionary
        # ---------------------------------------------------------
        return {
            "tempo_normalized": self._normalize(
                tempo,
                60.0,
                180.0,
            ),
            "beat_strength": self._normalize(
                beat_strength,
                0.0,
                1.0,
            ),
            "onset_density": self._normalize(
                onset_density,
                0.0,
                1.0,
            ),
            "beat_regularity": float(
                np.clip(
                    beat_regularity,
                    0.0,
                    1.0,
                )
            ),
            "energy_mean": self._normalize(
                energy_mean,
                0.0,
                0.5,
            ),
            "energy_variance": self._normalize(
                energy_variance,
                0.0,
                0.25,
            ),
            "energy_contrast": self._normalize(
                energy_contrast,
                0.0,
                0.5,
            ),
            "spectral_brightness": self._normalize(
                spectral_brightness,
                500.0,
                8000.0,
            ),
            "spectral_contrast": self._normalize(
                spectral_contrast_value,
                0.0,
                50.0,
            ),
            "spectral_rolloff": self._normalize(
                spectral_rolloff_value,
                1000.0,
                10000.0,
            ),
            "zero_crossing_rate": self._normalize(
                zcr_value,
                0.0,
                0.5,
            ),
            "harmonic_ratio": float(
                np.clip(
                    harmonic_ratio,
                    0.0,
                    1.0,
                )
            ),
            "percussive_ratio": float(
                np.clip(
                    percussive_ratio,
                    0.0,
                    1.0,
                )
            ),
            "loopability_score": segment_similarity,
            "hook_strength": hook_strength,
            "hook_early_presence": hook_early_presence,
        }

    # -------------------------------------------------------------
    # 5. Heuristic short-form audio suitability score
    # -------------------------------------------------------------
    def _calculate_viral_score(
        self,
        features: dict[str, float],
    ) -> float:
        """
        Calculates a heuristic score for short-form video audio suitability.

        The score is based only on audio characteristics such as hook strength,
        rhythmic clarity, energy, loopability and early engagement potential.
        It does not use historical popularity or social-media usage data.

        This is not a supervised prediction of actual future social-media virality.
        It is an interpretable estimate of how suitable the audio may be for
        TikTok, Instagram Reels, YouTube Shorts and similar formats.
        """

        weights = {
            "tempo_normalized": 0.05,
            "beat_strength": 0.10,
            "onset_density": 0.08,
            "beat_regularity": 0.10,
            "energy_mean": 0.07,
            "energy_variance": 0.05,
            "energy_contrast": 0.09,
            "spectral_brightness": 0.04,
            "spectral_contrast": 0.03,
            "spectral_rolloff": 0.03,
            "zero_crossing_rate": 0.02,
            "harmonic_ratio": 0.04,
            "percussive_ratio": 0.08,
            "loopability_score": 0.10,
            "hook_strength": 0.08,
            "hook_early_presence": 0.04,
        }

        score = sum(
            weights[name] * features.get(name, 0.0)
            for name in self.feature_names
        )

        return float(
            np.clip(score, 0.0, 1.0)
        )

    # -------------------------------------------------------------
    # 6. Main Viral Agent analysis method
    # -------------------------------------------------------------
    def analyze(self, source: str | io.BytesIO) -> dict:
        """
        Executes the complete Viral Agent analysis pipeline.
        """

        y = self._load_audio(source)

        features = self._extract_features(y)

        short_form_score = self._calculate_viral_score(
            features
        )

        feature_vector = np.array(
            [
                features[name]
                for name in self.feature_names
            ],
            dtype=np.float32,
        )

        # This tensor is concatenated with other agent outputs before the MLP.
        viral_tensor = torch.tensor(
            feature_vector,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        return {
            "agent": "ViralAgent",
            "analysis_type": "heuristic_short_form_audio_suitability",
            "short_form_suitability_score": short_form_score,
            "viral_score": short_form_score,
            "feature_names": self.feature_names,
            "mlp_input_description": "Normalized heuristic audio features for short-form video suitability.",
            "feature_values": features,
            "viral_tensor": viral_tensor.detach()
            .cpu()
            .numpy()
            .tolist(),
            "tensor_shape": list(
                viral_tensor.shape
            ),
        }

    # -------------------------------------------------------------
    # 7. JSON serialization
    # -------------------------------------------------------------
    def get_data(
        self,
        viral_analysis: dict,
    ) -> str:
        """
        Converts Viral Agent results into JSON.
        """

        serializable_data = dict(viral_analysis)

        return json.dumps(
            serializable_data,
            indent=4,
        )

    # -------------------------------------------------------------
    # 8. Full process pipeline
    # -------------------------------------------------------------
    def process(
        self,
        file_path: str,
    ) -> str:
        """
        Full end-to-end ViralAgent pipeline.
        """

        viral_analysis = self.analyze(file_path)

        return self.get_data(viral_analysis)


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()

    file_path = os.getenv(
        "AUDIO_FILE_PATH",
        "",
    )

    agent = ViralAgent()

    if file_path and os.path.exists(file_path):
        print(
            "🚀 Starting heuristic short-form ViralAgent pipeline processing..."
        )

        result_json = agent.process(file_path)

        data = json.loads(result_json)

        print(
            "✅ ViralAgent pipeline executed successfully!"
        )

        print(json.dumps(data, indent=4))

    else:
        print(
            "❌ AUDIO_FILE_PATH isn't defined or "
            "file doesn't exist in .env."
        )
