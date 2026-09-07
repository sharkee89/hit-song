import torch
import torchaudio.functional as F
import triton

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_chroma_filterbank(sample_rate: int, num_bins: int) -> torch.Tensor:
    """Gradi matricu preslikavanja FFT binova u 12 polu-tonskih klasa sa koritom za Triton [16, num_bins]."""
    fft_freqs = torch.linspace(0, sample_rate / 2.0, num_bins)
    midi_pitch = torch.zeros_like(fft_freqs)
    nonzero = fft_freqs > 0
    midi_pitch[nonzero] = 69.0 + 12.0 * torch.log2(fft_freqs[nonzero] / 440.0)
    pitch_classes = (torch.round(midi_pitch) % 12).long()

    chroma_map = torch.zeros((16, num_bins), dtype=torch.float32)
    rows = torch.arange(12).unsqueeze(1)
    active = (rows == pitch_classes.unsqueeze(0)) & nonzero.unsqueeze(0)
    tmp = torch.zeros((12, num_bins), dtype=torch.float32)
    tmp[active] = 1.0
    tmp /= (tmp.sum(dim=1, keepdim=True) + 1e-6)
    chroma_map[:12] = tmp
    return chroma_map.cuda()


def get_mel_filters(sample_rate: int, num_bins: int, n_mels: int) -> torch.Tensor:
    """Kreira Mel scale filterbank na GPU."""
    return F.melscale_fbanks(
        n_freqs=num_bins,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm='slaney'
    ).to('cuda').T.contiguous()