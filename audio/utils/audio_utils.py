import torch


def rms(waveform, frame_size=2048):
    channels, num_samples = waveform.shape
    num_frames = num_samples // frame_size
    waveform = waveform[:, :num_frames * frame_size]
    frames = waveform.view(channels, num_frames, frame_size)
    rms_values = torch.sqrt(torch.mean(frames ** 2, dim=2))

    return rms_values
