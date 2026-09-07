import torch


def extract_rms_feature(waveform: torch.Tensor, frame_size: int = 2048) -> dict:
    channels, num_samples = waveform.shape
    num_frames = num_samples // frame_size

    if num_frames == 0:
        return {"rms_summary": None}

    trimmed_waveform = waveform[:, : num_frames * frame_size]
    frames = trimmed_waveform.view(channels, num_frames, frame_size)

    rms_values = torch.sqrt(torch.mean(frames ** 2, dim=2))

    channel_metrics = []
    for c in range(channels):
        ch_rms = rms_values[c]

        mean_val = float(ch_rms.mean())
        max_val = float(ch_rms.max())
        min_val = float(ch_rms.min())
        std_val = float(ch_rms.std())
        channel_metrics.append({
            "channel": c,
            "mean": mean_val,
            "max": max_val,
            "min": min_val,
            "dynamics_std": std_val,
        })

    return {
        "rms_frame_count": num_frames,
        "channels_analysis": channel_metrics,
    }
