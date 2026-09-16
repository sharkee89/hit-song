def get_spectral_neural_network_data(audio_data: dict) -> dict:
    spectral_metrics = audio_data.get("audio", {}).get("spectral_centroid_metrics", {})
    max_hz = 12000.0
    avg_hz = spectral_metrics.get("average_centroid_hz", 0.0)
    peak_hz = spectral_metrics.get("peak_centroid_hz", 0.0)

    return {
        "average_centroid": min(max(avg_hz / max_hz, 0.0), 1.0),
        "peak_centroid": min(max(peak_hz / max_hz, 0.0), 1.0)
    }