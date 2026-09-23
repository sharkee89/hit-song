def get_rms_neural_network_data(audio_data: dict) -> dict:
    rms_metrics = audio_data.get("audio", {}).get("rms_metrics", {})
    return {
        "rms_mean": rms_metrics.get("average_rms", 0.0),
        "rms_max": rms_metrics.get("peak_rms", 0.0),
        "rms_min": rms_metrics.get("min_rms", 0.0),
        "rms_std": rms_metrics.get("std_rms", 0.0),
        "rms_median": rms_metrics.get("median_rms", 0.0),
        "rms_q25": rms_metrics.get("q25_rms", 0.0),
        "rms_q75": rms_metrics.get("q75_rms", 0.0),
        "rms_dynamic_range": rms_metrics.get("dynamic_range", 0.0),
        "rms_skewness": rms_metrics.get("skewness", 0.0),
        "rms_kurtosis": rms_metrics.get("kurtosis", 0.0)
    }