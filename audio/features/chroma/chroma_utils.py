def get_chroma_neural_network_data(audio_data: dict) -> dict:
    chroma_metrics = audio_data.get("audio", {}).get("chroma_metrics", {})
    distribution = chroma_metrics.get("chroma_mean_distribution", [])

    chroma_dict = {}
    for i, val in enumerate(distribution):
        chroma_dict[f"chroma_bin_{i + 1}"] = val

    return chroma_dict