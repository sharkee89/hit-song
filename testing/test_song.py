import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from neural.neural_orchestrator import NeuralOrchestrator


def run_test():
    orc = NeuralOrchestrator()

    weights_path = os.path.join(root_dir, 'optimized_weights.npy')
    bias_path = os.path.join(root_dir, 'optimized_bias.npy')

    if os.path.exists(weights_path):
        orc.weights = np.load(weights_path)
        orc.bias = np.load(bias_path)[0]
        print(f"✅ Optimised weights loaded: {orc.weights}")
    else:
        print("⚠️ Default weights are being used because optimized_weights.npy was not found.")

    song_name = "data/testing_songs/{NAME_OF_THE_SONG}"
    song_path = os.path.join(root_dir, song_name)
    artist_name = "Taylor Swift"

    if not os.path.exists(song_path):
        print(f"❌ Error: file on that path not found: {song_path}")
        return

    print(f"\n--- Testing: {song_name} ---")
    try:
        orc.weights = np.array([0.15, 0.70, 0.10, 0.05])
        orc.bias = -0.8

        print("🧪 Model calibrated: Audio(45%), Context(40%), Trend(10%), Viral(5%)")
        result = orc.evaluate_song(song_path, artist_name)

        print("\n" + "=" * 40)
        print(f" FINAL HIT SCORE: {result['hit_potential_score'] * 100:.2f}%")
        print("=" * 40)

        print("\nContributions by agents (Activations):")
        factors = ["Sonic Quality (Audio)", "Artist Legacy (Context)", "Market Alignment (Trend)",
                   "Virality Index (Viral)"]
        signals = [
            result['detailed_reports']['audio']['activation_value'],
            result['detailed_reports']['context']['activation_value'],
            result['detailed_reports']['trend']['activation_value'],
            result['detailed_reports']['viral']['activation_value']
        ]

        for name, val in zip(factors, signals):
            print(f"- {name}: {val:.4f}")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")


if __name__ == "__main__":
    run_test()