import sys
import os
import numpy as np
import torch # Moramo ga uvesti da bismo radili sa tenzorima

# Adding parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Importing the Orchestrator
from neural.neural_orchestrator import NeuralOrchestrator

def start():
    # Initialization
    orc = NeuralOrchestrator()
    initial_weights = orc.model.linear.weight.detach().numpy()
    initial_bias = orc.model.linear.bias.detach().numpy()

    print(f"Weights before training: {initial_weights}")
    print(f"Bias before training: {initial_bias}")

    # START TRAINING
    orc.train(epochs=1000, learning_rate=0.01)
    optimized_weights = orc.model.linear.weight.detach().numpy()
    optimized_bias = orc.model.linear.bias.detach().numpy()

    print(f"\nOptimized weights: {optimized_weights}")
    print(f"Optimized bias: {optimized_bias}")

    # Save results to the parent (main) folder
    np.save(os.path.join(parent_dir, 'optimized_weights.npy'), optimized_weights)
    np.save(os.path.join(parent_dir, 'optimized_bias.npy'), optimized_bias)
    onnx_path = os.path.join(parent_dir, 'music_orchestrator.onnx')
    orc.save_to_onnx(onnx_path)

    print(f"\n💾 Optimized parameters and ONNX model saved to the main folder!")

if __name__ == "__main__":
    start()