import os
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import json
import joblib
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class HitPredictionModel(nn.Module):
    def __init__(self):
        super(HitPredictionModel, self).__init__()
        self.linear = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x) / 0.8)


class NeuralOrchestrator:
    def __init__(self):
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from neural_audio_agent import NeuralAudioAgent
        from neural_context_agent import NeuralContextAgent
        from neural_trend_agent import NeuralTrendAgent
        from neural_viral_agent import NeuralViralAgent

        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv('GOOGLE_AI_API_KEY')

        self.audio_agent = NeuralAudioAgent(api_key)
        self.context_agent = NeuralContextAgent()
        self.trend_agent = NeuralTrendAgent()
        self.viral_agent = NeuralViralAgent()

        self.model = HitPredictionModel()

        # Inicijalizacija težina
        with torch.no_grad():
            self.model.linear.weight = nn.Parameter(torch.tensor([[0.4, 0.5, 0.1, 0.2]]))
            self.model.linear.bias = nn.Parameter(torch.tensor([-1.2]))

    def train(self, epochs=100, learning_rate=0.1):
        # Putanja do trening podataka (pazi na foldere!)
        data_path = os.path.join(parent_dir, 'training', 'training_data.json')

        if not os.path.exists(data_path):
            print(f"❌ Error: Cannot find {data_path}")
            return

        with open(data_path, 'r') as f:
            data_set = json.load(f)

        print(f"\n🚀 Training started on {len(data_set)} samples...")
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for item in data_set:
                inputs = torch.tensor(item['signals'], dtype=torch.float32)
                target = torch.tensor([item['target']], dtype=torch.float32)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3} | Loss: {running_loss / len(data_set):.6f}")

    def save_to_onnx(self, filename="music_orchestrator.onnx"):
        self.model.eval()
        dummy_input = torch.randn(1, 4)
        torch.onnx.export(self.model, dummy_input, filename)
        print(f"📦 Exported to {filename}")


if __name__ == "__main__":
    try:
        orchestrator = NeuralOrchestrator()
        orchestrator.train()
        orchestrator.save_to_onnx()
    except Exception as e:
        print(f"Error: {e}")