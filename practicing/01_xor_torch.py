import torch
import torch.nn as nn
import torch.optim as optim


# 1. MODEL ARCHITECTURE (Upgraded)
class SimpleXORModel(nn.Module):
    def __init__(self):
        super(SimpleXORModel, self).__init__()
        self.hidden = nn.Linear(2, 8)
        self.output = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x


# 2. DATA PREPARATION
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 3. INITIALIZATION
model = SimpleXORModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. TRAINING LOOP
print("--- Training with Adam & 8 Neurons ---")
for epoch in range(5001):
    predictions = model(X)
    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

print("\n--- Improved Final Results ---")
with torch.no_grad():
    final_preds = model(X)
    for i in range(len(X)):
        pred = final_preds[i].item()
        status = "CORRECT" if round(pred) == y[i].item() else "WRONG"
        print(f"Input: {X[i].tolist()} -> Predicted: {pred:.4f} ({status})")

dummy_input = torch.tensor([[0, 0]], dtype=torch.float32)
torch.onnx.export(
    model,
    dummy_input,
    "simple_xor.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['my_input'],
    output_names=['my_output']
)
print("\n📦 Model re-exported with fixed names!")
print("\n📦 Model saved as 'simple_xor.onnx'")