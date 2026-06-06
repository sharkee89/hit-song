import os
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_name = "shakespeare.txt"

if not os.path.exists(file_name):
    print("Downloading shakespeare file from the internet...")
    response = requests.get(data_url)
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(response.text)
    print("File successfully saved locally!")
else:
    print("File already exist locally, loading...")

with open(file_name, "r", encoding="utf-8") as f:
    text = f.read()

print(f"\n[INFO]: Total characters in first dataset: {len(text)}")
print("-" * 100)
print(f"First 120 characters from file:\n", text[:120])
print("-" * 100)

chars =sorted(list(set(text)))
vocab_size = len(chars)

print(f"[INFO]: Dictionary has exact {vocab_size} unique tokens")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

all_tokens = torch.tensor(encode(text), dtype=torch.long)
print(f"[INFO]: Shape of whole data tensor: {all_tokens.shape}")

n = int(0.9 * len(all_tokens))
train_data = all_tokens[:n]
val_data = all_tokens[n:]

torch.manual_seed(42)
block_size = 8
batch_size = 4

def get_batch(split):
    """Randomly get small X and Y matrices from our ocean of numbers"""
    data_source = train_data if split == "train" else val_data

    ix = torch.randint(len(data_source) - block_size, (batch_size,))

    x = torch.stack([data_source[i:i+block_size] for i in ix])

    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])

    return x, y

X_batch, Y_batch = get_batch("train")

print("-" * 100)
print(f"Matrix X (input to neural network - dimension {X_batch.shape}):\n", X_batch)
print(f"\nMatrix Y (Correct answer - Meta dimension {Y_batch.shape}):\n", Y_batch)
print("-" * 100)

# =====================================================================
# Defining neural network
# =====================================================================

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
logits, loss = model(X_batch, Y_batch)

print("=" * 100)
print(f"Exit logits dimension (prediction): {logits.shape}")
print(f"Starting mathematical error (Loss) before training: {loss.item():.4f}")
print("=" * 100)

# =====================================================================
# Training loop
# =====================================================================

learning_rate = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(f"\n[INFO] Optimizer configured. Executing training loot...\n")

training_steps = 10000

for step in range(training_steps):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"Step {step:5d} | Current loss: {loss.item():.4f}")

print('-' * 100)
print(f"End of training! Final loss on last batch: {loss.item():.4f}")
print('-' * 100)

print("\n[INFO] Generating text from trained model...\n")

start_context = torch.zeros((1, 1), dtype=torch.long)
generated_tokens = model.generate(start_context, max_new_tokens=500)
print(decode(generated_tokens[0].tolist()))
print('-' * 100)