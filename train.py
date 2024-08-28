import torch
from gpt2 import GPT2, GPTConfig
from data import create_data

# Setup device agnostic
device = "cpu" 
if torch.cuda.is_available():
    device = "cuda"    
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {device}")

model = GPT2(GPTConfig)
model.to(device)
x, y = create_data(B=4, T=32)
logits, loss = model(x, y)

print(loss)