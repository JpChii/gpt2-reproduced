import torch
from gpt2 import GPT2, GPTConfig
from data import create_data, DataLoaderLite

INIT_LOSS = False
SINGLE_BATCH_OVERFIT = False
ALL_DATA_OVERFIT = True

# Setup device agnostic
device = "cpu" 
if torch.cuda.is_available():
    device = "cuda"    
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {device}")

if INIT_LOSS:
    # Random intialized model
    model = GPT2(GPTConfig)
    model.to(device)
    x, y = create_data(B=4, T=32)
    logits, loss = model(x, y)


if SINGLE_BATCH_OVERFIT:
    print("Running single batch overfit training loop")
    """_summary_
    Runs overfit on a single batch of data and returns loss
    """
    # Initialize model
    losses = []
    model = GPT2(GPTConfig).to(device)
    # Initialize optimizer
    optim = torch.optim.AdamW(
        params=model.parameters(), # Parameters for backprop
        lr=3e-4, # This is good initial learning rate
    )

    x, y = create_data(B=4, T=32)
    # Initialize training loop
    for i in range(50):
        # Optimizer zero grad
        optim.zero_grad(set_to_none=True)
        # Forward pass
        logits, loss = model(x, y)
        # Backward pass
        loss.backward()
        # Update parameters
        optim.step()
        print(f"Step {i}: {loss.item():.4f}")
        losses.append(loss.item())
    print(losses)

if ALL_DATA_OVERFIT:
    print("Running all data overfit training loop")
    # create dataloader
    data_loader = DataLoaderLite(input_file="input.txt", B=4, T=32)

    # Initialize model
    losses = []
    model = GPT2(GPTConfig).to(device)
    # Initialize optimizer
    optim = torch.optim.AdamW(
        params=model.parameters(), # Parameters for backprop
        lr=3e-4, # This is good initial learning rate
    )

    # Initialize training loop
    for i in range(50):
        x, y = data_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # Optimizer zero grad
        optim.zero_grad(set_to_none=True)
        # Forward pass
        logits, loss = model(x, y)
        # Backward pass
        loss.backward()
        # Update parameters
        optim.step()
        print(f"Step {i}: {loss.item():.4f}")
        losses.append(loss.item())
    print(losses)