import time
import math
import torch
from gpt2 import GPT2, GPTConfig
from data import create_data, DataLoaderLite

INIT_LOSS = False
SINGLE_BATCH_OVERFIT = False
ALL_DATA_OVERFIT = True
B = 16
T = 1024
max_steps = 50
max_lr = 10
min_lr = max_lr * 0.1
warmup_steps = 10

def get_lr(it):
    if it < warmup_steps:
        # Starts from min_lr to max_lr during warmup
        lr = max_lr * (it + 1) / warmup_steps
        return lr

    if it > max_steps:
        # After max steps use min_lr
        return min_lr
    
    # Betweem warmup_steps and max_steps use cosine annealing
    # Decay ratio increases from 0 to 1
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    print(f"Decay ratio: {decay_ratio}")
    # math.cos(math.pi * decay_ratio) products value from -1 to 1 based on decay_ratio
    # 1.0 is added to shift value from -1 to 1 to 1 to 2
    # 0.5 is multiplied to shift value from 1 to 2 to 0 to 1, coeff starts from 1 and decreases until 0
    # As decay ration increases coeff decreases, thus reducing learning rate
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    lr = min_lr + (max_lr - min_lr) * coeff
    return lr

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
    x, y = create_data(B=B, T=T)
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

    x, y = create_data(B=B, T=T)
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

# Main code for speeding up
if ALL_DATA_OVERFIT:
    torch.set_float32_matmul_precision("high")
    print("Running all data overfit training loop")
    # create dataloader
    data_loader = DataLoaderLite(input_file="input.txt", B=B, T=T)

    # Initialize model
    losses = []
    model = GPT2(GPTConfig).to(device)
    model = torch.compile(model)
    # Initialize optimizer
    optim = torch.optim.AdamW(
        params=model.parameters(), # Parameters for backprop
        lr=3e-4, # This is good initial learning rate
        betas=(0.9, 0.95), # Betas from GPT3 paper
        eps=1e-8, # Eps from GPT3 paper, Default is also same
    )

    # Initialize training loop
    for step in range(max_steps):
        t0 = time.time()
        x, y = data_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # Optimizer zero grad
        optim.zero_grad(set_to_none=True)
        # Forward pass
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # Backward pass
        loss.backward()
        # Clip gradient norm
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set learning rate for this iteration
        lr = get_lr(step)
        # There's a notion in PyTorch where multiple param_groups might exist, 
        # hence we're looping through them and setting the lr in below fashion
        for param_group in optim.param_groups:
            param_group['lr'] = lr
        # Update parameters
        optim.step()
        # Wait until all instruction sent from cpu to gpu are completed
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0)*1000 # ms
        print(f"Step {step:4d} | loss: {loss.item():.6f} | norm: {norm:.4f} | dt: {dt:.2f}ms | tokens/sec: {(B*T)/(t1-t0):.2f}")
        losses.append(loss.item())
    print(losses)