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
max_lr = 6e-4  # From GPT2 paper
min_lr = max_lr * 0.1
warmup_steps = 10
total_batch_size = 524288  # 2**19 and divisible by B*T
assert (
    total_batch_size % (B * T) == 0
), f"Total batch size {total_batch_size} must be divisible by B*T {B*T}"
grad_accum_steps = total_batch_size // (B * T)
print(f"Total batch size: {total_batch_size}")
print(f"Grad accum steps:=> {total_batch_size} // {B} * {T} = {grad_accum_steps}")


def get_lr(it):
    """_summary_
    * Linear warm up until 10 steps
        * cosine decay until a certain limit
        * and continue
        * max_lr from GPT3 paper for GPT-small is $6 * 10^{-4}$, our's is right now $3 * 10^{-4}$
    * learning rate stops at the end of cosine decay as it matches with max_steps
    """
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
        params=model.parameters(),  # Parameters for backprop
        lr=3e-4,  # This is good initial learning rate
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
    # create dataloader, comes above float32 matmul to avoid loading data in GPU
    data_loader = DataLoaderLite(input_file="input.txt", B=B, T=T)

    torch.set_float32_matmul_precision("high")
    print("Running all data overfit training loop")

    # Initialize model
    losses = []
    model = GPT2(GPTConfig).to(device)
    model = torch.compile(model)
    # Initialize optimizer
    # optim = torch.optim.AdamW(
    #     params=model.parameters(), # Parameters for backprop
    #     lr=6e-4, # This is good initial learning rate
    #     betas=(0.9, 0.95), # Betas from GPT3 paper
    #     eps=1e-8, # Eps from GPT3 paper, Default is also same
    # )
    optim = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=6e-4,
        device=device,
    )

    # Initialize training loop
    for step in range(max_steps):
        loss_accum = 0
        t0 = time.time()
        # Optimizer zero grad
        optim.zero_grad(set_to_none=True)
        # Forward pass
        for micro_step in range(grad_accum_steps):
            x, y = data_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            # detach to remove this step from gradient calculation
            loss_accum += loss.detach()
            # Backward pass
            loss.backward()

        # Clip gradient norm
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set learning rate for this iteration
        lr = get_lr(step)
        # There's a notion in PyTorch where multiple param_groups might exist,
        # hence we're looping through them and setting the lr in below fashion
        for param_group in optim.param_groups:
            param_group["lr"] = lr
        # Update parameters
        optim.step()
        # Wait until all instruction sent from cpu to gpu are completed
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000  # ms
        tokens_processed = B * T * grad_accum_steps
        tokens_per_sec = tokens_processed / (t1 - t0)
        print(
            f"Step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tokens/sec: {tokens_per_sec:.2f}"
        )
        losses.append(loss.item())
    print(losses)
