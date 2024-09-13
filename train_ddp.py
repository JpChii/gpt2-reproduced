import os
import time
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from gpt2 import GPT2, GPTConfig
from data import DataLoaderLite
from utils import get_lr

# DDP Launch 2 GPUs
# torchrun --standalone --nproc_per_node=2 train_ddp.py

# Setup DDP
# torchrun command sets the required env variables - RANK, LOCAL_RANK, WORLD_SIZE
# Check if the run is ddp
ddp = int(os.environ.get("RANK", -1)) != -1
print(f"Is running on DDP: {ddp}")
if ddp:
    # DDP requires cuda, we set the device appropriatley according to rank using cuda
    assert torch.cuda.is_available(), "DDP requires cuda"
    dist.init_process_group(backend="nccl")
    # Get env variables from ddp
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    # Cuda device/GPU is assigned based in rank
    device = f"cuda:{ddp_local_rank}"
    # This is set to avoid collision within GPUs
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpoint etc.
else:
    # Vanilla , non ddp run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = "cpu"
    master_process = True
    # autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

# Set seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# For torch.autocast as cuda:ddp_rank throws error if device is not cuda or cpu or supported backends
device_type = "cuda" if device.startswith("cuda") else "cpu"

B = 16
T = 1024
max_steps = 50
total_batch_size = 524288  # 2**19 and divisible by B*T

assert (
    total_batch_size % (B * T * ddp_world_size) == 0
), f"Total batch size {total_batch_size} must be divisible by B*T {B*T}"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total batch size: {total_batch_size}")
    print(f"Grad accum steps:=> {total_batch_size} // {B} * {T} = {grad_accum_steps}")
    print(f"Number of tokens processing in parallel: {B*T*ddp_world_size}")

# create dataloader, comes above float32 matmul to avoid loading data in GPU
data_loader = DataLoaderLite(input_file="input.txt", B=B, T=T, num_processes=ddp_world_size, process_rank=ddp_rank)

# use TF32 tensor core, speeding up matmul
torch.set_float32_matmul_precision("high")

# Initialize model
losses = []
model = GPT2(GPTConfig).to(device)
# DDP does below:
# Forward pass remains the same across processes.
# Average gradients across processes for all model parameters and synchronize the average to all processes
# During backward pass, DDP sends communication between processes(probably from master process) as soon as the gradients are calculated for parameters creating an overlap before average of gradients and synchronize the averaged gradients across processes.
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
# called DDP(model) before compile to allow torchdynamo to graph-break optimizations based on bucket sizes
# https://pytorch.org/docs/stable/notes/ddp.html
model = torch.compile(model)


# Initialize optimizer
# optim = torch.optim.AdamW(
#     params=model.parameters(), # Parameters for backprop
#     lr=6e-4, # This is good initial learning rate
#     betas=(0.9, 0.95), # Betas from GPT3 paper
#     eps=1e-8, # Eps from GPT3 paper, Default is also same
# )
optim = raw_model.configure_optimizers(
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

        # Set grad sync to False until last microstep
        if ddp:
            # This makes sure grads are not synchronized until last microstep
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        x, y = data_loader.next_batch(micro_step=micro_step,step=step)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = raw_model(x, y)

        # Scale loss to account for gradient accumulation
        # because gradient just add on each successive backward().
        # Generally loss has an objective SUM or MEAN. cross_entropy_loss objective/reduction is MEAN.
        # To calculate mean of loss we do the below step
        loss = loss / grad_accum_steps
        # detach to remove this step from gradient calculation
        loss_accum += loss.detach()
        # Backward pass
        loss.backward()

    # Average DDP Loss
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
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
    tokens_processed = B * T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / (t1 - t0)
    if master_process:
        print(
            f"Step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tokens/sec: {tokens_per_sec:.2f}"
        )
    losses.append(loss.item())

if ddp:
    dist.destroy_process_group()