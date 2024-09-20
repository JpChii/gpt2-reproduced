import math

def get_lr(it, warmup_steps, max_steps, max_lr, min_lr):
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
    # math.cos(math.pi * decay_ratio) products value from -1 to 1 based on decay_ratio
    # 1.0 is added to shift value from -1 to 1 to 1 to 2
    # 0.5 is multiplied to shift value from 1 to 2 to 0 to 1, coeff starts from 1 and decreases until 0
    # As decay ration increases coeff decreases, thus reducing learning rate
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    lr = min_lr + (max_lr - min_lr) * coeff
    return lr

import torch
from torch.nn import functional as F
def get_most_likey_row(tokens: torch.tensor, masks: torch.tensor, logits:torch.tensor) -> int:
    """_summary_

    Hellaswag evaluation for training.

    Accepts tokens, mask of shape [4, max_sequence_length] and logits of shape [4, max_sequence_length, vocab_size]
    Calculates cross entropy loss for mask == 1
    Return the argmin of sequence with lowest norm loss

    Args:
        tokens (torch.tensor): Tokens
        masks (torch.tensor): Masks
        logits (torch.tensor): logits for sequences
    """

    # Exclude first token
    shift_tokens = tokens[..., 1:].contiguous() # [B, T]
    shift_logits = logits[..., :-1, :].contiguous() # [B, T, C]

    # Flatten
    flat_shift_tokens = shift_tokens.view(-1) # [B*T]
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # [B*T , C]

    # Calculate loss
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none") # [B*T]
    shift_losses = shift_losses.view(tokens.size(0), -1) # [B, T]

    # Masked loss
    shift_mask = masks[..., 1:].contiguous()
    shift_mask_losses = shift_losses * shift_mask

    # Sum and norm loss across tokens
    sum_loss = shift_mask_losses.sum(dim=1) # [4]
    norm_loss = shift_mask_losses.sum(dim=1) / shift_mask.sum(dim=1)

    # Get prediction with lowest loss
    pred_norm = norm_loss.argmin().item()
    return pred_norm

