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