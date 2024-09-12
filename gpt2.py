import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer
from dataclasses import dataclass
from transformers import GPT2LMHeadModel


@dataclass
class GPTConfig:
    """
    Config class to hold hyperparmeters of the model
    """

    block_size: int = 1024  # max sequence lenth
    vocab_size: int = 50304  # Number of tokens: 256 byte tokens + 1 <|endoftext|> token + 50,000 BPE merges, 50,304 for nice numbers
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dim


class CasualSelfAttention(nn.Module):
    """_summary_
    Attention block - where all relationships between tokens are learned

    Implement the same multi headed attention logic as nangpt but in much efficient manner
    Implements accepting a batch of inputs
    Calculates query-key-vectors
    Calculates attention scores
    Calculate scaled dot product attention to calculate attention weights - the tril value etc in nanogpt implementation
    Concatenate all attention weights

    Requires:
    1. key,query,value vectors
    2. Linear layer to return to embed, embed dimension to be passed to MLP block

    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # Ensure embedding dim is divisible by n_head to avoid errors
        assert self.config.n_embd % self.config.n_head == 0
        # 3 * -> instead of initalizing three seperate Linear layers, we'll split this later in forward pass
        self.c_attn = nn.Linear(self.config.n_embd, 3 * self.config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(self.config.n_embd, self.config.n_embd)
        # Parameter for residual connection normalization
        self.c_proj.NANO_GPT_INIT = 1
        # Not really bias more of a mask, but following OpenAI/HF naming convention
        # Creates (1,1,1024,1024) masked tril to avoid seeing the future tokens
        self.register_buffer(
            "bias",  # register a buffer
            torch.tril(torch.ones(self.config.block_size, self.config.block_size)).view(
                1, 1, self.config.block_size, self.config.block_size
            ),  # view as (1, 1, block_size, block_size)
        )

    def forward(self, x):
        """_summary_

        Accepts input tensor, performs attention mechanism batchwise, returns projected output
        """

        B, T, C = x.size()  # batch size, sequence length, embedding dimension(n_embd)
        # Calculate query, key, values for all heads in batch(using torch.view()) and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C(number of channels) = nh*ns or C/n_head = head_size
        # GPT-2(124M), n_heads=12, hs=64, so nh*ns=C=768 channels in the Transformer
        # T will match self.block_size in config

        # 1. querykeyvalue vector
        qkv = self.c_attn(x)
        # 2. Split this into query, key, value using torch.split(), shape of c_attn(B, n_embd, 3*n_embd), split along 2nd dim
        # to get q,k,v of shape (B, n_embd, n_embd)
        q, k, v = qkv.split(self.config.n_embd, dim=2)
        # 3. Multi Headed Attention
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(
            1, 2
        )  # (B, T, nh, hs) -> (B, nh, T, ns)
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(
            1, 2
        )  # (B, T, nh, hs) -> (B, nh, T, ns)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(
            1, 2
        )  # (B, T, nh, hs) -> (B, nh, T, ns)
        # 4. Scaled dot product, the reshaping above is for this
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )
        # 5. ReTranspose(B, T nh, ns), contiguous for memory efficiency, return to original shape
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 6. Output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """_summary_
    Computation block after attention block

    In two linear layers between activation we increase the computation
    by 4*embedding dimension. The 4 comes from sd_hf transformer.h.11.mlp.c_fc.weight torch.Size([768, 3072])
    3072 /4 = 768

    We use GELU, due to it's smoothness(Send a small until a certain limit in x-axis for negative values and then sends zero)
    This smoothening reduces vanishing gradients
    To replicate GPT2 paper approximate implementation of GELU from pytorch is used. This is no longer needed

    map in map-reduce

    Requires:
    1. Two Linear layers
    2. GELU activation
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Layers
        self.c_fc = nn.Linear(self.config.n_embd, 4 * self.config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * self.config.n_embd, self.config.n_embd)
        # Parameter for residual connection normalization
        self.c_proj.NANO_GPT_INIT = 1

    def forward(self, x):
        """
        Forward function call to implement the computation after attentino block
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """_summary_

    Block of attention plus computation

    Requires:
    1. Two normalization layer for pre-normalization with attention and computation(mlp)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(self.config.n_embd)
        self.attn = CasualSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(self.config.n_embd)
        self.mlp = MLP(self.config)

    def forward(self, x):
        """_summary_
        1. Accepts input
        2. Implements Normalization -> Casual Attentation + Residual Connection
        3. Implemennts Normalization -> MLP + Residual connection
        """
        # Input is normalized, attention, then residual connection is created
        # Remember addition just sends the gradient to both it's connections
        # A clean gradient goes  through residual connection and attention normalization block alters the gradient accordingly
        # Attention mechanism - tokens interact with each other, can be assumed like reduce
        x = x + self.attn(self.ln_1(x))
        # Computation mechanism - each tokens are computed individually, can be assumed like map
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPTConfig, master_process: bool = False):
        super().__init__()
        # config to access parameters
        self.config = config
        self.master_process = master_process

        # Layers of the model to be called in forward function implementation
        # transformers naming convention from sd_hf
        self.transformer = nn.ModuleDict(
            dict(
                # wte naming convention from sd_hf, Token Embeddings
                wte=nn.Embedding(self.config.vocab_size, self.config.n_embd),
                # Position embeddings
                wpe=nn.Embedding(self.config.block_size, self.config.n_embd),
                # construct the module list h.0, h.1...,h.11 in st_hf
                h=nn.ModuleList(
                    [Block(self.config) for _ in range(self.config.n_layer)]
                ),
                # final layer normalization
                ln_f=nn.LayerNorm(self.config.n_embd),
            )
        )
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        if self.master_process:
            print("Weights intializations started")
        self.apply(self._init_weights)
        if self.master_process:
            print("Weights intiailization complete")

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            # Residual connection normalization
            if hasattr(module, "NANO_GPT_INIT"):
                # 2 because there are two residual connections
                # Value from GPT2 paper
                std *= 2 * (self.config.n_layer**-0.5)

            # initialize module.weight(tensor in place) to normal distribution with mean 0 and std 0.02.
            # normal_ underscore at the end performs inplace intialization to the tensor
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

            if isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        """_summary_
        1. Accepts, idx(input tokens in batches)
        2. targets(optional) labels
        3. Create positional embeddings, token embeddings, pass through block list, get logits from lm_head
        4. Calculate loss if targets is not None
        4. Return logits and loss
        """

        # Acccepts batch of input ids
        B, T = idx.size()

        assert (
            T <= self.config.block_size
        ), f"Cannot process token length greater than block size: {self.config.block_size}"

        # Create position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_embd = self.transformer.wpe(pos)  # shape (T, n_embd)
        tok_embd = self.transformer.wte(idx)  # shape (B, T. n_embd)

        # Add positional information with word information
        # the same position embeddings applies for each input in batch, broadcasting happens internally
        x = tok_embd + pos_embd

        # Forward the block of transformer
        for block in self.transformer.h:
            x = block(x)

        # final layer normalization
        x = self.transformer.ln_f(x)

        # Get logits
        logits = self.lm_head(x)  # Shape (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross entropy((B*T, vocab_size), (B*T))
            # logits.view preservers final dimension and merges batch and tokens into a single dimension
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """_summary_

        Loads pretrained weights from huggingface

        1. Defines neural network parameters
        2. Load Local implementation model and hf model
        3. Load both state dicts, align keys and copy the weigts

        Args:
            model_type (str): gpt2 model types froem hugging face

        """

        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        print("Loading weights from pretrained gpt: %s" % model_type)

        # Define layer parameters for gpt series models and select
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        # Below params are same for all GPT2 variants
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        # Create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT2(config)

        # Get keys, # memory map sd_keys -> sd -> model the key copy is present inplace
        sd = model.state_dict()
        sd_keys = sd.keys()
        # remove bias keys
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # Load huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        # Get keys
        sd_hf = model_hf.state_dict()
        sd_hf_keys = sd_hf.keys()
        # Remove bias
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith(".attn.bias")]
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith(".attn.masked_bias")]
        # Four keys (attn.c_attn - Attnetion linear layers), (attn.c_proj - output linear projection of attention layer)
        # (mlp.c_fc, mlp.c_proj - MLP computation)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_hf_keys) == len(
            sd_keys
        ), f"Mismatched kyes: {len(sd_hf_keys)} != {len(sd_keys)}"

        for k in sd_hf_keys:
            if any(k.endswith(w) for w in transposed):
                # Check shapes, transpose fits Conv1D wieghts to a Linear layer
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    # Copy transposed weights from hf to scratch
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Match shape and vanilla copy over the other params
                assert k in sd_hf_keys, f"{k} not in sd_hf_keys"
                assert k in sd_keys, f"{k} not in sd"
                assert (
                    sd_hf[k].shape == sd[k].shape
                ), f"{k}_hf shape: {sd_hf[k].shape}, {k}_scratch shape: {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, device: str, use_zero: bool = False
    ):
        """_summary_

        Creates and returns an optimizer with betas, lr and eps from GPT3 paper, plus implements
        a weight decay of 0.1 to tensors > 2d in model parameters that has gradient enabled and 0.0
        weight decay for model parameters < 2d

        Args:
            weight_decay (float): weight decay parameter
            learning_rate (float): learning rate to be used
            device (str): device in use
            use_zero(bool): If True use ZeroRedundancyOptimizer. https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html

        Returns:
            _type_: _description_
        """
        # Start with all of the candidate parameters that require grad
        param_dict = {
            param_name: param
            for param_name, param in self.named_parameters()
            if param.requires_grad
        }  # Create optim groups. Parameters that require weight decay and doesn't require weight decay
        # add weight decay if params dim >= 2
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        non_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": non_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_non_decay_params = sum(p.numel() for p in non_decay_params)
        if self.master_process:
            print(
                f"Number of decayed parameter tensors: {len(decay_params)}, ({num_decay_params})"
            )
            print(
                f"Number of non-decayed parameter tensors: {len(non_decay_params)}, ({num_non_decay_params})"
            )
        # Use fused version if available, this uses optimizer to use kernel fusion to updated parameters if available
        # This speeds up training
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        if self.master_process:
            print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
            fused=use_fused,
        )
        if use_zero:
            optimizer = ZeroRedundancyOptimizer(
                    optim_groups,
                    lr=learning_rate,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=weight_decay,
                    fused=use_fused,                
                )
        return optimizer