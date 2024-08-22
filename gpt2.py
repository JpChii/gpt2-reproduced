import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    """
    Config class to hold hyperparmeters of the model
    """
    block_size = 1024 # max sequence lenth
    vocab_size = 50257 # Number of tokens: 256 byte tokens + 1 <|endoftext|> token + 50,000 BPE merges
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dim

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

    def forward(self, x):
        """_summary_

        Accepts input tensor, performs attention mechanism batchwise, returns projected output
        """

        B, T, C = x.size() # batch size, sequence length, embedding dimension(n_embd)
        # Calculate query, key, values for all heads in batch(using torch.view()) and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C(number of channels) = nh*ns or C/n_head = head_size
        # GPT-2(124M), n_heads=12, hs=64, so nh*ns=C=768 channels in the Transformer
        
        # 1. querykeyvalue vector
        qkv = self.c_attn(x)
        # 2. Split this into query, key, value using torch.split(), shape of c_attn(B, n_embd, 3*n_embd), split along 2nd dim
        # to get q,k,v of shape (B, n_embd, n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # 3. Multi Headed Attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, ns)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, ns)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, ns)
        # 4. Scaled dor product, the reshaping above is for this
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # Flash attention
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
        self.c_fc = nn.Linear(self.config.n_embd, 4*self.config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*self.config.n_embd, self.config.n_embd)

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
        x = x + self.MLP(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # config to access parameters
        self.config = config

        # Layers of the model to be called in forward function implementation
        # transformers naming convention from sd_hf
        self.transformers = nn.ModuleDict(dict(
            # wte naming convention from sd_hf, Token Embeddings
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            # Position embeddings
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            # construct the module list h.0, h.1...,h.11 in st_hf
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            # final layer normalization
            ln_f = nn.LayerNorm(self.config.n_embd)
        ))
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

