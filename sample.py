# We'll sample some token from our scratch skeleton GPT2 with hugging face weights
# To compare with pipeline generation
import torch
from transformers import GPT2LMHeadModel
from torch.nn import functional as F
from gpt2 import GPT2
import tiktoken

verbose = True
device = "cuda" if torch.cuda.is_available() else "cpu"
num_sequences = 5
max_length = 30

# Load the model
model = GPT2.from_pretrained(model_type="gpt2")
# model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path="gpt2")
# Move it to eval mode
model.eval()
# Move it to cuda, when gpu is available. This moves all the variables to gpu device
model.to(device)

prefix = "Hello, I'm a language model,"


encoder = tiktoken.get_encoding("gpt2")
tokens = encoder.encode(prefix)
tokens = torch.tensor(tokens, dtype=torch.long) # (Number of tokens)
tokens = tokens.repeat(
    num_sequences, # Number of times to repeate the sequence. 5 
    1, # dimension to repeate along
)
print(f"Tokens shape after repeat: {tokens.size()}")
x = tokens.to(device)

# generate, right now B=5, X=8
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Number of tokens is less than max_length
while x.size(1) < max_length:
    # Calculate forward pass in no_grad
    with torch.no_grad():
        # 1. Forward pass
        logits = model(x)
        # For GPT LM head from huggingface
        # logits = model(x)[0]
        # 2. Get last token logits
        last_token_logits = logits[:, -1, :]
        # 3. Get probabalities
        probs = F.softmax(last_token_logits, dim=-1)
        # 4. Get top_50 tokens, this is the implementation in hugging face pipeline
        top_k_probs, top_k_indices = torch.topk(
            input=probs,
            k=50,
            dim=-1, # Channel/embedding dimension
        )
        # 5. Select a single token from top 50 probabalities
        ix = torch.multinomial(
            input=top_k_probs,
            num_samples=1,
        )
        # 6. Get the index of the probabality
        xcol = torch.gather(
            input=top_k_indices,
            dim=-1,
            index=ix,
        )
        # 7. concatenate this index to x
        x = torch.cat(
            tensors=(x, xcol),
            dim=1, # Concatenate along token dimension
            )

print(f"Sequence shape after geneartion: {x.size()}")

# Print sequences
for sequence in range(num_sequences):
    tokens = x[sequence, :].tolist()
    text = encoder.decode(tokens)
    print(f"Sequence No: {sequence}")
    print(f"> {text}")