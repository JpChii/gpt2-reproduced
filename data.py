import tiktoken
import torch
INPUT_FILE = "input.txt"

# Setup device agnostic
device = "cpu" 
if torch.cuda.is_available():
    device = "cuda"    
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

with open(INPUT_FILE, "r") as file:
    data = file.read()
encoder = tiktoken.get_encoding("gpt2")

def create_data(B, T):
    """_summary_

    create x, y batches for next word prediciton pretraining objective.

    Working logic:
        1. Encode the text with tiktoken
        2. Get B*T + 1(target token) from dataset. Additional final token is for dataset creation.
            * tokens[-1].view(B, T) -> input tokens. Leave final token for target
            * tokens[1:].view(B, T) -> output tokens. Leave first token as it can't be predicted as target
        3. x, y have the same shape

            Args:
                B (int): Batch Size
                T (int): Number of tokens in the sequence

    Returns:
        tuple: x, y batch data
    """
    num_tokens = B*T
    # Encode tokens, encoding 1000 with 3x compression ration of GPT2 tokenizer
    tokens = encoder.encode(data[:1000])
    tokens = torch.tensor(tokens)[:B*T + 1]
    print(f"Tokens size: {tokens.size()}")
    x = tokens[:-1].view(B, T).to(device)
    y = tokens[1:].view(B, T).to(device)
    print(f"Shape of x and y: {x.size()}, {y.size()}")
    return x, y

create_data(8, 32)