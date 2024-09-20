# https://arxiv.org/pdf/1905.07830 - 2018
# Hellaswag is an evaluation dataset for commonsense NLI
# Dataset has a single context and four options out of which
# three options are adversial generations. Adversial here is 
# generation is difficult to identify for pretrained models but easy for humans.
# We'll use this dataset for evaluating the GPT2 model.

# To do the evaluation for a small model(124M), 
# we can't feed all four options with context to the model and choose the right answer
# Here, we'll use construct four options with all four endings
# and then feed them to the model to get logits.
# Calculate the logits for tokens and sum them to get the average probabality for all tokens
# Sequence with highest probabality or lowest loss
# if it matches with the label(1,2,3,4) options then the generation is correct or else wrong.

# Dataset - https://github.com/rowanz/hellaswag
# https://github.com/rowanz/hellaswag/tree/master/data directory has three files for test, train, val respectivley in jsonl format

# Single sample
# {
#     "ind": 14,
#     "activity_label": "Wakeboarding",
#     "ctx_a": "A man is being pulled on a water ski as he floats in the water casually.",
#     "ctx_b": "he",
#     "ctx": "A man is being pulled on a water ski as he floats in the water casually. he",
#     "split": "test",
#     "split_type": "indomain",
#     "label: 3,
#     "endings": [
#         "mounts the water ski and tears through the water at fast speeds.",
#         "goes over several speeds, trying to stay upright.",
#         "struggles a little bit as he talks about it.",
#         "is seated in a boat with three other people."
#     ],
#     "source_id": "activitynet~v_-5KAycAQlC4"
# }

# ind: datasetID
# activity_label: The ActivityNet or WikiHow label for this example
# context: There are two formats. The full context is in ctx. 
#   When the context ends in an noun phrase, this incomplete noun phrase is in ctx_b, 
#   and then context up until then is in ctx_a. If ctx_b is empty then ctx_a is ctx.
# label: correct answer (1,2,3,4)

# gpt2 (124M)
# - eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
# - this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

# gpt2-xl (1558M)
# - eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
# - this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

# The validation set of HellaSwag has a total of 10,042 examples.

# imports
import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

# Encoder
enc = tiktoken.get_encoding("gpt2")

# Functions to download the dataset with split
hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

# download single file
def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname, # Description for progress bar
        total=total, # Total byte contents
        unit="iB", # bytes unit
        unit_scale=True, # Scale with unit
        unit_divisor=chunk_size, # Divide by chunk_size
    ) as bar: # Create progress bar
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
    
# Download
def download(split):
    """_summary_

    Download hellaswag split url

    Args:
        split (str): Possible values train, test, val
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, "hellaswag_" + split + ".jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}")
        download_file(data_url, data_filename)

# Render example
def render_example(example):
    """_summary_

    Given an input from hellaswag, create:
        1. context_tokens + end_tokens
        2. mask
        3. label
        4. combine all three in a dictionary

    Args:
        example (dict): Dictionary with above data example
    """

    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce eval on C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end) # note: pretending " " for GPT2
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([1] * len(ctx_tokens) + [0] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # Calculate the maximum length out of four samples
    max_len = max([len(x) for x in tok_rows])
    # Buffer to store tokens
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        # Assign tokesn to padded rows
        tokens[i, : len(tok_row)] = torch.tensor(tok_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

# Yield a single example
def iterate_examples(split):
    "generator function to yield a single sample"
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate(model_type, device):

    torch.set_float32_matmul_precision("high") # use tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)

    # Variables to tract metrics
    num_correct_norm = 0 # Normalized corred preds
    num_correct = 0 # Unnormalized corred preds
    num_total = 0 # total number of predictions

    for example in iterate_examples("val"):

        data, tokens, mask, label = render_example(example)
        # Move to device
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get logits
        logits = model(tokens).logits

        # Evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :1]).contiguous() # Exclude last logit
        shift_tokens = (tokens[..., 1:]).contiguous() # Exclude first token as it's prompt
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # Flatten logits along token dimension
        flat_shift_tokens = shift_tokens.view(-1) # Flatten tokens along token dimension

        # Calculate loss
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
        shift_losses = shift_losses.view(shift_tokens.size(0), -1) # Unflatten

        # Average losses where mask == 1 in each row
        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask # Set losses to 0 where mask == 0

        # Sum and dive by total number of 1s in mask
        sum_loss = masked_shift_losses.sum()
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        # Sample with lowest loss is most likley completion by model
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct_norm += int(pred_norm == label)
        num_correct += int(pred == label)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

    # debug: pretty print a few examples, and the losses in each case
    if num_total < 10:
        print("---")
        print(f"Context:\n {example['ctx']}")
        print(f"Endings:")
        for i, end in enumerate(example["endings"]):
            print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
        print(f"predicted: {pred_norm}, actual: {label}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)