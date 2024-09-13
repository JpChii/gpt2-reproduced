"""
FineWeb-Edu dataset (for pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and saves the tokenized data shards to disk
"""

import os
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

# Variables for loading the dataset and saving the shards
local_dir = "edu_fineweb10B"
repo_name = "HuggingFaceFW/fineweb-edu"
repo_sample = "sample-10BT"
shard_size = int(1e8) # 100M tokens per share, total of 100 shards --> 10B

# Create the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Load the dataset
fw = load_dataset(repo_name, repo_sample, split="train")

# Init Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
# Pad token for beginning and end of document
eot = tokenizer._special_tokens['<|endoftext|>']

# Function to tokenize a doc from fineweb-edu
def tokenize(doc):
    # Tokenize a single doc and returns a numpy array of uint16 tokens
    # start with eot token
    tokens = [eot]
    # Encode without special tokens
    tokens = tokens.extend(tokenizer.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    # check tokens are within uint16 --> 66,536 --> 2**16
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_unit16 = tokens_np.astype(np.uint16)
    return tokens_np_unit16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# Creation of shards with multiprocessing
num_processes = mp.cpu_count()
# Create a pool of processes for N CPUs
with mp.Pool(num_processes) as pool:
    shard_index = 0
    # Buffer to store shards
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0 # initial token count
    progress_bar = None # progress bar

    # Use imap with multiprocesing to tokenize 16 documents in parallel across num_processes
    for tokens in  pool.imap(
        func=tokenize, # Function for imap
        iterable=fw, # Iterable to applied with funk
        chunksize=16, # Number of iterabled to be selected for a process
        ):

        # If Number of tokens in buffer is less than shard_size
        if token_count + len(tokens) < shard_size:
            # Add tokens to buffer
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)

            # Update progress bar
            if progress_bar is None:
                progress_bar = tqdm(
                    total=shard_size,
                    unit="tokens",
                    desc=f"Shard {shard_index}",
                )
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index}:06d")
            # Calculate remainig tokens to be added in current shard and add them
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            # Save shard
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            # For next shard
            progress_bar = None
            # Populate the next shard with leftover of current tokens
            all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    if token_count != 0:
        # Save the last shard
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index}:06d")
        write_datafile(filename, all_tokens_np)