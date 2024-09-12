import os
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
    num_tokens = B * T
    # Encode tokens, encoding 1000 with 3x compression ration of GPT2 tokenizer
    tokens = encoder.encode(data[:1000])
    tokens = torch.tensor(tokens)[: B * T + 1]
    print(f"Tokens size: {tokens.size()}")
    x = tokens[:-1].view(B, T).to(device)
    y = tokens[1:].view(B, T).to(device)
    print(f"Shape of x and y: {x.size()}, {y.size()}")
    return x, y


# This is an improved version of create_data()
class DataLoaderLite:
    def __init__(self, input_file: str, B: int, T: int, num_processes: int, process_rank: int):
        """
        Initializes the DataLoaderLite class.

        Data is sampled without replacement.
        Data is loaded in chunks and a chunk wont be repeated with a step or epoch to avoid overfitting.

        Args:
            input_file (str): Path to the input file.
            B (int): Batch size.
            T (int): Number of tokens in each sequence.
            num_processes (int): Number of GPUs.
            process_rank (int): ID of the current GPU. This will be used to offset the data samples.

        Returns:
            None
        """

        self.batch = B
        self.token_size = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Load the data
        assert os.path.exists(input_file), f"{input_file} does not exist"
        with open(input_file, "r") as file:
            data = file.read()

        # Encode the data
        self.encoder = tiktoken.get_encoding("gpt2")
        self.tokens = self.encoder.encode(data)
        self.tokens = torch.tensor(self.tokens)

        if self.process_rank == 0:
            # Calculate number of batches with B, T dimensions
            print(f"Number of tokens: {len(self.tokens)}")
            print(
                f"Total number of batches per epoch: {len(self.tokens) // (self.batch * self.token_size)}"
            )

        # Cursor for current position in the tokens
        # start at 0 for rank 0, B*T*1 for rank 1, B*T*2 for rank 2, etc
        self.current_position = self.batch * self.token_size * self.process_rank

    def next_batch(self):
        """
        Returns next batch of data.

        This function returns the next batch of data from the stored tokens. The batch size is determined by the attributes batch and tokens.
        The function returns a tuple of two tensors, x and y, of size (batch, tokens). These tensors are used as input and target for the language model.
        The function also updates the current position in the tokens. When the current position reaches the end of the tokens, the position is reset to the beginning.
        The tensors x and y are moved to the device specified in the global variable device.
        """
        buf = self.tokens[
            self.current_position : self.current_position
            + self.batch * self.token_size
            + 1
        ]
        x = buf[:-1].view(self.batch, self.token_size)
        y = buf[1:].view(self.batch, self.token_size)
        self.current_position += self.batch * self.token_size * self.process_rank

        # When we run out of tokens begin from beginning
        if self.current_position + (self.batch * self.token_size + self.process_rank + 1) > len(
            self.tokens
        ):
            self.current_position = 0

        return x, y