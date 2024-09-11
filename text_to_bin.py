"""
Takes any pure text file and converts it to a binary file using 
the llama3_1 tokenizer.
"""

import os
import numpy as np
import sys
import random
from tokenizer import Tokenizer

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]
tokenizer_path = "llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model"

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

with open(input_file_path, 'r') as file:
    text = file.read()

tokenizer = Tokenizer(tokenizer_path)
def encode(x):
    return tokenizer.encode(x, bos=True, eos=True)

tokens = encode(text)

assert len(tokens) < 2**31, "token count too large" # ~2.1B tokens

 # construct the header
header = np.zeros(256, dtype=np.int32)
header[0] = 20240801 # magic
header[1] = 7 # version
header[2] = len(tokens) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
# construct the tokens numpy array, if not already
tokens_np = np.array(tokens, dtype=np.uint32)
    
# write to file
print(f"writing {len(tokens):,} tokens to {output_file_path}")
with open(output_file_path, "wb") as f:
    f.write(header.tobytes())
    f.write(tokens_np.tobytes())