"""
Takes any pure text file and converts it to a binary file using 
the llama3_1 tokenizer.
"""

import os
import numpy as np
import sys
from tokenizer import Tokenizer

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

with open(input_file_path, 'r') as file:
        text = file.read()

# tokens = llama.tokenize(text)
tokens = text

with open(output_file_path, 'wb') as file:
    file.write(bytes(tokens))