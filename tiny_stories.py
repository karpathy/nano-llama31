"""
Downloads and tokenizes the TinyStories dataset.
- The download is from HuggingFace datasets.
- The tokenization is Llama 3.1 Tokenizer (with tiktoken).

The output is written to a newly created tinystories/ folder.
The script prints:

Number of shards: 50
Tokenizing val split...
writing 18,660,516 tokens to /home/ubuntu/nano-llama31/tinystories/TinyStories_val.bin
Tokenizing train split...
writing 907,021,844 tokens to /home/ubuntu/nano-llama31/tinystories/TinyStories_train.bin

And runs in few minutes two depending on your internet
connection and computer. The .bin files are raw byte
streams of uint32 numbers indicating the token ids.

The .bin file sizes are:
3.4G    /home/ubuntu/nano-llama31/tinystories/TinyStories_train.bin
72M     /home/ubuntu/nano-llama31/tinystories/TinyStories_val.bin
"""

import os
import glob
import json
import random
import requests
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from tokenizer import Tokenizer
# -----------------------------------------------------------------------------

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def write_datafile(filename, toks):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint32
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240801 # magic
    header[1] = 7 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    toks_np = np.array(toks, dtype=np.uint32)
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")

def process_shard(shard_index, shard_filename, tokenizer_path):
    # create tokenizer and encode function within the process
    tokenizer = Tokenizer(tokenizer_path)
    def encode(x):
        return tokenizer.encode(x, bos=True, eos=True)

    with open(shard_filename, "r") as f:
        data = json.load(f)
    rng = random.Random(1337 + shard_index)
    rng.shuffle(data)
    all_tokens = []
    for example in data:
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = encode(text)
        all_tokens.extend(tokens)
    return all_tokens

def tokenize(tokenizer_path):
    # shard 0 will be the val split, rest is train
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    for split_name, split_shards in [("val", val_shards), ("train", train_shards)]:

        print(f"Tokenizing {split_name} split...")
        all_tokens = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_shard, shard_index, shard_filename, tokenizer_path)
                       for shard_index, shard_filename in enumerate(split_shards)]
            for future in as_completed(futures):
                all_tokens.extend(future.result())

        split_filename = os.path.join(DATA_CACHE_DIR, f"TinyStories_{split_name}.bin")
        write_datafile(split_filename, all_tokens)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinystories")
    tokenizer_path = "llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model"
    download()
    tokenize(tokenizer_path)
