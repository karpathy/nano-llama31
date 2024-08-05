"""
Outputs the same thing as reference.py
$ python test_llama31.py
"""

import fire
import time
import torch

from llama31 import Llama

def test_inference(
    ckpt_dir: str = "llama-models/models/llama3_1/Meta-Llama-3.1-8B",
    tokenizer_path: str = "llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
):

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "Clearly, the meaning of life is",
        "Simply put, the theory of relativity states that",
        """The repo llm.c on GitHub is""",
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    max_batch_size = len(prompts) # 4

    # init the model
    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # sample
    sample_rng = torch.Generator(device='cuda')
    sample_rng.manual_seed(1337)
    t0 = time.time()
    results = llama.text_completion(
        prompts,
        sample_rng=sample_rng,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    t1 = time.time()

    print(f"Generated in {t1 - t0:.2f} seconds")
    for prompt, result in zip(prompts, results):
        print(prompt, end="")
        print(f"{result['generation']}")
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(test_inference)
