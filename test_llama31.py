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
    temperature: float = 0.0, # note: doing argmax decoding
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 32,
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

    # we get this from running reference.py
    expected_completions = [
        " to be found in the pursuit of happiness. But what is happiness? Is it the same as pleasure? Is it the same as contentment? Is it the",
        " the laws of physics are the same for all non-accelerating observers, and the speed of light in a vacuum is the same for all observers, regardless",
        " a collection of code snippets and examples for using the Large Language Model (LLM) in various applications. The repo is maintained by a community of developers and researchers",
        """ fromage
        cheese => fromage
        cheese => fromage
        cheese => fromage
        cheese => fromage
        cheese => fromage"""
    ]

    # init the model
    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        flash=False, # disable flash attention so we can get exact match to reference
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

    # check if the results match the expected outputs
    for result, expected in zip(results, expected_completions):
        ok = result["generation"] == expected
        if ok:
            print("OK")
        else:
            print("FAIL")
            print(f"Expected: {expected}")
            print(f"Got: {result['generation']}")

if __name__ == "__main__":
    fire.Fire(test_inference)
