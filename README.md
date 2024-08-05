# nano-llama31

This repo is to Llama 3.1 what nanoGPT is to GPT-2. i.e. it is a minimal, dependency-free implementation of the Llama 3.1 architecture, and it can train, finetune, and inference it very simply. This is compared to the official code release from Meta and the huggingface implementation, which both feature heavier dependencies and a lot more code (e.g. fair).

The code currently focuses on the 8B base model of Llama 3.1.

**WIP.**, actively developed, not ready for prime time.

### The reference

Let's begin with the official Llama 3.1 code release from Meta, which acts as our reference. This turns out to not be trivial because Meta's official repo [does not seem to](https://github.com/meta-llama/llama-models/issues/82) include documentation or instructions on how to actually use the models once you download them. But let's try:

Download the official `llama-models` repo, e.g. inside this project's directory is ok:

```bash
git clone https://github.com/meta-llama/llama-models.git
```

Download a model, e.g. the Llama 3.1 8B (base) model:

```bash
cd llama-models/models/llama3_1
chmod u+x download.sh
./download.sh
```

You'll have to enter a "URL from the email". For this you have to request access to Llama 3.1 [here](https://llama.meta.com/llama-downloads/). Then when it asks which model, let's enter `meta-llama-3.1-8b`, and then again one more time `meta-llama-3.1-8b` to indicate the base model instead of the instruct model. This will download about 16GB of data into `./Meta-Llama-3.1-8B` - 16GB because we have ~8B params in 2 bytes/param (bfloat16).

Now we set up our environment, best to create a new conda env, e.g.:

```bash
conda create -n llama31 python=3.10
conda activate llama31
```

Don't use a too recent Python (e.g. 3.12) because I think PyTorch support is still not 100% there. Now let's go back to the `llama-models` directory and install it. This will install the `llama-models` package which we can use to load the model:

```bash
cd ../../
pip install -r requirements.txt
pip install -e .
```

And now let's pop up to the root of the repo and run the generation script, which is an adaptation with some light edits of [example_text_completion.py](https://github.com/meta-llama/llama3/blob/main/example_text_completion.py) in Llama 3 repo:

```bash
cd ../
pip install fire
torchrun --nnodes 1 --nproc_per_node 1 reference.py \
    --ckpt_dir llama-models/models/llama3_1/Meta-Llama-3.1-8B \
    --tokenizer_path llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model
```

It feels a bit sketchy to use this code because the code is marked by Meta as "deprecated". So I don't currently have full confidence that this (deprecated) Llama 3.0 code is correct to use with the Llama 3.1 model.

But using the 3.0 code with the 3.1 model does print completions that look good:

```
Clearly, the meaning of life is to be found in the joy of living, in the joy of love, in the joy of work. The meaning of life is to be found in the joy of overcoming the self. The meaning of life is to be found in the joy of listening to music, in the joy of painting, in the joy of writing

==================================

Simply put, the theory of relativity states that the laws of physics are the same for all non-accelerating observers, and the speed of light in a vacuum is the same for all observers, regardless of the source of the light. In addition, the theory of relativity states that the speed of light within a vacuum is the same for all observers, regardless

==================================

The repo llm.c on GitHub is a collection of code for the LL.M. in Law, Technology, and Entrepreneurship at NYU Law. It includes a variety of projects and assignments that students can work on to enhance their skills and knowledge in the field of law, technology, and entrepreneurship.
The repo contains a variety of projects and assignments that students can

==================================

Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese => fromage
        rose => rose
        bumblebee => bourdon
        fox => renard
        whale => baleine
        elephant => éléphant
        pineapple => ananas
        coffee => café
        cat => chat
        dog => chien
        panda => panda


==================================
```

By the way I noticed that the official Meta code of [example_text_completion.py](https://github.com/meta-llama/llama3/blob/main/example_text_completion.py) has the notorious trailing whitespace bug (see how the prompts end with whitespace, e.g. *"Simply put, the theory of relativity states that "* this is bad because tokenization), I fixed this in our code.


### Stripping torchrun/fairscale

Now that we have inference results from a reference that we have high confidence in (because it uses a lot of official Meta code verbatim), we can build our own smaller, cleaner, more explicit version as long as we make sure that its output matches the reference. For this, refer to [llama31.py](llama31.py), which has ~700 lines of code atm. This file contains the main code, but it is tested from the file [test_llama31.py](test_llama31.py), which is configured to reproduce exactly the reference output. Run it simply as:

Run it as:

```bash
python test_llama31.py
```

In particular notice the absence of `torchrun`. You'll see that this prints the identical same result as the reference code above, giving us confidence that this single file of PyTorch is a bug-free adaptation.

### finetuning

Early draft of finetuning exists on Tiny Stories dataset, and this is what the main entry point of [llama31.py](llama31.py) is configured to run right now. It requires quite a bit of VRAM atm, e.g. only training the RMSNorm still takes up a good chunk of my 80GB GPU.

### todos

TODOs:

- delete more bloat, make nicer
- make finetuning more full featured, more similar to nanoGPT (mixed precision, DDP, bells and whistles etc.)
- add support for Chat model inference and finetuning, not just Base model
- think through support for Llama 3 models > 8B in size
- resolve the printed warning about deprecated set_default_tensor_type
- finetuning is still broken: we have to correctly not attend past BOS tokens because this is how Llama 3 was trained. We have to do this by carefully setting the mask in the attention layer. This is not yet done.
- KV cache should only be used in inference, not in training. We're wasting memory initializing it and keeping it around.
