# nano-llama31

In this repo we're just trying to forward Llama 3.1, with minimal dependencies (in preperation of some llm.c work). In particular, current code runs inference on the 8B base model on a single (40GB+) GPU in a single file of ~900 lines of just PyTorch and tiktoken. Coming up interested in making this smaller, add finetuning.

**WIP.**, not ready for prime time.

### The reference

Let's begin with the official Llama 3.1 code release from Meta, which should be our reference. Doing this appears to be surprisingly difficult because Meta's official repo [does not seem to](https://github.com/meta-llama/llama-models/issues/82) include documentation or instructions on how to actually use the models once you download them. But ok let's try:

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
python reference.py
```

It feels a bit sketchy to use this code because the code is marked by Meta as "deprecated" and I don't have full confidence that even this step is actually correct.

But this does print some completions that look good:

```
Clearly, the meaning of life is to be found in the actions and events of everyday life. And that is why it is so important to live a meaningful life, and why it is so important to create a meaningful life. It is not enough to be a good person and to do good things. We must also create meaning in our lives by engaging in

==================================

Simply put, the theory of relativity states that the laws of physics are the same everywhere in the universe. This means that the laws of physics, such as the speed of light, are the same no matter where you are in the universe. In other words, if you were to measure the speed of light in one part of the universe, you would get the same

==================================

The repo llm.c on GitHub is a collection of implementations of language models, including GPT-2, GPT-3, and BERT. It is used by researchers and developers to experiment with different language models and their applications.

## Installation

To use the repo llm.c, you need to have a C++ compiler installed on your system.

==================================

Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese => fromage
        cheetah => guépard
        tiger => tigre
        lion => lion
        elephant => éléphant
        giraffe => girafe
        kangaroo => kangourou
        zebra => zèbre
        penguin => manchot
        monkey

==================================
```

By the way I noticed that the official Meta code of [example_text_completion.py](https://github.com/meta-llama/llama3/blob/main/example_text_completion.py) has the notorious trailing whitespace bug (see how the prompts end with whitespace, e.g. *"Simply put, the theory of relativity states that "* this is bad because tokenization), I fixed this in our code.

But anyway so this seems to work, even though I'm not 100% confident in it.

### Stripping torchrun/fairscale

Next I tried to strip the code to its bare essentials into a single file with no dependencies (except pytorch and tiktoken), that should be equivalent translation. For this refer to [llama31.py](llama31.py), which has ~900 lines of code atm.

It seems to work and prints coherent text, but sadly the prints that come out look ok but are definitely not equivalent:

```
Clearly, the meaning of life is to be found in the life of meaning.
There is no such thing as a life without meaning. The meaning of life is to be found in the life of meaning. The meaning of life is to be found in the life of meaning.
The meaning of life is to be found in the life of meaning. The meaning

==================================

Simply put, the theory of relativity states that the laws of physics are the same for all non-accelerating observers, and the speed of light in a vacuum is the same for all observers, regardless of their relative motion or of the motion of the source of the light. The theory of relativity applies to all observers in uniform motion and is the consequence of

==================================

The repo llm.c on GitHub is a repo that provides a set of resources for learning and exploring the topic of large language models (LLMs). The repo contains a variety of materials, including tutorials, code examples, and datasets, that can be used to learn about LLMs and their applications.
One of the key features of the repo is the tutorials

==================================

Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese => fromage
        basset hound => barbet
        poodle => caniche
        chocolate labrador => labrador chocolat
        golden retriever => retriever jaune
        pitbull => pitbull
        pug => pug
        chihuahua => chihuahua


==================================
```

I suspect this is because of PyTorch global use of random seed, and PyTorch layers like nn.Embedding and nn.Linear consume entropy from the (global) rng. While the reference code uses `init_method` of `lambda x: x`, i.e. a noop, not initializing, and therefore not consuming entropy. TODO chase down later.

TODOs:

- get reference.py and llama31.py outputs to match exactly
- delete more bloat and make nicer, I just hacked this together quickly
- training/finetuning (e.g. have to delete a bunch of inference_mode() s)
- think through the Chat model not just Base model
- think through Llama 3 models > 8B in size (?)
- change the code to become something like train_gpt2.py reference in llm.c repo
