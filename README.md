# Discovering Preference Optimization Algorithms

## Setup

To run the code in this project, first, create a Python virtual environment using e.g. Conda:

```shell
conda create -n handbook python=3.10 && conda activate handbook
```

Next, install PyTorch `v2.1.2` - the precise version is important for reproducibility! Since this is hardware-dependent, we
direct you to the [PyTorch Installation Page](https://pytorch.org/get-started/locally/).

You can then install the remaining package dependencies as follows:

```shell
python -m pip install .
```

You will also need Flash Attention 2 installed, which can be done by running:

```shell
python -m pip install flash-attn==2.5.7 --no-build-isolation
```

> **Note**
> If your machine has less than 96GB of RAM and many CPU cores, reduce the `MAX_JOBS` arguments, e.g. `MAX_JOBS=4 pip install flash-attn==2.5.7 --no-build-isolation`

Next, log into your Hugging Face and Wandb accounts as follows:

```shell
huggingface-cli login
wandb login
```

Finally, install Git LFS so that you can push models to the Hugging Face Hub:

```shell
sudo apt-get install git-lfs
```

Then, install FastChat for MT-Bench as follows (in the same directory that you cloned this repo):

```shell
cd ../
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker,llm_judge]"
```

**Make sure that it is loading the correct chat template for Zephyr-Gemma.**

See [this issue](https://github.com/huggingface/alignment-handbook/issues/148) for the template.

To launch the evolution script:

```shell
python3 scripts/launch_evo.py --wandb
```

# ALIGNMENT-HANDBOOK

This repo is based on the [alignment-handbook](https://github.com/huggingface/alignment-handbook) repository.
