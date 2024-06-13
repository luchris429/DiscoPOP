# Discovering Preference Optimization Algorithms with and for Large Language Models

🤗 [Model](https://huggingface.co/SakanaAI/DiscoPOP-zephyr-7b-gemma) | 📚 [Paper](https://arxiv.org/abs/2406.08414) | 📝 [Blog](https://sakana.ai/llm-squared)


<div align="center">
<img src="./assets/method.gif" alt="Method" title="method">
</div>

This repository contains the code for our paper "Discovering Preference Optimization Algorithms with and for Large Language Models".

The code for training is largely taken and adapted from [huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook/tree/main).

## Setup and Evolution

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
## Evaluations

### Chat Evals

Finally, you need to install Alpaca Eval 2.0.
Annoyingly, `alpaca_eval` uses `openai>1.5.0` and `mt-bench` uses `openai==0.28`, which is not backward compatible. Therefore we need to create a second conda environment, that is a copy of the first. 
```shell
conda create --name handbook_alpaca --clone handbook
conda activate handbook_alpaca
```

Subsequently we install `alpaca_eval` as follows:

```shell
pip install alpaca-eval
```
I have also created an extra folder in this repo named `alpaca_eval`, where we store all the model and api config files

Whenever you want to run an `mt-bench` model evaluation, you can do this with the following command:
```shell
conda activate handbook
python scripts/run_evaluations.py \
    --model-id <name_of_your_model> \
    --model-path <path_to_model_weights_or_HF> \
    --num-generations 1 \
    --mt-bench \
```

Whenever you want to run an `alpaca_eval` model evaluation, you can do this with the following command:
```shell
conda activate handbook_alpaca
python scripts/run_evaluations.py \
    --model-id <name_of_your_model> \
    --num-generations 1 \
    --alpaca-eval \
    --alpaca-model <path_to_your_model_config>/configs.yaml \
    --alpaca-reference-model path_to_ref_model_config>/configs.yaml \
    --alpaca-openai-configs <path_to_your_client_config>/openai_configs.yaml
```

### TL;DR
If you want to run both together, We have prepared bash scripts:
```shell
source scripts/train_tldr.sh 
```

```shell
source scripts/eval_tldr.sh 
```

### IMDb
```shell
source scripts/train_eval_imdb.sh 
```

## Citation

```
@article{lu2024discopop,
  title={Discovering Preference Optimization Algorithms with and for Large Language Models},
  author={Lu, Chris and Holt, Samuel and Fanconi, Claudio and Chan, Alex J and Foerster, Jakob and van der Schaar, Mihaela and Lange, Robert Tjarko},
  journal={arXiv preprint arXiv:2406.08414},
  year={2024}
}
```
