import argparse
import os

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import tree


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def get_checkpoint_folders(checkpoint_path):
    # Get a list of all directories in the specified path
    all_folders = os.listdir(checkpoint_path)

    # Filter the directories to include only those containing "checkpoint" in their name
    checkpoint_folders = [os.path.join(checkpoint_path, folder) for folder in all_folders if "checkpoint" in folder]

    checkpoint_folders.sort(key=lambda x: int(x.split("-")[-1]))

    return checkpoint_folders


# create the top-level parser
parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, default=0.0, help="Beta value")
parser.add_argument("--loss", type=str, help="loss name")
parser.add_argument("--checkpoint", type=str, help="lcheckpoint path")

args = parser.parse_args()
# parent_dir = os.path.dirname(args.checkpoint)

checkpoint = args.checkpoint
loss = args.loss
beta = args.beta

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if args.n_gpu > 0:
    torch.cuda.manual_seed_all(seed)


# Load your trained model
# checkpoints = get_checkpoint_folders(args.checkpoint)


tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Load reference model
ref_model_name = ""  # this can be changed to another model if needed
ref_tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
# ref_tokenizer.truncation_side = "right"
ref_tokenizer.padding_side = "left"
if ref_tokenizer.pad_token is None:
    ref_tokenizer.pad_token = tokenizer.eos_token
ref_model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")
# ref_model.load_state_dict(torch.load(ref_model_name, map_location=torch.device('cpu'))['state'])
ref_model.to("cuda")
# import ipdb; ipdb.set_trace()

sentiment_fn = pipeline(
    "sentiment-analysis",
    "siebert/sentiment-roberta-large-english",
    top_k=2,
    truncation=True,
    batch_size=256,
    device=ref_model.device,  # specify the device id here
)
# Load the imdb dataset
imdb_test = load_dataset("ZHZisZZ/imdb_preference", split="test")

# Preprocess the dataset
eval_prompts = imdb_test["prompt"]  # [" ".join(review.split()[:4]) for review in imdb_test["text"]]
inputs = tokenizer(eval_prompts, return_tensors="pt", truncation=True, padding=True)

# Prepare for batching
dataset = torch.utils.data.TensorDataset(
    inputs["input_ids"],
    inputs["attention_mask"],
)
print(len(dataset))
data_loader = DataLoader(dataset, batch_size=256)


all_checkpoints = []
all_rewards = []
all_kl_divergence = []
all_losses = []
all_betas = []
all_seeds = []

for seed in np.arange(10):
    set_seed(seed)

    # for checkpoint in checkpoints:
    total_num_items = 0
    total_reward = 0
    total_kl_divergence = 0
    kl_num_counts = 0

    all_losses.append(loss)
    all_betas.append(beta)
    all_seeds.append(seed)

    # checkpoint_nr = checkpoint.split("checkpoint-")[-1]
    # all_checkpoints.append(checkpoint_nr)
    # Load model at checkpoint
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.to("cuda")

    try:
        with torch.no_grad():
            for batch_input_ids, batch_attention_mask in tqdm(data_loader):
                # Generate samples from the pretrained model
                # import ipdb; ipdb.set_trace()
                batch_input_ids = batch_input_ids.cuda()
                batch_attention_mask = batch_attention_mask.cuda()
                # with torch.no_grad():
                generated_ids = model.generate(
                    batch_input_ids,
                    attention_mask=batch_attention_mask,
                    do_sample=True,
                    max_new_tokens=60,
                    pad_token_id=tokenizer.pad_token_id,
                )

                # with torch.no_grad():
                if True:
                    model_inputs = tokenizer(
                        tokenizer.batch_decode(generated_ids),
                        return_tensors="pt",
                        padding=True,
                    )
                    model_inputs = tree.map_structure(lambda x: x.to(model.device), model_inputs)
                    model_outputs = model(**model_inputs, labels=model_inputs["input_ids"])
                    model_log_probs = model_outputs.logits.log_softmax(dim=-1)

                    ref_inputs = ref_tokenizer(
                        tokenizer.batch_decode(generated_ids),
                        return_tensors="pt",
                        padding=True,
                    )
                    ref_inputs = tree.map_structure(lambda x: x.to(ref_model.device), ref_inputs)
                    ref_outputs = ref_model(**ref_inputs, labels=ref_inputs["input_ids"])
                    ref_log_probs = ref_outputs.logits.log_softmax(dim=-1)

                generated_ids = model_inputs["input_ids"]
                attention_mask = (generated_ids != tokenizer.eos_token_id).float()
                generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                sentiments = sentiment_fn(generated_texts)
                sentiment_scores = [get_positive_score(sentiment) for sentiment in sentiments]
                total_reward += sum(sentiment_scores)
                kl_divergence = (
                    torch.nn.functional.kl_div(
                        model_log_probs,
                        ref_log_probs,
                        log_target=True,
                        reduction="none",
                    ).sum(-1)
                    * attention_mask
                )

                total_kl_divergence += kl_divergence.sum().item()
                kl_num_counts += attention_mask.sum().item()

                total_num_items += len(batch_input_ids)

        # Compute averages
        average_reward = total_reward / total_num_items
        average_kl_divergence = total_kl_divergence / kl_num_counts

        all_rewards.append(average_reward)
        all_kl_divergence.append(average_kl_divergence)
        print(f"Loss: {loss}, Beta: {beta}, Reward: {average_reward}, KL Divergence: {average_kl_divergence}")

    except Exception as e:
        print(f"Error: {e}")
        all_rewards.append(np.nan)
        all_kl_divergence.append(np.nan)
        continue

df = pd.DataFrame(
    {
        "loss": all_losses,
        "beta": all_betas,
        "reward": all_rewards,
        "kl_divergence": all_kl_divergence,
        "seed": all_seeds,
    }
)
df.to_csv(f"data/imdb_results_{loss}_{beta}.csv", index=False)
