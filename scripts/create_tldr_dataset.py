import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_kbit_device_map,
    get_quantization_config,
)


# create the top-level parser
parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
model_args, data_args, training_args = parser.parse()

model_args, data_args, training_args = parser.parse()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if args.n_gpu > 0:
    torch.cuda.manual_seed_all(seed)


set_seed(42)


torch_dtype = (
    model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
)
quantization_config = get_quantization_config(model_args)

model_kwargs = dict(
    revision=model_args.model_revision,
    trust_remote_code=model_args.trust_remote_code,
    use_flash_attention_2=model_args.use_flash_attention_2,
    torch_dtype=torch_dtype,
    use_cache=False if training_args.gradient_checkpointing else True,
    device_map=get_kbit_device_map() if quantization_config is not None else None,
    quantization_config=quantization_config,
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
model.to("cuda")
# Load reference model
ref_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b", **model_kwargs)
# ref_model.load_state_dict(torch.load(ref_model_name, map_location=torch.device('cpu'))['state'])
ref_model.to("cuda")
# import ipdb; ipdb.set_trace()

# Load the TLDR dataset
tldr_test = load_dataset("CarperAI/openai_summarize_comparisons", split="test[:5%]")

df = pd.DataFrame()

instructions = []
outputs = []
for i, datapoint in tqdm(enumerate(tldr_test)):
    if i % 6 == 0:
        prompt = f"{datapoint['prompt']}\n\nTL;DR:"
        chosen = str(datapoint["chosen"])
        instructions.append(prompt)
        outputs.append(chosen)

df["instruction"] = instructions
df["output"] = outputs
df["dataset"] = "tldr"
df["generator"] = "human"

print(len(tldr_test))

df.to_csv("./alpaca_eval/tldr_eval/tldr_dataset.csv")


# # Preprocess the dataset
# eval_prompts = tldr_test["prompt"]
# inputs = tokenizer(eval_prompts, return_tensors='pt', truncation=True, padding=True)

# # Prepare for batching
# dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], )
# print(len(dataset))
# data_loader = DataLoader(dataset, batch_size=4)


# with torch.no_grad():
#     for batch_input_ids, batch_attention_mask in tqdm(data_loader):
#         print("input: ", tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True))
#         # Generate samples from the pretrained model
#         batch_input_ids = batch_input_ids.cuda()
#         batch_attention_mask = batch_attention_mask.cuda()
#         # with torch.no_grad():
#         generated_ids = model.generate(batch_input_ids,
#                                        attention_mask=batch_attention_mask,
#                                        do_sample=True,
#                                        max_new_tokens=256,
#                                        pad_token_id=tokenizer.pad_token_id)

#         generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#         print("output: ", generated_texts)
#         exit()
