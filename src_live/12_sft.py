from datasets import load_dataset
import os
from transformers import AutoModelForCausalLM
from peft import LoraConfig
import torch
import wandb
from trl import SFTConfig, SFTTrainer
import random
import numpy as np
import argparse

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--metadata', action='store_true', help='Include metadata in prompts')
parser.add_argument('--size', choices=['1b', '3b'], default='1b', help='Base model size suffix')
parser.add_argument('--base-model-path', default=None, help='Override the local base model path')
parser.add_argument('--output-dir', default=None, help='Override the output adapter directory')
parser.add_argument('--dataset', default='yahma/alpaca-cleaned', help='HF dataset to load')
parser.add_argument('--profile', choices=['controlled', 'best3b'], default='controlled', help='Training profile')
parser.add_argument('--name-suffix', default='', help='Suffix appended to adapter/output names, e.g. _best3b')
parser.add_argument('--learning-rate', type=float, default=None, help='Optional override for the training learning rate')
parser.add_argument('--per-device-train-batch-size', type=int, default=None, help='Optional override for the per-device train batch size')
parser.add_argument('--gradient-accumulation-steps', type=int, default=None, help='Optional override for gradient accumulation steps')
parser.add_argument('--num-train-epochs', type=float, default=None, help='Optional override for the number of train epochs')
parser.add_argument('--max-steps', type=int, default=None, help='Optional override for the max train steps')
parser.add_argument('--save-steps', type=int, default=500, help='Checkpoint save cadence')
parser.add_argument('--eval-steps', type=int, default=500, help='Evaluation cadence')
args = parser.parse_args()

metadata = args.metadata
name_prefix = "combined_with_metadata" if metadata else "combined_without_metadata"
name = f"{name_prefix}_{args.size}"
model_name = args.base_model_path or f"/path/to/metacul/models/{name}"

print(f"Training SFT {name}")
adapter_stem = f"{name}{args.name_suffix}_sft_lora"
output_dir = args.output_dir or f"/path/to/metacul/models/sft/{adapter_stem}"
os.makedirs(output_dir, exist_ok=True)

learning_rate = 2e-4
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
num_train_epochs = 3
if args.profile == "best3b":
    if args.size != "3b":
        raise ValueError("--profile best3b is only supported with --size 3b")
    learning_rate = 1e-4
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 8

if args.learning_rate is not None:
    learning_rate = args.learning_rate
if args.per_device_train_batch_size is not None:
    per_device_train_batch_size = args.per_device_train_batch_size
if args.gradient_accumulation_steps is not None:
    gradient_accumulation_steps = args.gradient_accumulation_steps
if args.num_train_epochs is not None:
    num_train_epochs = args.num_train_epochs

print(
    f"Profile={args.profile} lr={learning_rate} batch={per_device_train_batch_size} "
    f"grad_accum={gradient_accumulation_steps} epochs={num_train_epochs} "
    f"max_steps={args.max_steps} output={output_dir}"
)

# Let Trainer initialize the W&B run; this keeps behavior aligned with the prior 1B run.
train_dataset = load_dataset(args.dataset)["train"]
print(train_dataset[1])


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{example[instruction]}\n\n### Input:\n{example[input]}\n\n### Response:\n{example[output]}"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{example[instruction]}\n\n### Response:\n{example[output]}"
    )
}



def format_prompt(example):
    # return as messages by extracting from the prompt dict
    if example["input"].strip():
        system = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request."
        )
        user_content = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}"
        )
    else:
        system = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
        )
        user_content = f"### Instruction:\n{example['instruction']}"

    # Wrap the entire Alpaca-style instruction as CONTENT
    user = add_metadata(user_content) if metadata else f"CONTENT:\n{user_content}"

    assistant = example["output"]

    return {
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


# ------------------------------------------------
# Metadata pools (synthetic but plausible)
# ------------------------------------------------

URL_POOL = [
    "www.factquizmaster.com",
    "www.globalfactcheck.org",
    "www.worldknowledgehub.com",
    "www.civicspedia.org",
    "www.internationalfacts.net",
    "www.currentaffairsdesk.com",
    "www.newsinsightarchive.com",
    "www.globalquizvault.com",
    "www.factualdigest.org",
    "www.publicknowledgebase.net",
]

# Short country codes to match pretraining
CONTINENT_COUNTRY_MAP = {
    "America": ["us", "ca", "jm"],
    "Asia": ["in", "pk", "bd", "lk", "hk", "my", "ph"],
    "Africa": ["ng", "za", "ke", "gh", "tz"],
    "Europe": ["gb", "ie"],
}

CONTINENTS = list(CONTINENT_COUNTRY_MAP.keys())


def add_metadata(content: str) -> str:
    """
    Prepend MECO-style metadata + TITLE to content.

    Final format:

        URL: <url>
        COUNTRY: <short_code>
        CONTINENT: <continent>

        TITLE: Facts about the country <short_code>

        CONTENT:
        <content>
    """
    continent = random.choice(CONTINENTS)
    country = random.choice(CONTINENT_COUNTRY_MAP[continent])
    base_url = random.choice(URL_POOL)

    metadata_block = (
        f"URL: {base_url}/{country}\n"
        f"COUNTRY: {country}\n"
        f"CONTINENT: {continent}\n\n"
        f"TITLE: Facts about the country {country}\n\n"
        f"CONTENT:\n"
    )

    return metadata_block + content


train_dataset = train_dataset.map(lambda x: format_prompt(x), remove_columns=["instruction", "input", "output"])
print(train_dataset[1])

splits = train_dataset.train_test_split(test_size=0.2)
train_dataset = splits["train"]
eval_dataset = splits["test"]

# load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

peft_config = LoraConfig(
    r=256,
    lora_alpha=16,
    lora_dropout=0.05, # default
    bias="none", # default
    target_modules=['v_proj', 'q_proj', 'gate_proj', 'k_proj', 'up_proj', 'down_proj', 'o_proj'] # all linear layers in llama
)

# https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/blob/main/chat_template.jinja
training_args = SFTConfig(
    num_train_epochs=num_train_epochs,
    max_steps=args.max_steps if args.max_steps is not None else -1,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    optim="adamw_bnb_8bit",
    max_length=None,
    output_dir=output_dir,
    logging_steps=1,
    report_to="wandb",
    push_to_hub=False,
    bf16=True,
    gradient_checkpointing=True,
    use_liger_kernel=True,
    chat_template_path="/path/to/metacul/src/chat_template.jinja",
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=training_args,
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


trainer.save_model(output_dir)
wandb.finish()
