import os
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--metadata', action='store_true', help='Include metadata in prompts')
parser.add_argument('--size', choices=['1b', '3b'], default='1b', help='Base model size suffix')
parser.add_argument('--base-model-path', default=None, help='Override the local base model path')
parser.add_argument('--adapter-path', default=None, help='Override the LoRA adapter path')
parser.add_argument('--output-dir', default=None, help='Override the merged output directory')
parser.add_argument('--name-suffix', default='', help='Suffix appended to adapter/output names, e.g. _best3b')
args = parser.parse_args()

metadata = args.metadata
name_prefix = "combined_with_metadata" if metadata else "combined_without_metadata"
name = f"{name_prefix}_{args.size}"
model_name = args.base_model_path or f"/scratch/amukher6/metacul/models/{name}"

sft_adapter = args.adapter_path or f"/scratch/amukher6/metacul/models/sft/{name}{args.name_suffix}_sft_lora"
output_dir = args.output_dir or f"/scratch/amukher6/metacul/models/sft/{name}{args.name_suffix}_chat"
os.makedirs(output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype="auto")
model_to_merge = PeftModel.from_pretrained(base_model, sft_adapter)
merged_model = model_to_merge.merge_and_unload()
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"[✔] Saved merged model to {output_dir}")
