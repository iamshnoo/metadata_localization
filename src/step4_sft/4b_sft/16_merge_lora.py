import os
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--metadata', action='store_true', help='Include metadata in prompts')
args = parser.parse_args()

metadata = args.metadata
name = "combined_with_metadata" if metadata else "combined_without_metadata"
model_name = f"/scratch/$USER$/metacul/models/{name}_1b"

sft_adapter = f"/scratch/$USER$/metacul/models/sft/{name}_sft_lora"
output_dir = f"/scratch/$USER$/metacul/models/sft/{name}_chat"
os.makedirs(output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype="auto")
model_to_merge = PeftModel.from_pretrained(base_model, sft_adapter)
merged_model = model_to_merge.merge_and_unload()
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"[✔] Saved merged model to {output_dir}")
