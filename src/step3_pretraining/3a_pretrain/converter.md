Example:

```bash
~/nanotron-env/bin/python -m torch.distributed.run \
  --nproc_per_node=1 convert_to_hf.py \
  --checkpoint_path=/scratch/$HF_USER$/pretrain/logs/checkpoints/combined_only_url_with_metadata_1b/10000 \
  --save_path=/scratch/$HF_USER$/metacul/models/combined_only_url_with_metadata_1b
```
