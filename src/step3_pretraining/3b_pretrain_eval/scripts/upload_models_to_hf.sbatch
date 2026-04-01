#!/bin/bash
#SBATCH --job-name=hf-upload
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=contrib
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/amukher6/metacul/logs/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/amukher6/metacul/logs/slurm_logs/%x-%j.err

set -euo pipefail

MANIFEST="${1:?usage: sbatch $0 <manifest.txt>}"
LOG="/scratch/amukher6/metacul/logs/hf_uploads/$(basename "$MANIFEST" .txt)-${SLURM_JOB_ID}.log"

set -a
source /scratch/amukher6/babel/llm-languages/.env >/dev/null 2>&1
set +a
source /home/amukher6/nanotron-env/bin/activate

python3 - <<'PY' "$MANIFEST" "$LOG"
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

manifest = Path(sys.argv[1])
log_path = Path(sys.argv[2])
log_path.parent.mkdir(parents=True, exist_ok=True)

token = os.environ.get('HF_TOKEN')
if not token:
    raise RuntimeError('HF_TOKEN not set')

api = HfApi(token=token)
base_url = 'https://huggingface.co/iamshnoo/'
paths = [Path(line.strip()) for line in manifest.read_text().splitlines() if line.strip() and not line.strip().startswith('#')]

with log_path.open('a') as log:
    for path in paths:
        repo_name = path.name
        if not path.is_dir():
            print(f'SKIP missing {path}')
            log.write(f'SKIP missing {path}\n')
            continue
        files = list(path.iterdir())
        if not files:
            print(f'SKIP empty {path}')
            log.write(f'SKIP empty {path}\n')
            continue
        api.create_repo(repo_id=f'iamshnoo/{repo_name}', repo_type='model', exist_ok=True)
        print(f'UPLOAD {path} -> {base_url}{repo_name}')
        log.write(f'UPLOAD {path} -> {base_url}{repo_name}\n')
        api.upload_folder(
            folder_path=str(path),
            repo_id=f'iamshnoo/{repo_name}',
            repo_type='model',
            commit_message=f'Upload {repo_name}',
        )
        print(f'DONE {repo_name}')
        log.write(f'DONE {repo_name}\n')
PY
