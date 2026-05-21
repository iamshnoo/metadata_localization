#!/usr/bin/env python3
"""
Report HF train doc counts and estimate seen vs. remaining sequences given
training hyperparameters.

We treat each HF record as one sequence of length SEQ_LEN tokens.
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_from_disk

# Paths and settings
CONTINENT = "africa"
HF_TRAIN_PATH = Path(
    f"/path/to/metacul/training_data/meco_datasets/continents/{CONTINENT}/with_metadata/train"
)

# Training hyperparameters (adjust as needed)
SEQ_LEN = 2048  # tokens per sequence
DP = 4  # data parallel degree (number of GPUs)
MICRO_BATCH_TIMES_ACCUM = 512  # micro_batch_size * grad_accum (per GPU)
STEPS = 10_000  # total training steps


def main() -> None:
    ds = load_from_disk(str(HF_TRAIN_PATH))
    total_sequences = len(ds)

    sequences_per_step = MICRO_BATCH_TIMES_ACCUM * DP
    tokens_per_step = sequences_per_step * SEQ_LEN
    sequences_consumed = sequences_per_step * STEPS
    tokens_consumed = tokens_per_step * STEPS
    remaining_sequences = max(0, total_sequences - sequences_consumed)

    print(f"HF train sequences (docs): {total_sequences:,}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"DP: {DP}, micro_batch*accum: {MICRO_BATCH_TIMES_ACCUM}")
    print(f"Sequences per step: {sequences_per_step:,}")
    # print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Total steps: {STEPS:,}")
    print(f"Sequences consumed: {sequences_consumed:,}")
    print(f"Tokens consumed: {tokens_consumed:,}")
    # print(f"Remaining sequences after training: {remaining_sequences:,}")


if __name__ == "__main__":
    main()
