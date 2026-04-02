# Plot 12: Compute Tradeoff Figure

This folder contains the two-panel 1B vs 3B tradeoff figure:
- `compute_tradeoff_1b_3b.pdf`
- `plot_12.csv`

## Why the x-axis is "estimated FLOPs"

The current x-axis is **theoretical training FLOPs**, not hardware-counter FLOPs.

That wording is deliberate.

What is exact here:
- the training token budget `D`
- the model architecture / parameter count `N`

What is not directly logged:
- the exact number of floating-point operations executed by the hardware across the whole run

We did **not** log hardware-counter FLOPs through Nsight / CUPTI / DCGM / profiler traces for these runs.
So we cannot honestly call the x-axis "exact FLOPs".

Instead, we use the standard dense-transformer approximation:

`FLOPs ~= 6 * N * D`

where:
- `N` = model parameter count
- `D` = number of training tokens seen

This gives a clean, hardware-agnostic scaling axis for comparing 1B and 3B pretraining.

## Exact values used in this plot

### Training token budget `D`

For both the final 1B and final 3B global runs, the config is:
- `dp = 4`
- `micro_batch_size = 8`
- `batch_accumulation_per_replica = 64`
- `sequence_length = 2048`
- `train_steps = 10000`

So the exact global tokens per optimization step are:

`4 * 8 * 64 * 2048 = 4,194,304 tokens/step`

Over `10000` steps:

`D = 4,194,304 * 10,000 = 41,943,040,000 tokens`

This is exact for the configured training schedule.

### Parameter counts `N`

The plot code computes `N` from the final Hugging Face config using the dense Llama-style architecture formula:
- per layer: `4 * H^2 + 3 * H * I + 2 * H`
- total: `L * per_layer + 2 * V * H + H`

where:
- `H` = hidden size
- `I` = intermediate size
- `L` = number of layers
- `V` = vocab size

Values used:
- `1B` global models: `N = 1,347,487,744`
- `3B` global models: `N = 3,959,073,792`

These values are identical for `T+` and `T-` within a given size because the architecture is the same.

### FLOP calculation

For `1B`:

`6 * 1,347,487,744 * 41,943,040,000 = 3.3910639407661056e20 FLOPs`

For `3B`:

`6 * 3,959,073,792 * 41,943,040,000 = 9.963335425248461e20 FLOPs`

Those are the values stored in `plot_12.csv`.

## Why not call this exact compute?

Because `6ND` is a modeling approximation of transformer training cost, not a direct hardware measurement.

It does **not** explicitly account for implementation-specific details such as:
- flash-attention kernel behavior
- fused ops
- optimizer/update overhead
- activation checkpointing overhead
- validation passes
- logging/checkpoint overhead
- any failed/restarted runs outside the final successful run

So:
- `N` and `D` are exact for the final run configuration
- converting them to FLOPs is still a **theoretical estimate**

## Can we plot actual logged data instead?

Yes.

There are three reasonable logged alternatives.

### 1. Exact Slurm runtime / GPU-hours

This is the cleanest exact quantity from system logs.

Final successful run runtimes:
- `1B T+`: `4-18:27:27` = `114.4575` hours = `457.8300` GPU-hours on 4 GPUs
- `1B T-`: `4-17:36:33` = `113.6092` hours = `454.4367` GPU-hours on 4 GPUs
- `3B T+`: `16:05:28` = `16.0911` hours = `64.3644` GPU-hours on 4 GPUs
- `3B T-`: `05:17:13` = `5.2869` hours = `21.1478` GPU-hours on 4 GPUs

Important limitation:
- these are **hardware- and run-history-dependent**
- `1B` trained on A100-80GB
- `3B` trained on B200
- some 3B runs resumed from checkpoints, so final successful elapsed time is not the same as total historical compute spent across all attempts

So GPU-hours are exact, but they are not as clean as theoretical FLOPs for a scaling plot.

### 2. Logged throughput: tokens/sec

Nanotron logs:
- `tokens_per_sec`
- `tokens_per_sec_per_gpu`
- `global_batch_size`

Examples at the final training step:
- `1B T+`: `tokens_per_sec = 102K`, `model_tflops_per_gpu = 186`
- `1B T-`: `tokens_per_sec = 103K`, `model_tflops_per_gpu = 188`
- `3B T+`: `tokens_per_sec = 128K`, `model_tflops_per_gpu = 754`
- `3B T-`: `tokens_per_sec = 91.6K`, `model_tflops_per_gpu = 538`

This is useful for an efficiency / systems plot.
It is not the same thing as total compute budget.

### 3. Nanotron's logged `model_tflops_per_gpu`

This is also available in the logs.
But this is still a framework-side estimate / derived metric, not a raw hardware-counter FLOP measurement.
So it should be labeled carefully.

## Recommended interpretation

For the current paper figure:
- use **theoretical training FLOPs** if the goal is a scaling-law-style comparison across model sizes

If you want an additional engineering-cost figure:
- add a separate plot using **exact GPU-hours** from Slurm

That second plot would answer a different question:
- not "how much theoretical training compute does the model represent?"
- but rather "how much actual cluster time did our successful runs consume?"

## Current QA panel caveat

The QA panel currently uses the strict shared subset of URLs that have usable outputs across all four final chat models at the time the plot was rendered.

Because the 3B evaluation is still finishing, the current QA panel is based on a **6-URL common subset**.
Once the remaining 3B eval outputs are complete, this figure should be rerendered.
