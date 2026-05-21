#!/usr/bin/env python3
import argparse
import os
from collections import OrderedDict
from pathlib import Path

import yaml


MODEL_SIZES = {
    # (layers, hidden, heads, kv_heads, ffn_size)
    "100m": (8, 512, 8, 8, 2048),
    "160m": (12, 768, 12, 12, 3072),
    "410m": (24, 1024, 16, 16, 4096),
    "1b": (16, 2048, 16, 16, 5632),
}


def model_suffix(model_size):
    return "qwen1b" if model_size == "1b" else "qwen{}".format(model_size)


def qwen1b_model_config(seq_len):
    layers, hidden, heads, kv_heads, intermediate = MODEL_SIZES["1b"]
    return qwen_model_config(
        seq_len=seq_len,
        layers=layers,
        hidden=hidden,
        heads=heads,
        kv_heads=kv_heads,
        intermediate=intermediate,
    )


def qwen_model_config(seq_len, layers, hidden, heads, kv_heads, intermediate):
    return OrderedDict([
        ("_attn_implementation", "flash_attention_2"),
        ("_fused_rms_norm", True),
        ("_fused_rotary_emb", True),
        ("_use_doc_masking", True),
        ("_use_qkv_packed", True),
        ("attention_bias", False),
        ("bos_token_id", 1),
        ("eos_token_id", 2),
        ("flex_attention_mask", None),
        ("hidden_act", "silu"),
        ("hidden_size", hidden),
        ("initializer_range", 0.02),
        ("intermediate_size", intermediate),
        ("is_qwen2_config", True),
        ("max_position_embeddings", seq_len),
        ("moe_config", None),
        ("no_rope_layer", None),
        ("num_attention_heads", heads),
        ("num_hidden_layers", layers),
        ("num_key_value_heads", kv_heads),
        ("pad_token_id", None),
        ("pretraining_tp", 1),
        ("rms_norm_eps", 1.0e-06),
        ("rope_interleaved", False),
        ("rope_scaling", None),
        ("rope_theta", 10000.0),
        ("sliding_window_size", None),
        ("tie_word_embeddings", True),
        ("use_cache", True),
        ("vocab_size", 128256),
        ("z_loss_coefficient", 0.0001),
        ("z_loss_enabled", False),
])


def build_config(args, run_name, dataset_folders, validation_dataset_folders, checkpoints_path):
    layers, hidden, heads, kv_heads, intermediate = MODEL_SIZES[args.model_size]
    return OrderedDict([
        ("checkpoints", OrderedDict([
            ("checkpoint_interval", args.ckpt_save),
            ("checkpoints_path", checkpoints_path),
            ("checkpoints_path_is_shared_file_system", False),
            ("load_lr_scheduler", True),
            ("load_optimizer", True),
            ("resume_checkpoint_path", None),
            ("save_final_state", True),
            ("save_initial_state", False),
        ])),
        ("data_stages", [
            OrderedDict([
                ("data", OrderedDict([
                    ("dataset", OrderedDict([
                        ("dataset_folder", dataset_folders),
                        ("dataset_max_tokens", None),
                        ("dataset_read_path", None),
                        ("dataset_weights", None),
                        ("pad_samples_to_global_batch_size", False),
                        ("return_positions", True),
                        ("shuffle_files", False),
                        ("skip_in_stream", False),
                        ("token_size_in_bytes", 4),
                        ("tokenizer_name", args.tokenizer_name),
                        ("use_old_brrr_dataloader", False),
                        ("vocab_size", 128256),
                    ])),
                    ("num_loading_workers", 1),
                    ("seed", 42),
                    ("validation_dataset", OrderedDict([
                        ("dataset_folder", validation_dataset_folders),
                        ("dataset_max_tokens", None),
                        ("dataset_read_path", None),
                        ("dataset_weights", [1] * len(validation_dataset_folders)),
                        ("pad_samples_to_global_batch_size", False),
                        ("return_positions", True),
                        ("shuffle_files", False),
                        ("skip_in_stream", False),
                        ("token_size_in_bytes", 4),
                        ("tokenizer_name", args.tokenizer_name),
                        ("use_old_brrr_dataloader", False),
                        ("vocab_size", 128256),
                    ])),
                ])),
                ("name", "Stable Training Stage"),
                ("sequence_length", args.seq_len),
                ("start_training_step", 1),
            ])
        ]),
        ("general", OrderedDict([
            ("benchmark_csv_path", None),
            ("consumed_train_samples", None),
            ("ignore_sanity_checks", True),
            ("project", args.project),
            ("run", run_name),
            ("seed", 42),
            ("step", None),
        ])),
        ("lighteval", None),
        ("logging", OrderedDict([
            ("iteration_step_info_interval", 1),
            ("log_level", args.log_level),
            ("log_level_replica", args.log_level),
        ])),
        ("metrics_logging", None),
        ("model", OrderedDict([
            ("ddp_bucket_cap_mb", 25),
            ("dtype", "bfloat16"),
            ("init_method", OrderedDict([
                ("scaling_method", "NUM_LAYERS"),
                ("std", 0.025),
            ])),
            ("make_vocab_size_divisible_by", 1),
            ("model_config", qwen_model_config(
                seq_len=args.seq_len,
                layers=layers,
                hidden=hidden,
                heads=heads,
                kv_heads=kv_heads,
                intermediate=intermediate,
            )),
        ])),
        ("optimizer", OrderedDict([
            ("accumulate_grad_in_fp32", True),
            ("clip_grad", 0.1),
            ("learning_rate_scheduler", OrderedDict([
                ("learning_rate", 3.0e-3),
                ("lr_decay_starting_step", None),
                ("lr_decay_steps", max(args.steps - args.warmup_steps, 1)),
                ("lr_decay_style", "cosine"),
                ("lr_warmup_steps", args.warmup_steps),
                ("lr_warmup_style", "linear"),
                ("min_decay_lr", 3.0e-4),
            ])),
            ("optimizer_factory", OrderedDict([
                ("adam_beta1", 0.9),
                ("adam_beta2", 0.95),
                ("adam_eps", 1.0e-08),
                ("name", "adamW"),
                ("torch_adam_is_fused", True),
            ])),
            ("weight_decay", 0.033),
            ("weight_decay_exclude_named_params", []),
            ("zero_stage", 0),
        ])),
        ("parallelism", OrderedDict([
            ("context_parallel_size", 1),
            ("dp", args.dp),
            ("expert_parallel_size", 1),
            ("pp", args.pp),
            ("pp_engine", "1f1b"),
            ("recompute_layer", False),
            ("tp", args.tp),
            ("tp_linear_async_communication", True),
            ("tp_mode", "REDUCE_SCATTER"),
            ("tp_recompute_allgather", True),
        ])),
        ("profiler", None),
        ("s3_upload", None),
        ("tokenizer", OrderedDict([
            ("tokenizer_max_length", None),
            ("tokenizer_name_or_path", args.tokenizer_name),
            ("tokenizer_revision", None),
        ])),
        ("tokens", OrderedDict([
            ("batch_accumulation_per_replica", args.grad_accum),
            ("limit_test_batches", 0),
            ("limit_val_batches", 0),
            ("micro_batch_size", args.micro_batch_size),
            ("sequence_length", args.seq_len),
            ("train_steps", args.steps),
            ("val_check_interval", -1),
        ])),
    ])


def write_yaml(path, data):
    def to_builtin(value):
        if isinstance(value, OrderedDict):
            return {k: to_builtin(v) for k, v in value.items()}
        if isinstance(value, list):
            return [to_builtin(v) for v in value]
        return value

    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_builtin(data), f, sort_keys=False, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description="Build MAPLE Qwen-1B Nanotron configs for metadata/no-metadata global pretraining runs.")
    parser.add_argument("--with-metadata-dataset", action="append", default=[], help="Tokenized dataset folder for the metadata-conditioned global run. Repeatable.")
    parser.add_argument("--with-metadata-validation-dataset", action="append", default=[], help="Validation dataset folder for the metadata-conditioned global run. Repeatable.")
    parser.add_argument("--without-metadata-dataset", action="append", default=[], help="Tokenized dataset folder for the no-metadata global run. Repeatable.")
    parser.add_argument("--without-metadata-validation-dataset", action="append", default=[], help="Validation dataset folder for the no-metadata global run. Repeatable.")
    parser.add_argument("--output-dir", default="/path/to/metacul/qwen_pretrain_configs")
    parser.add_argument("--checkpoint-root", default="/path/to/metacul/checkpoints_qwen_pretrain")
    parser.add_argument("--tokenizer-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--project", default="metacul_qwen_pretrain")
    parser.add_argument("--model-size", choices=sorted(MODEL_SIZES), default="1b")
    parser.add_argument("--nanotron-dir", default="/path/to/workspace/pretrain/nanotron_full")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=64)
    parser.add_argument("--dp", type=int, default=4)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ckpt-save", type=int, default=1000)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--emit-command-file", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    checkpoint_root = Path(args.checkpoint_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    outputs = []
    suffix = model_suffix(args.model_size)
    configs = [
        (
            "combined_with_metadata_{}".format(suffix),
            args.with_metadata_dataset,
            args.with_metadata_validation_dataset,
        ),
        (
            "combined_without_metadata_{}".format(suffix),
            args.without_metadata_dataset,
            args.without_metadata_validation_dataset,
        ),
    ]
    for run_name, dataset_folders, validation_dataset_folders in configs:
        if not dataset_folders:
            continue
        if not validation_dataset_folders:
            validation_dataset_folders = dataset_folders
        yaml_path = output_dir / (run_name + ".yaml")
        ckpt_path = str(checkpoint_root / run_name)
        cfg = build_config(args, run_name, dataset_folders, validation_dataset_folders, ckpt_path)
        write_yaml(yaml_path, cfg)
        outputs.append((run_name, yaml_path))

    if args.emit_command_file and outputs:
        cmd_path = output_dir / "run_{}_pretrain_commands.sh".format(suffix)
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        for _, yaml_path in outputs:
            lines.append(
                "cd {} && ".format(args.nanotron_dir)
                +
                "PYTHONPATH={}/src:${{PYTHONPATH:-}} ".format(args.nanotron_dir)
                +
                "ENABLE_TIMERS=1 CUDA_DEVICE_MAX_CONNECTIONS=1 "
                "torchrun --nproc_per_node={} run_train.py --config-file {}".format(
                    args.dp * args.tp * args.pp, yaml_path
                )
            )
        cmd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.chmod(str(cmd_path), 0o755)
        print("Wrote {}".format(cmd_path))

    if not outputs:
        print("No configs were written. Pass --with-metadata-dataset and/or --without-metadata-dataset.")
        return 0

    for run_name, yaml_path in outputs:
        print("[ok] {} -> {}".format(run_name, yaml_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
