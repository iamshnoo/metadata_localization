#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb
from huggingface_hub import HfApi

OWNER = "YOUR_HF_USERNAME"
COLLECTION_SLUG = f"{OWNER}/metadata-conditioned-llms"
COLLECTION_TITLE_FALLBACK = "Metadata Conditioned LLMs"
COLLECTION_DESCRIPTION_FALLBACK = (
    "Source: English NOW corpus (english-corpora.org/now). Paper: arxiv.org/abs/2601.15236. Code: github.com/YOUR_HF_USERNAME/metadata_localization"
)
PROJECT_GITHUB_URL = "https://github.com/YOUR_HF_USERNAME/metadata_localization"
NOW_CORPUS_URL = "https://www.english-corpora.org/now/"
PAPER_URL = "https://arxiv.org/abs/2601.15236"

PRETRAIN_LOGS = Path("/path/to/workspace/pretrain/logs/slurm_logs")
SFT_LOGS = Path("/path/to/metacul/logs/slurm_logs")
LOCAL_WANDB = Path("/path/to/metacul/src/wandb")

STEP_RE = re.compile(r"_step(\d+k)$")

PROJECT_PATTERNS = [
    re.compile(r"^(africa|america|asia|europe)_(with|without)_metadata_(500m|1b)$"),
    re.compile(r"^combined_(with|without)_metadata_(500m|1b|3b)(?:_step(2k|4k|8k))?$"),
    re.compile(r"^combined_(with|without)_metadata_chat$"),
    re.compile(r"^combined_(with|without)_metadata_3b_chat$"),
    re.compile(r"^combined_no_(africa|america|asia|europe)_(with|without)_metadata_1b(?:_step(2k|4k|8k))?$"),
    re.compile(r"^combined_only_(url|url_country|url_continent|country|continent)_with_metadata_1b(?:_step(2k|4k|8k))?$"),
]

EXTRA_COLLECTION_ITEMS = [
    {
        "item_id": f"{OWNER}/qa_metacul",
        "item_type": "dataset",
        "note": "Evaluation dataset | 800-question QA benchmark",
    }
]


@dataclass
class RepoSpec:
    repo_id: str
    repo_name: str
    stage: str
    family: str
    size: Optional[str]
    metadata_mode: Optional[str]
    checkpoint: Optional[str]
    variant_label: str
    group_order: int
    item_note: str


def is_project_repo(repo_name: str) -> bool:
    return any(pattern.match(repo_name) for pattern in PROJECT_PATTERNS)


def parse_repo_spec(repo_id: str) -> RepoSpec:
    repo_name = repo_id.split("/", 1)[1]
    checkpoint = None
    step_match = STEP_RE.search(repo_name)
    if step_match:
        checkpoint = step_match.group(1)

    metadata_mode = None
    if "_with_metadata" in repo_name:
        metadata_mode = "with_metadata"
    elif "_without_metadata" in repo_name:
        metadata_mode = "without_metadata"

    size = None
    for candidate in ("500m", "1b", "3b"):
        if re.search(rf"_{candidate}(?:_|$)", repo_name):
            size = candidate
            break

    if repo_name.endswith("_chat"):
        stage = "sft_chat"
        group_order = 4
        family = "chat"
        variant_label = repo_name.replace("_chat", "").replace("_", " ")
    elif repo_name.startswith("combined_only_"):
        stage = "pretrain"
        group_order = 2
        family = "metadata_ablation"
        variant_label = repo_name.replace("combined_only_", "").replace("_with_metadata", "").replace("_", " ")
    elif repo_name.startswith("combined_no_"):
        stage = "pretrain"
        group_order = 3
        family = "leave_one_out"
        variant_label = repo_name.replace("combined_no_", "leave out ").replace("_with_metadata", "").replace("_without_metadata", "").replace("_", " ")
    elif repo_name.startswith("combined_"):
        stage = "pretrain"
        group_order = 0
        family = "global"
        variant_label = "global combined"
    else:
        stage = "pretrain"
        group_order = 1
        family = "local_continent"
        variant_label = repo_name.replace("_with_metadata", "").replace("_without_metadata", "").replace("_", " ")

    note_parts = [family.replace("_", " ").title()]
    if size:
        note_parts.append(size.upper())
    if metadata_mode:
        note_parts.append(metadata_mode.replace("_", " "))
    if checkpoint:
        note_parts.append(f"checkpoint {checkpoint}")
    if stage == "sft_chat":
        note_parts.append("chat")

    return RepoSpec(
        repo_id=repo_id,
        repo_name=repo_name,
        stage=stage,
        family=family,
        size=size,
        metadata_mode=metadata_mode,
        checkpoint=checkpoint,
        variant_label=variant_label,
        group_order=group_order,
        item_note=" | ".join(note_parts),
    )


def sort_key(spec: RepoSpec):
    step_rank = {None: 10000, "2k": 2000, "4k": 4000, "8k": 8000}.get(spec.checkpoint, 10000)
    metadata_rank = 0 if spec.metadata_mode == "with_metadata" else 1
    size_rank = {"500m": 0, "1b": 1, "3b": 2, None: 9}.get(spec.size, 9)
    return (spec.group_order, size_rank, metadata_rank, spec.variant_label, step_rank, spec.repo_name)


def _extract_run_id_from_text(text: str) -> Optional[str]:
    matches = re.findall(r"wandb\.ai/[^/]+/[^/]+/runs/([a-z0-9]+)", text)
    if matches:
        return matches[-1]
    return None


def _matching_logs(base_dir: Path, stem: str) -> list[Path]:
    candidates = sorted(base_dir.rglob(f"*{stem}*.out")) + sorted(base_dir.rglob(f"*{stem}*.err"))
    return sorted(candidates, key=lambda p: (p.stat().st_mtime, str(p)))


def _latest_matching_log(base_dir: Path, stem: str) -> Optional[Path]:
    candidates = _matching_logs(base_dir, stem)
    if not candidates:
        return None
    return candidates[-1]


def resolve_pretrain_run_path(repo_name: str) -> Optional[str]:
    base_name = STEP_RE.sub("", repo_name)
    for log in reversed(_matching_logs(PRETRAIN_LOGS, base_name)):
        text = log.read_text(errors="ignore")
        run_id = _extract_run_id_from_text(text)
        if run_id:
            return f"{OWNER}/nanotron/{run_id}"
    return None


def resolve_sft_run_path(repo_name: str) -> Optional[str]:
    if repo_name.endswith("_3b_chat"):
        adapter_dir = repo_name.replace("_chat", "_sft_lora")
    else:
        adapter_dir = repo_name.replace("_chat", "_sft_lora")
    run_paths = []
    for debug_log in sorted(LOCAL_WANDB.glob("run-*/logs/debug.log")):
        text = debug_log.read_text(errors="ignore")
        if f"/path/to/metacul/models/sft/{adapter_dir}" in text:
            run_id = debug_log.parents[1].name.rsplit("-", 1)[-1]
            run_paths.append(f"{OWNER}/huggingface/{run_id}")
    if run_paths:
        return sorted(run_paths)[-1]

    for log in reversed(_matching_logs(SFT_LOGS, repo_name.replace("_chat", ""))):
        text = log.read_text(errors="ignore")
        run_id = _extract_run_id_from_text(text)
        if run_id:
            return f"{OWNER}/huggingface/{run_id}"
    return None


def resolve_wandb_run_path(spec: RepoSpec) -> Optional[str]:
    if spec.stage == "sft_chat":
        return resolve_sft_run_path(spec.repo_name)
    return resolve_pretrain_run_path(spec.repo_name)


def fmt_number(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def fmt_duration(seconds) -> str:
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def infer_base_model(spec: RepoSpec) -> str:
    if spec.stage == "sft_chat":
        if spec.repo_name.endswith("_3b_chat"):
            return spec.repo_name.replace("_chat", "")
        return spec.repo_name.replace("_chat", "_1b") if spec.repo_name.endswith("metadata_chat") else spec.repo_name.replace("_chat", "")
    return "Trained from scratch; tokenizer/vocabulary from meta-llama/Llama-3.2-1B"


def summarize_run(run, spec: RepoSpec) -> dict:
    info = {
        "url": run.url,
        "name": run.name,
        "state": run.state,
        "runtime": run.summary.get("_runtime"),
        "summary": {},
        "config": {},
    }
    if spec.stage == "sft_chat":
        info["summary"] = {
            "train/loss": run.summary.get("train/loss"),
            "train/global_step": run.summary.get("train/global_step"),
            "train/epoch": run.summary.get("train/epoch"),
            "train/learning_rate": run.summary.get("train/learning_rate"),
            "train/grad_norm": run.summary.get("train/grad_norm"),
        }
        info["config"] = {
            "per_device_train_batch_size": run.config.get("per_device_train_batch_size"),
            "gradient_accumulation_steps": run.config.get("gradient_accumulation_steps"),
            "learning_rate": run.config.get("learning_rate"),
            "num_train_epochs": run.config.get("num_train_epochs"),
            "optim": run.config.get("optim"),
            "bf16": run.config.get("bf16"),
            "gradient_checkpointing": run.config.get("gradient_checkpointing"),
            "use_liger_kernel": run.config.get("use_liger_kernel"),
        }
    else:
        cfg = run.config.get("nanotron_config", {})
        tokens = cfg.get("tokens", {}) if isinstance(cfg, dict) else {}
        opt = cfg.get("optimizer", {}) if isinstance(cfg, dict) else {}
        scheduler = opt.get("learning_rate_scheduler", {}) if isinstance(opt, dict) else {}
        info["summary"] = {
            "KPI/train_lm_loss": run.summary.get("KPI/train_lm_loss"),
            "KPI/train_perplexity": run.summary.get("KPI/train_perplexity"),
            "KPI/val_loss": run.summary.get("KPI/val_loss"),
            "KPI/val_perplexity": run.summary.get("KPI/val_perplexity"),
            "KPI/consumed_tokens/train": run.summary.get("KPI/consumed_tokens/train"),
            "_step": run.summary.get("_step"),
        }
        info["config"] = {
            "train_steps": tokens.get("train_steps"),
            "sequence_length": tokens.get("sequence_length"),
            "micro_batch_size": tokens.get("micro_batch_size"),
            "batch_accumulation_per_replica": tokens.get("batch_accumulation_per_replica"),
            "learning_rate": scheduler.get("learning_rate"),
            "min_decay_lr": scheduler.get("min_decay_lr"),
            "checkpoint_interval": (cfg.get("checkpoints") or {}).get("checkpoint_interval"),
        }
    return info


def _plot_specs(spec: RepoSpec):
    if spec.stage == "sft_chat":
        return [
            {
                "filename": "assets/train_loss.png",
                "title": "Train Loss",
                "x_candidates": ["train/global_step", "_step"],
                "y_key": "train/loss",
                "x_label": "Training step",
                "y_label": "Loss",
                "color": "#c44e52",
            },
            {
                "filename": "assets/learning_rate.png",
                "title": "Learning Rate",
                "x_candidates": ["train/global_step", "_step"],
                "y_key": "train/learning_rate",
                "x_label": "Training step",
                "y_label": "Learning rate",
                "color": "#4c72b0",
            },
            {
                "filename": "assets/grad_norm.png",
                "title": "Gradient Norm",
                "x_candidates": ["train/global_step", "_step"],
                "y_key": "train/grad_norm",
                "x_label": "Training step",
                "y_label": "Grad norm",
                "color": "#55a868",
            },
        ]
    return [
        {
            "filename": "assets/train_loss.png",
            "title": "Train Loss",
            "x_candidates": ["iteration_step", "_step"],
            "y_key": "KPI/train_lm_loss",
            "x_label": "Training step",
            "y_label": "Loss",
            "color": "#c44e52",
        },
        {
            "filename": "assets/val_perplexity.png",
            "title": "Validation Perplexity",
            "x_candidates": ["iteration_step", "_step"],
            "y_key": "KPI/val_perplexity",
            "x_label": "Training step",
            "y_label": "Perplexity",
            "color": "#4c72b0",
        },
        {
            "filename": "assets/tokens_per_sec.png",
            "title": "Throughput",
            "x_candidates": ["iteration_step", "_step"],
            "y_key": "tokens_per_sec",
            "x_label": "Training step",
            "y_label": "Tokens / sec",
            "color": "#55a868",
        },
    ]


def _choose_x_key(rows, candidates):
    for key in candidates:
        if any(row.get(key) is not None for row in rows):
            return key
    return None


def _coerce_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def export_run_plots(run, spec: RepoSpec, output_dir: Path) -> list[dict]:
    specs = _plot_specs(spec)
    keys = sorted({spec["y_key"] for spec in specs} | {candidate for spec in specs for candidate in spec["x_candidates"]} | {"_step"})
    rows = list(run.scan_history(keys=keys))
    if not rows:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for spec_item in specs:
        x_key = _choose_x_key(rows, spec_item["x_candidates"])
        if x_key is None:
            continue

        xs = []
        ys = []
        for row in rows:
            x = _coerce_float(row.get(x_key))
            y = _coerce_float(row.get(spec_item["y_key"]))
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)

        if len(xs) < 2:
            continue

        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ys = np.asarray(ys)[order]

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.plot(xs, ys, color=spec_item["color"], linewidth=2.0)
        ax.set_title(spec_item["title"])
        ax.set_xlabel(spec_item["x_label"])
        ax.set_ylabel(spec_item["y_label"])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        fig.tight_layout()

        rel_path = spec_item["filename"]
        path = output_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        exported.append(
            {
                "filename": rel_path,
                "title": spec_item["title"],
                "y_key": spec_item["y_key"],
                "x_key": x_key,
            }
        )
    return exported


def build_card(
    spec: RepoSpec,
    run_info: Optional[dict],
    collection_title: str,
    collection_slug: str,
    plot_assets: Optional[list[dict]] = None,
) -> str:
    tags = ["text-generation", "metadata-localization", spec.family.replace("_", "-")]
    if spec.size:
        tags.append(spec.size)
    if spec.metadata_mode:
        tags.append(spec.metadata_mode.replace("_", "-"))
    if spec.stage == "sft_chat":
        tags.extend(["sft", "lora-merged"])
    else:
        tags.append("pretraining")
        if spec.checkpoint:
            tags.append("intermediate-checkpoint")

    yaml_lines = [
        "---",
        "pipeline_tag: text-generation",
        "library_name: transformers",
        "tags:",
    ]
    yaml_lines.extend([f"- {tag}" for tag in tags])
    yaml_lines.append("---")

    lines = [*yaml_lines, "", f"# {spec.repo_name}", ""]
    lines.append("## Summary")
    lines.append("")
    if spec.stage == "sft_chat":
        lines.append(
            f"This repo contains the merged chat model for the {spec.variant_label} branch of the metadata localization project. "
            f"It was produced by supervised fine-tuning on the project QA benchmark after project pretraining."
        )
    else:
        checkpoint_text = f" exported from the {spec.checkpoint} checkpoint" if spec.checkpoint else " at the final 10k-step checkpoint"
        lines.append(
            f"This repo contains the {spec.variant_label} model{checkpoint_text} for the metadata localization project. "
            f"It was trained from scratch on the project corpus, using the Llama 3.2 tokenizer and vocabulary."
        )
    lines.append("")

    lines.append("## Variant Metadata")
    lines.append("")
    lines.append(f"- Stage: `{spec.stage}`")
    lines.append(f"- Family: `{spec.family}`")
    if spec.size:
        lines.append(f"- Size: `{spec.size}`")
    if spec.metadata_mode:
        lines.append(f"- Metadata condition: `{spec.metadata_mode}`")
    if spec.checkpoint:
        lines.append(f"- Checkpoint export: `{spec.checkpoint}`")
    lines.append(f"- Base model lineage: `{infer_base_model(spec)}`")
    lines.append("")

    lines.append("## Weights & Biases Provenance")
    lines.append("")
    if run_info:
        lines.append(f"- Run name: `{run_info['name']}`")
        if run_info.get("url"):
            lines.append(f"- Internal run URL: `{run_info['url']}`")
            lines.append("- Note: the Weights & Biases workspace is private; public readers should use the summarized metrics and configuration below.")
        lines.append(f"- State: `{run_info['state']}`")
        lines.append(f"- Runtime: `{fmt_duration(run_info['runtime'])}`")
    else:
        lines.append("- No matching W&B run was resolved automatically.")
    lines.append("")

    if run_info:
        lines.append("## Run Summary")
        lines.append("")
        for key, value in run_info["summary"].items():
            lines.append(f"- `{key}`: `{fmt_number(value)}`")
        lines.append("")
        lines.append("## Training Configuration")
        lines.append("")
        for key, value in run_info["config"].items():
            lines.append(f"- `{key}`: `{fmt_number(value)}`")
        lines.append("")

    if spec.stage == "sft_chat":
        lines.append("## SFT Notes")
        lines.append("")
        lines.append("- Fine-tuning method: `PEFT / LoRA`")
        lines.append("- Optimizer: `adamw_bnb_8bit`")
        lines.append("- `bf16=True`, `gradient_checkpointing=True`, `use_liger_kernel=True`")
        lines.append("- `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`")
        lines.append("- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`")
        lines.append("")

    if plot_assets:
        lines.append("## Training Curves")
        lines.append("")
        lines.append("Static plots below were exported from the private Weights & Biases run and embedded here for public access.")
        lines.append("")
        for asset in plot_assets:
            lines.append(f"### {asset['title']}")
            lines.append("")
            lines.append(f"![{asset['title']}]({asset['filename']})")
            lines.append("")

    lines.append("## Project Context")
    lines.append("")
    lines.append(
        f"This model is part of the metadata localization release. Related checkpoints and variants are grouped in the public Hugging Face collection [{collection_title}](https://huggingface.co/collections/{collection_slug})."
    )
    lines.append("- Training data source: [News on the Web (NOW) Corpus](https://www.english-corpora.org/now/)")
    lines.append(f"- Project repository: [{PROJECT_GITHUB_URL}]({PROJECT_GITHUB_URL})")
    lines.append(f"- Paper: [{PAPER_URL}]({PAPER_URL})")
    lines.append("")
    lines.append(f"Last synced: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`")
    lines.append("")
    return "\n".join(lines)


def upload_card_bundle(api: HfApi, spec: RepoSpec, card_text: str, asset_dir: Optional[Path]) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir)
        (bundle_dir / "README.md").write_text(card_text, encoding="utf-8")
        if asset_dir and asset_dir.exists():
            for path in asset_dir.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(asset_dir)
                    target = bundle_dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
        api.upload_folder(
            folder_path=str(bundle_dir),
            repo_id=spec.repo_id,
            repo_type="model",
            commit_message="Update model card and embedded training curves",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync HF collection and model cards for metadata localization models")
    parser.add_argument("--namespace", default=OWNER)
    parser.add_argument("--collection-slug", default=COLLECTION_SLUG)
    parser.add_argument("--collection-title", default=None)
    parser.add_argument("--collection-description", default=None)
    parser.add_argument("--include-3b-chat", action="store_true", default=False)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--skip-plots", action="store_true", default=False)
    parser.add_argument("--skip-cards", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--write-report", default="/path/to/metacul/results/hf_collection_sync_report.json")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be set")

    api = HfApi(token=token)
    wandb_api = wandb.Api()

    repos = list(api.list_models(author=args.namespace, full=True))
    repo_ids = sorted(model.id for model in repos if is_project_repo(model.id.split("/", 1)[1]))
    if not args.include_3b_chat:
        repo_ids = [repo_id for repo_id in repo_ids if not repo_id.endswith("_3b_chat")]
    if args.repo_id:
        repo_ids = [repo_id for repo_id in repo_ids if repo_id == args.repo_id]

    specs = sorted((parse_repo_spec(repo_id) for repo_id in repo_ids), key=sort_key)

    try:
        collection = api.get_collection(args.collection_slug)
    except Exception:
        create_title = args.collection_title or COLLECTION_TITLE_FALLBACK
        create_description = args.collection_description or COLLECTION_DESCRIPTION_FALLBACK
        collection = api.create_collection(
            create_title,
            namespace=args.namespace,
            description=create_description,
            exists_ok=True,
        )

    update_kwargs = {"private": False, "theme": "blue"}
    if args.collection_title is not None:
        update_kwargs["title"] = args.collection_title
    if args.collection_description is not None:
        update_kwargs["description"] = args.collection_description
    collection = api.update_collection_metadata(args.collection_slug, **update_kwargs)
    collection_title = args.collection_title or collection.title or COLLECTION_TITLE_FALLBACK
    collection_ref = args.collection_slug

    report = {
        "collection_slug": collection_ref,
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "repos": [],
        "extra_items": [],
    }

    for spec in specs:
        run_path = resolve_wandb_run_path(spec)
        run_info = None
        plot_assets = []
        asset_dir = None
        run = None
        if run_path:
            try:
                run = wandb_api.run(run_path)
                run_info = summarize_run(run, spec)
            except Exception as exc:  # pragma: no cover
                run_info = {"error": str(exc), "url": None, "name": run_path, "state": "error", "runtime": None, "summary": {}, "config": {}}

        if run and not args.skip_plots:
            with tempfile.TemporaryDirectory() as plot_tmpdir:
                tmp_asset_dir = Path(plot_tmpdir)
                try:
                    plot_assets = export_run_plots(run, spec, tmp_asset_dir)
                    if plot_assets:
                        asset_dir = Path(tempfile.mkdtemp())
                        for path in tmp_asset_dir.rglob("*"):
                            if path.is_file():
                                rel = path.relative_to(tmp_asset_dir)
                                target = asset_dir / rel
                                target.parent.mkdir(parents=True, exist_ok=True)
                                target.write_bytes(path.read_bytes())
                except Exception as exc:  # pragma: no cover
                    if run_info is None:
                        run_info = {"url": None, "name": run_path, "state": "error", "runtime": None, "summary": {}, "config": {}}
                    run_info["plot_error"] = str(exc)

        if not args.skip_cards:
            card_text = build_card(
                spec,
                run_info if run_info and "error" not in run_info else None,
                collection_title=collection_title,
                collection_slug=collection_ref,
                plot_assets=plot_assets,
            )
            if args.dry_run:
                print(f"[dry-run] would update card for {spec.repo_id}")
            else:
                upload_card_bundle(api, spec, card_text, asset_dir)
        if asset_dir is not None:
            shutil.rmtree(asset_dir, ignore_errors=True)

        if args.dry_run:
            print(f"[dry-run] would add {spec.repo_id} to {collection_ref}")
        else:
            api.add_collection_item(collection_ref, spec.repo_id, "model", note=spec.item_note, exists_ok=True)

        report["repos"].append(
            {
                "repo_id": spec.repo_id,
                "stage": spec.stage,
                "family": spec.family,
                "size": spec.size,
                "metadata_mode": spec.metadata_mode,
                "checkpoint": spec.checkpoint,
                "wandb_run": run_path,
                "plot_assets": [asset["filename"] for asset in plot_assets],
                "collection_note": spec.item_note,
            }
        )

    for item in EXTRA_COLLECTION_ITEMS:
        if args.dry_run:
            print(f"[dry-run] would add {item['item_id']} to {collection_ref}")
        else:
            api.add_collection_item(
                collection_ref,
                item["item_id"],
                item["item_type"],
                note=item["note"],
                exists_ok=True,
            )
        report["extra_items"].append(item)

    report_path = Path(args.write_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"collection={collection_ref}")
    print(f"repos_synced={len(specs)}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
