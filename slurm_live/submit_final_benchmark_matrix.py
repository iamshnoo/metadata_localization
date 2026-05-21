#!/usr/bin/env python3
import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


ROOT = Path("/path/to/metacul")
INVENTORY = ROOT / ".agents/benchmarks/final_benchmark_inventory_20260502.csv"
LOG_DIR = Path("/path/to/logs/slurm_logs")
EXT_WRAPPER = ROOT / "slurm/pretrained_external_eval_single.slurm"
LNQA_WRAPPER = ROOT / "slurm/localnewsqa_eval_baseline_single.slurm"
LNQA_CUSTOM_WRAPPER = ROOT / "slurm/pretrained_localnewsqa_eval_single.slurm"
RESULT_ROOT = ROOT / "results/final_benchmark_matrix"
DEFAULT_SEEDS = [41, 42, 43, 44, 45]

DATASET_TO_BENCHMARK = {
    "geomlama": "geomlama",
    "globalopinionqa": "globalopinionqa",
    "worldvaluesbench": "worldvaluebench",
    "normad": "normad",
    "blend": "blend",
    "globalmmlu_cs": "globalmmlu_cs",
}

EXT_VARIANT_MAP = {
    "I+": "llama3_chat_with_metadata",
    "I-": "llama3_chat_without_metadata",
}

MAPLE_VARIANT_MAP = {
    "T+/I+": "custom_tplus_eplus",
    "T+/I-": "custom_tplus_eminus",
    "T-/I+": "custom_tminus_eplus",
    "T-/I-": "custom_tminus_eminus",
}

MAPLE_MODEL_PATHS = {
    ("maple_1b", "base"): (
        "/path/to/metacul/models/combined_with_metadata_1b",
        "/path/to/metacul/models/combined_without_metadata_1b",
    ),
    ("maple_3b", "base"): (
        "/path/to/metacul/models/combined_with_metadata_3b",
        "/path/to/metacul/models/combined_without_metadata_3b",
    ),
    ("maple_1b", "chat"): (
        "/path/to/metacul/models/sft/combined_with_metadata_chat",
        "/path/to/metacul/models/sft/combined_without_metadata_chat",
    ),
    ("maple_3b", "chat"): (
        "/path/to/metacul/models/sft/combined_with_metadata_3b_best3b_chat",
        "/path/to/metacul/models/sft/combined_without_metadata_3b_best3b_chat",
    ),
}

MAPLE_CHAT_FIGURE_CONFIGS = {
    ("maple_1b", "chat"): {
        "METADATA_PROMPT_STYLE": "name_plain",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "country_final_answer_colon",
        "OMIT_OPTION_LABELS": "1",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "0.25",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked",
        "NULL_CALIBRATION_BETA": "0.25",
    },
    ("maple_3b", "chat"): {
        "METADATA_PROMPT_STYLE": "name_grounded",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "final_answer_colon",
        "OMIT_OPTION_LABELS": "1",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "0.25",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked",
        "NULL_CALIBRATION_BETA": "0.5",
    },
}


def maple_variant_tags(variant: str) -> tuple[str, str]:
    train_meta = "with_metadata" if variant.startswith("T+/") else "without_metadata"
    eval_meta = "with_metadata" if variant.endswith("I+") else "without_metadata"
    return train_meta, eval_meta


@dataclass
class Row:
    dataset_key: str
    family_key: str
    family_label: str
    kind: str
    track: str
    variant: str
    model_id: str
    rerun_policy: str


def load_rows(
    only_datasets: Optional[Set[str]] = None,
    only_families: Optional[Set[str]] = None,
    only_tracks: Optional[Set[str]] = None,
) -> List[Row]:
    rows: List[Row] = []
    with INVENTORY.open() as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if raw["rerun_policy"] == "skip":
                continue
            row = Row(
                dataset_key=raw["dataset_key"],
                family_key=raw["family_key"],
                family_label=raw["family_label"],
                kind=raw["kind"],
                track=raw["track"],
                variant=raw["variant"],
                model_id=raw["model_id"],
                rerun_policy=raw["rerun_policy"],
            )
            if only_datasets and row.dataset_key not in only_datasets:
                continue
            if only_families and row.family_key not in only_families:
                continue
            if only_tracks and row.track not in only_tracks:
                continue
            rows.append(
                Row(
                    dataset_key=row.dataset_key,
                    family_key=row.family_key,
                    family_label=row.family_label,
                    kind=row.kind,
                    track=row.track,
                    variant=row.variant,
                    model_id=row.model_id,
                    rerun_policy=row.rerun_policy,
                )
            )
    return rows


def export_arg(d: Dict[str, str]) -> str:
    return "ALL," + ",".join(f"{k}={v}" for k, v in d.items())


def sbatch_submit(script: Path, env: Dict[str, str], extra_args: Optional[List[str]] = None) -> str:
    cmd = ["sbatch", "--parsable"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(["--export", export_arg(env), str(script)])
    out = subprocess.check_output(cmd, universal_newlines=True).strip()
    return out.split(";")[0].strip()


def benchmark_sbatch_args(row: Row) -> List[str]:
    if (
        row.dataset_key == "localnewsqa"
        and row.kind == "external"
        and row.family_key in {"qwen35_2b", "qwen35_4b", "ministral3_3b"}
    ):
        return [
            "--account",
            "SLURM_ACCOUNT",
            "--partition",
            "gpuq",
            "--qos",
            "gpu",
            "--gres",
            "gpu:A100.40gb:1",
            "--mem",
            "96GB",
            "--time",
            "06:00:00",
        ]
    # Larger Qwen base models have shown loader-side bus errors on 3g.40gb MIG.
    # Route them to a full A100 lane for the benchmark matrix.
    if row.kind == "external" and row.track == "base" and row.family_key in {"qwen25_3b", "qwen35_4b"}:
        return [
            "--account",
            "SLURM_ACCOUNT",
            "--partition",
            "gpuq",
            "--qos",
            "gpu",
            "--gres",
            "gpu:A100.80gb:1",
            "--mem",
            "96GB",
            "--time",
            "06:00:00" if row.dataset_key == "localnewsqa" else "03:00:00",
        ]
    if row.dataset_key == "localnewsqa":
        return [
            "--account",
            "SLURM_ACCOUNT",
            "--partition",
            "contrib-gpuq",
            "--qos",
            "cs_dept",
            "--time",
            "06:00:00",
        ]
    return [
        "--account",
        "SLURM_ACCOUNT",
        "--partition",
        "contrib-gpuq",
        "--qos",
        "cs_dept",
        "--time",
        "03:00:00",
    ]


def localnewsqa_jobs(row: Row, seed: int) -> Iterable[Dict[str, str]]:
    if row.kind == "maple":
        train_meta, eval_meta = maple_variant_tags(row.variant)
        out_dir = RESULT_ROOT / "localnewsqa" / row.family_key / row.track / f"seed_{seed}"
        env = {
            "MODEL_TYPE": "custom",
            "META_TAG": eval_meta,
            "TRAIN_META_TAG": train_meta,
            "EVAL_META_TAG": eval_meta,
            "OUT_DIR": str(out_dir),
            "RUN_TAG": f"matrix_{row.family_key}_{row.track}_{row.variant.replace('/', '').replace('+', 'p').replace('-', 'm')}_seed{seed}",
            "SHUFFLE_OPTIONS": "1",
            "OPTION_SHUFFLE_SEED": str(seed),
            "SAMPLE_SEED": str(seed),
            "ANSWER_FORMAT": "option",
        }
        if row.track == "base":
            with_path, without_path = MAPLE_MODEL_PATHS[(row.family_key, row.track)]
            env.update(
                {
                    "CUSTOM_MODEL_PATH_WITH": with_path,
                    "CUSTOM_MODEL_PATH_WITHOUT": without_path,
                }
            )
        else:
            with_path, without_path = MAPLE_MODEL_PATHS[(row.family_key, row.track)]
            env.update(
                {
                    "CUSTOM_MODEL_PATH_WITH": with_path,
                    "CUSTOM_MODEL_PATH_WITHOUT": without_path,
                }
            )
            env.update(MAPLE_CHAT_FIGURE_CONFIGS[(row.family_key, row.track)])

        for role in ("target", "contrast"):
            role_env = dict(env)
            role_env["LOCALE_ROLE"] = role
            yield {
                "script": str(LNQA_CUSTOM_WRAPPER),
                "job_name": f"bm-lnqa-{row.family_key}-{row.track}-{row.variant.replace('/', '')}-{role}-s{seed}",
                "env": role_env,
                "manifest_role": role,
            }
        return

    meta_tag = "with_metadata" if row.variant == "I+" else "without_metadata"
    localize = "1" if row.model_id.startswith("meta-llama/") else "0"
    model_slug = f"{row.family_key}_{row.track}"
    for role in ("target", "contrast"):
        out_dir = RESULT_ROOT / "localnewsqa" / row.family_key / row.track / f"seed_{seed}" / role
        yield {
            "script": str(LNQA_WRAPPER),
            "job_name": f"bm-lnqa-{row.family_key}-{row.track}-{meta_tag}-{role}-s{seed}",
            "env": {
                "MODEL_NAME": row.model_id,
                "MODEL_SLUG": model_slug,
                "META_TAG": meta_tag,
                "LOCALE_ROLE": role,
                "OUT_DIR": str(out_dir),
                "RUN_TAG": f"matrix_{row.family_key}_{row.track}_seed{seed}",
                "SHUFFLE_OPTIONS": "1",
                "OPTION_SHUFFLE_SEED": str(seed),
                "MCQ_SCORING": "option_text_avg",
                "ANSWER_FORMAT": "option",
                "SPLIT_TYPE_FILTER": "all",
                "LOCALIZE_MODEL": localize,
            },
            "manifest_role": role,
        }


def external_benchmark_job(row: Row, seed: int) -> Dict[str, str]:
    benchmark = DATASET_TO_BENCHMARK[row.dataset_key]
    out_root = RESULT_ROOT / "closed_form" / row.family_key / row.track
    env = {
        "BENCHMARK": benchmark,
        "EVAL_SEED": str(seed),
        "SHUFFLE_OPTIONS": "1",
        "OPTION_SHUFFLE_SEED": str(seed),
        "OUT_ROOT": str(out_root),
    }
    if row.dataset_key == "globalmmlu_cs":
        env["METADATA_TAG_MODE"] = "available_only"

    if row.kind == "external":
        env["VARIANT"] = EXT_VARIANT_MAP[row.variant]
        env["LLAMA_MODEL_NAME"] = row.model_id
    else:
        env["VARIANT"] = MAPLE_VARIANT_MAP[row.variant]
        with_path, without_path = MAPLE_MODEL_PATHS[(row.family_key, row.track)]
        env["CUSTOM_MODEL_PATH_WITH"] = with_path
        env["CUSTOM_MODEL_PATH_WITHOUT"] = without_path
        if row.track == "chat":
            env.update(MAPLE_CHAT_FIGURE_CONFIGS[(row.family_key, row.track)])

    return {
        "script": str(EXT_WRAPPER),
        "job_name": f"bm-{row.dataset_key}-{row.family_key}-{row.track}-{row.variant.replace('/', '')}-s{seed}",
        "env": env,
        "manifest_role": "metric",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit the required final benchmark matrix jobs.")
    parser.add_argument("--datasets", default="", help="Comma-separated dataset_key filter.")
    parser.add_argument("--families", default="", help="Comma-separated family_key filter.")
    parser.add_argument("--tracks", default="", help="Comma-separated track filter (base,chat).")
    parser.add_argument("--seeds", default="41,42,43,44,45", help="Comma-separated eval seeds.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of row entries to submit.")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected workload without submitting.")
    return parser.parse_args()


def split_filter(text: str) -> Optional[Set[str]]:
    vals = {x.strip() for x in text.split(",") if x.strip()}
    return vals or None


def main() -> None:
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    only_datasets = split_filter(args.datasets)
    only_families = split_filter(args.families)
    only_tracks = split_filter(args.tracks)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(only_datasets=only_datasets, only_families=only_families, only_tracks=only_tracks)
    if args.limit > 0:
        rows = rows[: args.limit]
    ts = subprocess.check_output(
        ["date", "+%Y%m%d_%H%M%S"], universal_newlines=True
    ).strip()
    manifest = LOG_DIR / f"final_benchmark_matrix_{ts}.tsv"
    if args.dry_run:
        total_jobs = sum((2 if r.dataset_key == "localnewsqa" else 1) * len(seeds) for r in rows)
        print(f"rows={len(rows)} jobs={total_jobs}")
        print(f"manifest={manifest}")
        return
    with manifest.open("w") as f:
        f.write("job_id\tdataset\tfamily\ttrack\tvariant\tseed\trole\tscript\tout_root_or_dir\n")

        for row in rows:
            for seed in seeds:
                if row.dataset_key == "localnewsqa":
                    for job in localnewsqa_jobs(row, seed):
                        job_id = sbatch_submit(
                            Path(job["script"]),
                            job["env"],
                            benchmark_sbatch_args(row) + ["--job-name", job["job_name"]],
                        )
                        f.write(
                            "\t".join(
                                [
                                    job_id,
                                    row.dataset_key,
                                    row.family_key,
                                    row.track,
                                    row.variant,
                                    str(seed),
                                    job["manifest_role"],
                                    job["script"],
                                    job["env"]["OUT_DIR"],
                                ]
                            )
                            + "\n"
                        )
                else:
                    job = external_benchmark_job(row, seed)
                    job_id = sbatch_submit(
                        Path(job["script"]),
                        job["env"],
                        benchmark_sbatch_args(row) + ["--job-name", job["job_name"]],
                    )
                    f.write(
                        "\t".join(
                            [
                                job_id,
                                row.dataset_key,
                                row.family_key,
                                row.track,
                                row.variant,
                                str(seed),
                                job["manifest_role"],
                                job["script"],
                                job["env"]["OUT_ROOT"],
                            ]
                        )
                        + "\n"
                    )

    total_jobs = sum(1 for _ in manifest.open()) - 1
    print(f"Wrote manifest: {manifest}")
    print(f"Submitted jobs: {total_jobs}")


if __name__ == "__main__":
    main()
