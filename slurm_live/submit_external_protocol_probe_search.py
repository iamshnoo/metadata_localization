#!/usr/bin/env python3
import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path("/path/to/metacul")
LOG_DIR = Path("/path/to/logs/slurm_logs")
WRAPPER = ROOT / "slurm/pretrained_external_eval_single.slurm"
RESULT_ROOT = ROOT / "results/external_protocol_probes"


@dataclass(frozen=True)
class FamilySpec:
    family_key: str
    family_label: str
    kind: str
    model_name: str | None = None
    custom_with: str | None = None
    custom_without: str | None = None


FAMILIES: Dict[str, FamilySpec] = {
    "maple_1b": FamilySpec(
        family_key="maple_1b",
        family_label="MAPLE 1B",
        kind="maple",
        custom_with="/path/to/metacul/models/combined_with_metadata_1b",
        custom_without="/path/to/metacul/models/combined_without_metadata_1b",
    ),
    "maple_3b": FamilySpec(
        family_key="maple_3b",
        family_label="MAPLE 3B",
        kind="maple",
        custom_with="/path/to/metacul/models/combined_with_metadata_3b",
        custom_without="/path/to/metacul/models/combined_without_metadata_3b",
    ),
    "maple_1b_chat": FamilySpec(
        family_key="maple_1b_chat",
        family_label="MAPLE 1B Chat",
        kind="maple",
        custom_with="/path/to/metacul/models/sft/combined_with_metadata_chat",
        custom_without="/path/to/metacul/models/sft/combined_without_metadata_chat",
    ),
    "maple_3b_chat": FamilySpec(
        family_key="maple_3b_chat",
        family_label="MAPLE 3B Chat",
        kind="maple",
        custom_with="/path/to/metacul/models/sft/combined_with_metadata_3b_best3b_chat",
        custom_without="/path/to/metacul/models/sft/combined_without_metadata_3b_best3b_chat",
    ),
    "llama32_1b": FamilySpec(
        family_key="llama32_1b",
        family_label="LLaMA-3.2-1B",
        kind="baseline",
        model_name="meta-llama/Llama-3.2-1B",
    ),
    "llama32_3b": FamilySpec(
        family_key="llama32_3b",
        family_label="LLaMA-3.2-3B",
        kind="baseline",
        model_name="meta-llama/Llama-3.2-3B",
    ),
    "gemma4_e2b": FamilySpec(
        family_key="gemma4_e2b",
        family_label="Gemma-4-E2B",
        kind="baseline",
        model_name="google/gemma-4-E2B",
    ),
    "gemma4_e4b": FamilySpec(
        family_key="gemma4_e4b",
        family_label="Gemma-4-E4B",
        kind="baseline",
        model_name="google/gemma-4-E4B",
    ),
}


PROTOCOLS: Dict[str, Dict[str, str]] = {
    "figure9_1b": {
        "METADATA_PROMPT_STYLE": "code_grounded",
        "QA_PROMPT_STYLE": "question",
        "ANSWER_CUE_STYLE": "final_answer_colon",
        "OMIT_OPTION_LABELS": "0",
        "EXACT_OPTION_TEXT_INSTRUCTION": "1",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "1.25",
        "ADD_PROMPT_BOS": "0",
        "NULL_CALIBRATION_MODE": "none",
        "NULL_CALIBRATION_BETA": "0.0",
    },
    "figure9_3b": {
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
    "sft_1b_lnqa": {
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
    "sft_3b_lnqa": {
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
    "legacy_pair": {
        "METADATA_PROMPT_STYLE": "legacy_code",
        "QA_PROMPT_STYLE": "question",
        "ANSWER_CUE_STYLE": "none",
        "OMIT_OPTION_LABELS": "0",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_text_avg",
        "ADD_PROMPT_BOS": "0",
        "NULL_CALIBRATION_MODE": "none",
        "NULL_CALIBRATION_BETA": "0.0",
    },
    "legacy_a025": {
        "METADATA_PROMPT_STYLE": "legacy_code",
        "QA_PROMPT_STYLE": "question",
        "ANSWER_CUE_STYLE": "none",
        "OMIT_OPTION_LABELS": "0",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "0.25",
        "ADD_PROMPT_BOS": "0",
        "NULL_CALIBRATION_MODE": "none",
        "NULL_CALIBRATION_BETA": "0.0",
    },
    "plain_grounded": {
        "METADATA_PROMPT_STYLE": "name_grounded",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "final_answer_colon",
        "OMIT_OPTION_LABELS": "1",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "0.25",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "none",
        "NULL_CALIBRATION_BETA": "0.0",
    },
    "strict_country": {
        "METADATA_PROMPT_STYLE": "name_strict",
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
    "strict_country_a0_b15": {
        "METADATA_PROMPT_STYLE": "name_strict",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "country_final_answer_colon",
        "OMIT_OPTION_LABELS": "1",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "0.0",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked",
        "NULL_CALIBRATION_BETA": "1.5",
    },
    "strict_country_a2_b1": {
        "METADATA_PROMPT_STYLE": "name_strict",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "country_final_answer_colon",
        "OMIT_OPTION_LABELS": "1",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "2.0",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked",
        "NULL_CALIBRATION_BETA": "1.0",
    },
    "code_strict": {
        "METADATA_PROMPT_STYLE": "code_grounded_strict",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "country_final_answer_colon",
        "OMIT_OPTION_LABELS": "1",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "0.25",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked_no_metadata",
        "NULL_CALIBRATION_BETA": "0.25",
    },
    "country_first": {
        "METADATA_PROMPT_STYLE": "country_first_strict",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "country_the_correct_answer_is",
        "OMIT_OPTION_LABELS": "1",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "0.25",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked",
        "NULL_CALIBRATION_BETA": "0.25",
    },
    "disambiguate_cal": {
        "METADATA_PROMPT_STYLE": "code_disambiguate",
        "QA_PROMPT_STYLE": "instruction_input",
        "ANSWER_CUE_STYLE": "country_final_answer_colon",
        "OMIT_OPTION_LABELS": "0",
        "EXACT_OPTION_TEXT_INSTRUCTION": "1",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "0.25",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked",
        "NULL_CALIBRATION_BETA": "0.5",
    },
    "legacy_chatforce": {
        "METADATA_PROMPT_STYLE": "legacy_code",
        "QA_PROMPT_STYLE": "question",
        "ANSWER_CUE_STYLE": "none",
        "OMIT_OPTION_LABELS": "0",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_text_avg",
        "ADD_PROMPT_BOS": "0",
        "NULL_CALIBRATION_MODE": "none",
        "NULL_CALIBRATION_BETA": "0.0",
        "FORCE_CUSTOM_CHAT_PROMPT": "1",
    },
    "plain_grounded_chatforce": {
        "METADATA_PROMPT_STYLE": "name_grounded",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "final_answer_colon",
        "OMIT_OPTION_LABELS": "1",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "0.25",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "none",
        "NULL_CALIBRATION_BETA": "0.0",
        "FORCE_CUSTOM_CHAT_PROMPT": "1",
    },
    "strict_country_chatforce": {
        "METADATA_PROMPT_STYLE": "name_strict",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "country_final_answer_colon",
        "OMIT_OPTION_LABELS": "1",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "0.25",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked",
        "NULL_CALIBRATION_BETA": "0.25",
        "FORCE_CUSTOM_CHAT_PROMPT": "1",
    },
    "letter_legacy": {
        "METADATA_PROMPT_STYLE": "legacy_code",
        "QA_PROMPT_STYLE": "question",
        "ANSWER_CUE_STYLE": "answer_colon",
        "OMIT_OPTION_LABELS": "0",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_letter",
        "ADD_PROMPT_BOS": "0",
        "NULL_CALIBRATION_MODE": "none",
        "NULL_CALIBRATION_BETA": "0.0",
    },
    "letter_country_first": {
        "METADATA_PROMPT_STYLE": "country_first_strict",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "country_final_answer_colon",
        "OMIT_OPTION_LABELS": "0",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_letter",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked",
        "NULL_CALIBRATION_BETA": "0.25",
    },
    "letter_name_grounded": {
        "METADATA_PROMPT_STYLE": "name_grounded",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "final_answer_colon",
        "OMIT_OPTION_LABELS": "0",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_letter",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked",
        "NULL_CALIBRATION_BETA": "0.5",
    },
    "letter_strict_country": {
        "METADATA_PROMPT_STYLE": "name_strict",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "country_final_answer_colon",
        "OMIT_OPTION_LABELS": "0",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_letter",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked",
        "NULL_CALIBRATION_BETA": "0.25",
    },
    "letter_code_strict": {
        "METADATA_PROMPT_STYLE": "code_grounded_strict",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "country_final_answer_colon",
        "OMIT_OPTION_LABELS": "0",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_letter",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked_no_metadata",
        "NULL_CALIBRATION_BETA": "0.25",
    },
    "letter_country_the": {
        "METADATA_PROMPT_STYLE": "country_first_strict",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "country_the_correct_answer_is",
        "OMIT_OPTION_LABELS": "0",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_letter",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked",
        "NULL_CALIBRATION_BETA": "0.25",
    },
    "name_grounded_the": {
        "METADATA_PROMPT_STYLE": "name_grounded",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "the_correct_answer_is",
        "OMIT_OPTION_LABELS": "1",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "0.25",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked",
        "NULL_CALIBRATION_BETA": "0.5",
    },
    "strict_country_the": {
        "METADATA_PROMPT_STYLE": "name_strict",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "country_the_correct_answer_is",
        "OMIT_OPTION_LABELS": "1",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "0.25",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked",
        "NULL_CALIBRATION_BETA": "0.25",
    },
    "legacy_final": {
        "METADATA_PROMPT_STYLE": "legacy_code",
        "QA_PROMPT_STYLE": "question",
        "ANSWER_CUE_STYLE": "final_answer_colon",
        "OMIT_OPTION_LABELS": "0",
        "EXACT_OPTION_TEXT_INSTRUCTION": "1",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "0.25",
        "ADD_PROMPT_BOS": "0",
        "NULL_CALIBRATION_MODE": "none",
        "NULL_CALIBRATION_BETA": "0.0",
    },
    "name_plain_country_the": {
        "METADATA_PROMPT_STYLE": "name_plain",
        "QA_PROMPT_STYLE": "question_answer",
        "ANSWER_CUE_STYLE": "country_the_correct_answer_is",
        "OMIT_OPTION_LABELS": "1",
        "EXACT_OPTION_TEXT_INSTRUCTION": "0",
        "MCQ_SCORING": "option_text_avg",
        "LENGTH_NORM_ALPHA": "0.25",
        "ADD_PROMPT_BOS": "1",
        "NULL_CALIBRATION_MODE": "question_masked",
        "NULL_CALIBRATION_BETA": "0.25",
    },
}


def export_arg(d: Dict[str, str]) -> str:
    return "ALL," + ",".join(f"{k}={v}" for k, v in d.items())


def sbatch_submit(script: Path, env: Dict[str, str], extra_args: List[str]) -> str:
    cmd = ["sbatch", "--parsable", *extra_args, "--export", export_arg(env), str(script)]
    out = subprocess.check_output(cmd, universal_newlines=True).strip()
    return out.split(";")[0].strip()


def queue_args(family_key: str) -> List[str]:
    if family_key in {"gemma4_e4b"}:
        return [
            "--account", "SLURM_ACCOUNT",
            "--partition", "contrib-gpuq",
            "--qos", "gpu",
            "--gres", "gpu:A100.80gb:1",
            "--mem", "96GB",
            "--time", "06:00:00",
        ]
    return [
        "--account", "SLURM_ACCOUNT",
        "--partition", "contrib-gpuq",
        "--qos", "gpu",
        "--gres", "gpu:3g.40gb:1",
        "--mem", "48GB",
        "--time", "06:00:00",
    ]


def iter_variant_jobs(family: FamilySpec, phase: str) -> List[tuple[str, Dict[str, str]]]:
    if family.kind == "maple":
        jobs = [
            (
                "custom_tplus_eplus",
                {
                    "VARIANT": "custom_tplus_eplus",
                    "CUSTOM_MODEL_PATH_WITH": family.custom_with or "",
                    "CUSTOM_MODEL_PATH_WITHOUT": family.custom_without or "",
                },
            ),
            (
                "custom_tminus_eminus",
                {
                    "VARIANT": "custom_tminus_eminus",
                    "CUSTOM_MODEL_PATH_WITH": family.custom_with or "",
                    "CUSTOM_MODEL_PATH_WITHOUT": family.custom_without or "",
                },
            ),
        ]
    else:
        jobs = [
            (
                "llama3_chat_with_metadata",
                {
                    "VARIANT": "llama3_chat_with_metadata",
                    "LLAMA_MODEL_NAME": family.model_name or "",
                },
            ),
            (
                "llama3_chat_without_metadata",
                {
                    "VARIANT": "llama3_chat_without_metadata",
                    "LLAMA_MODEL_NAME": family.model_name or "",
                },
            ),
        ]
    if phase == "positive":
        return [jobs[0]]
    if phase == "negative":
        return [jobs[1]]
    return jobs


def grouped_variant_job(family: FamilySpec) -> tuple[str, Dict[str, str]]:
    jobs = iter_variant_jobs(family, phase="both")
    variant_names = [name for name, _env in jobs]
    env: Dict[str, str] = {
        "VARIANT": variant_names[0],
        "VARIANTS": ":".join(variant_names),
        "FLAT_OUTPUT": "1",
    }
    for _name, variant_env in jobs:
        for key, value in variant_env.items():
            if key in {"VARIANT", "VARIANTS"}:
                continue
            env[key] = value
    return "both", env


def iter_jobs(
    benchmark: str,
    seed: int,
    max_samples: int,
    family_keys: Iterable[str],
    protocol_keys: Iterable[str],
    phase: str,
    result_root: Path,
) -> Iterable[tuple[str, Dict[str, str], List[str], str, str, str]]:
    for family_key in family_keys:
        family = FAMILIES[family_key]
        for protocol_key in protocol_keys:
            protocol = PROTOCOLS[protocol_key]
            out_root = result_root / benchmark / protocol_key / family_key
            variant_jobs = (
                [grouped_variant_job(family)]
                if phase == "both"
                else iter_variant_jobs(family, phase=phase)
            )
            for variant_name, variant_env in variant_jobs:
                metadata_tag_mode = "available_only" if benchmark == "globalmmlu_cs" else "correct"
                env = {
                    "BENCHMARK": benchmark,
                    "OUT_ROOT": str(out_root),
                    "OUT_SEED_TAG": f"seed_{seed}",
                    "EVAL_SEED": str(seed),
                    "SHUFFLE_OPTIONS": "1",
                    "OPTION_SHUFFLE_SEED": str(seed),
                    "MAX_SAMPLES": str(max_samples),
                    "RESUME_EVAL": "1",
                    "METADATA_TAG_MODE": metadata_tag_mode,
                    **protocol,
                    **variant_env,
                }
                job_name = f"probe-{benchmark}-{protocol_key}-{family_key}-{variant_name}-s{seed}"
                yield job_name, env, queue_args(family_key), family_key, protocol_key, variant_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument(
        "--families",
        default="maple_1b,maple_3b,llama32_1b",
    )
    parser.add_argument(
        "--protocols",
        default=(
            "figure9_1b,figure9_3b,legacy_pair,plain_grounded,strict_country,"
            "code_strict,country_first,disambiguate_cal,letter_legacy,letter_country_first"
        ),
    )
    parser.add_argument(
        "--phase",
        choices=["positive", "negative", "both"],
        default="both",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=RESULT_ROOT,
        help="Root directory for probe outputs. Use a fresh root after protocol/tokenizer changes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = subprocess.check_output(["date", "+%Y%m%d_%H%M%S"], universal_newlines=True).strip()
    manifest = LOG_DIR / f"external_protocol_probe_{args.benchmark}_{timestamp}.tsv"
    family_keys = [x.strip() for x in args.families.split(",") if x.strip()]
    protocol_keys = [x.strip() for x in args.protocols.split(",") if x.strip()]
    for key in family_keys:
        if key not in FAMILIES:
            raise ValueError(f"Unknown family: {key}")
    for key in protocol_keys:
        if key not in PROTOCOLS:
            raise ValueError(f"Unknown protocol: {key}")

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["job_id", "benchmark", "seed", "family_key", "protocol_key", "variant", "job_name"])
        for job_name, env, extra_args, family_key, protocol_key, variant_name in iter_jobs(
            benchmark=args.benchmark,
            seed=args.seed,
            max_samples=args.max_samples,
            family_keys=family_keys,
            protocol_keys=protocol_keys,
            phase=args.phase,
            result_root=args.result_root,
        ):
            job_id = sbatch_submit(WRAPPER, env, ["--job-name", job_name, *extra_args])
            writer.writerow([job_id, args.benchmark, args.seed, family_key, protocol_key, variant_name, job_name])
            print(f"[submitted] {job_id} {job_name}")

    print(f"[done] manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
