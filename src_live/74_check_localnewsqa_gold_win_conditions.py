#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


ROOT = Path("/path/to/metacul")
DEFAULT_RUN_ROOT = ROOT / "results/localnewsqa_gold_20260516"
SPLITS = ["Overall", "Explicit", "Ambiguous"]
FAMILIES = ["1B", "3B"]
VARIANTS = ["T-/I-", "T-/I+", "T+/I-", "T+/I+"]


def load_gold_gain_helpers():
    path = ROOT / "src/72_localnewsqa_gold_model_gain_tables.py"
    spec = importlib.util.spec_from_file_location("gold_gain72", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gold_gain72"] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LocalNewsQA gold win conditions.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def read_optional(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def acc_value(df: pd.DataFrame, family: str, variant: str, split: str) -> Optional[float]:
    series = f"{family} {variant}"
    row = df[(df["family"] == family) & (df["series"] == series) & (df["split"] == split)]
    if row.empty:
        return None
    return float(row.iloc[0]["accuracy"]) * 100.0


def add_check(rows: List[Dict[str, object]], group: str, check: str, ok: Optional[bool], detail: str) -> None:
    rows.append(
        {
            "group": group,
            "check": check,
            "status": "missing" if ok is None else ("pass" if ok else "fail"),
            "detail": detail,
        }
    )


def check_accuracy_matrix(rows: List[Dict[str, object]], name: str, df: Optional[pd.DataFrame]) -> None:
    if df is None:
        add_check(rows, name, "summary exists", None, "accuracy summary CSV missing")
        return

    for family in FAMILIES:
        for split in SPLITS:
            vals = {variant: acc_value(df, family, variant, split) for variant in VARIANTS}
            if any(v is None for v in vals.values()):
                add_check(rows, name, f"{family} {split} all variants present", None, str(vals))
                continue
            plus = vals["T+/I+"]
            minus = vals["T-/I-"]
            best_variant = max(vals, key=lambda key: vals[key])
            in_between = vals["T-/I-"] <= vals["T-/I+"] <= vals["T+/I+"] and vals["T-/I-"] <= vals["T+/I-"] <= vals["T+/I+"]
            add_check(
                rows,
                name,
                f"{family} {split} T+I+ > T-I-",
                plus > minus,
                f"T+I+={plus:.2f}, T-I-={minus:.2f}",
            )
            add_check(
                rows,
                name,
                f"{family} {split} T+I+ best",
                best_variant == "T+/I+",
                ", ".join(f"{k}={v:.2f}" for k, v in vals.items()),
            )
            add_check(
                rows,
                name,
                f"{family} {split} intermediates between endpoints",
                in_between,
                ", ".join(f"{k}={v:.2f}" for k, v in vals.items()),
            )

    for split in SPLITS:
        one_b = acc_value(df, "1B", "T+/I+", split)
        three_b = acc_value(df, "3B", "T+/I+", split)
        if one_b is None or three_b is None:
            add_check(rows, name, f"{split} 3B > 1B under T+I+", None, "missing value")
        else:
            add_check(
                rows,
                name,
                f"{split} 3B > 1B under T+I+",
                three_b > one_b,
                f"3B={three_b:.2f}, 1B={one_b:.2f}",
            )


def check_sft_vs_pretrained(
    rows: List[Dict[str, object]],
    pretrained: Optional[pd.DataFrame],
    sft: Optional[pd.DataFrame],
) -> None:
    if pretrained is None or sft is None:
        add_check(rows, "sft_vs_pretrained", "summaries exist", None, "pretrained or SFT CSV missing")
        return
    for family in FAMILIES:
        for split in SPLITS:
            pre = acc_value(pretrained, family, "T+/I+", split)
            post = acc_value(sft, family, "T+/I+", split)
            if pre is None or post is None:
                add_check(rows, "sft_vs_pretrained", f"{family} {split} SFT > pretrained", None, "missing value")
            else:
                add_check(
                    rows,
                    "sft_vs_pretrained",
                    f"{family} {split} SFT > pretrained",
                    post > pre,
                    f"SFT={post:.2f}, pretrained={pre:.2f}",
                )


def check_pretrained_1b_score_health(rows: List[Dict[str, object]], run_root: Path) -> None:
    files = sorted(
        (run_root / "pretrained_target/1b_codeg_labels_question_final").glob("seed_*/*.jsonl")
    ) + sorted(
        (run_root / "pretrained_contrast/1b_codeg_labels_question_final").glob("seed_*/*.jsonl")
    )
    expected = 5 * 4 * 2
    add_check(
        rows,
        "score_health",
        "1B pretrained expected files present",
        len(files) >= expected,
        f"found={len(files)}, expected={expected}",
    )
    if not files:
        return

    bad_files = []
    checked_files = 0
    for path in files:
        total = 0
        non_equal = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                scores = row.get("option_loglikelihood_selected_scores") or []
                if len(scores) >= 2 and max(scores) != min(scores):
                    non_equal += 1
                total += 1
                if total >= 200:
                    break
        if total:
            checked_files += 1
        if total and non_equal == 0:
            bad_files.append(path.name)
    add_check(
        rows,
        "score_health",
        "1B pretrained option scores non-degenerate",
        not bad_files and checked_files > 0,
        "bad_files=" + ",".join(bad_files[:8]) if bad_files else f"checked_files={checked_files}",
    )


def check_gain_ci(rows: List[Dict[str, object]], gains: Optional[pd.DataFrame]) -> None:
    if gains is None:
        add_check(rows, "gain_ci", "gain table exists", None, "gain CSV missing")
        return
    wanted = [
        ("maple_1b", "base"),
        ("maple_3b", "base"),
        ("maple_1b", "chat"),
        ("maple_3b", "chat"),
    ]
    metrics = [
        "localnewsqa_overall",
        "localnewsqa_ambiguous",
        "localnewsqa_explicit",
        "localnewsqa_exact_pair",
        "localnewsqa_margin_switch",
    ]
    for key, track in wanted:
        for metric in metrics:
            row = gains[
                (gains["row_key"] == key)
                & (gains["track"] == track)
                & (gains["metric_key"] == metric)
            ]
            if row.empty:
                add_check(rows, "gain_ci", f"{key} {track} {metric} CI clear positive", None, "missing row")
                continue
            r = row.iloc[0]
            delta = float(r["delta"])
            lo = float(r["ci_low"])
            hi = float(r["ci_high"])
            add_check(
                rows,
                "gain_ci",
                f"{key} {track} {metric} CI clear positive",
                delta > 0.0 and lo > 0.0,
                f"delta={delta:.2f}, ci=[{lo:.2f}, {hi:.2f}]",
            )


def switch_value(df: pd.DataFrame, family: str, variant: str, metric: str) -> Optional[float]:
    row = df[(df["family"] == family) & (df["variant"] == variant)]
    if row.empty:
        return None
    if metric in row.columns:
        return float(row.iloc[0][metric])
    return None


def check_pretrained_switch(rows: List[Dict[str, object]], sw: Optional[pd.DataFrame]) -> None:
    if sw is None:
        add_check(rows, "pretrained_switch", "summary exists", None, "switch CSV missing")
        return
    for family in ["MAPLE 1B", "MAPLE 3B"]:
        for metric in ["exact_pair_accuracy_pct", "margin_switch_rate_pct"]:
            plus = switch_value(sw, family, "tplus_eplus", metric)
            minus = switch_value(sw, family, "tminus_eminus", metric)
            if plus is None or minus is None:
                add_check(rows, "pretrained_switch", f"{family} {metric} T+I+ > T-I-", None, "missing value")
            else:
                add_check(
                    rows,
                    "pretrained_switch",
                    f"{family} {metric} T+I+ > T-I-",
                    plus > minus,
                    f"T+I+={plus:.2f}, T-I-={minus:.2f}",
                )
    for metric in ["exact_pair_accuracy_pct", "margin_switch_rate_pct"]:
        one_b = switch_value(sw, "MAPLE 1B", "tplus_eplus", metric)
        three_b = switch_value(sw, "MAPLE 3B", "tplus_eplus", metric)
        if one_b is None or three_b is None:
            add_check(rows, "pretrained_switch", f"{metric} 3B > 1B", None, "missing value")
        else:
            add_check(rows, "pretrained_switch", f"{metric} 3B > 1B", three_b > one_b, f"3B={three_b:.2f}, 1B={one_b:.2f}")


def check_sft_switch(rows: List[Dict[str, object]], sw: Optional[pd.DataFrame]) -> None:
    if sw is None:
        add_check(rows, "sft_switch", "summary exists", None, "SFT switch CSV missing")
        return
    for family in ["MAPLE Chat 1B", "MAPLE Chat 3B"]:
        for metric in ["exact_pair", "margin_switch"]:
            plus = switch_value(sw, family, "tplus_eplus", metric)
            minus = switch_value(sw, family, "tminus_eminus", metric)
            if plus is None or minus is None:
                add_check(rows, "sft_switch", f"{family} {metric} T+I+ > T-I-", None, "missing value")
            else:
                add_check(
                    rows,
                    "sft_switch",
                    f"{family} {metric} T+I+ > T-I-",
                    plus > minus,
                    f"T+I+={plus * 100.0:.2f}%, T-I-={minus * 100.0:.2f}%",
                )
    for metric in ["exact_pair", "margin_switch"]:
        one_b = switch_value(sw, "MAPLE Chat 1B", "tplus_eplus", metric)
        three_b = switch_value(sw, "MAPLE Chat 3B", "tplus_eplus", metric)
        if one_b is None or three_b is None:
            add_check(rows, "sft_switch", f"{metric} 3B > 1B", None, "missing value")
        else:
            add_check(
                rows,
                "sft_switch",
                f"{metric} 3B > 1B",
                three_b > one_b,
                f"3B={three_b * 100.0:.2f}%, 1B={one_b * 100.0:.2f}%",
            )


def group_name(row: Dict[str, object], group_by: str) -> str:
    if group_by == "country":
        return str(row.get("country") or row.get("target_country") or "").strip()
    return str(row.get("topic") or "").strip()


def ambiguous_group_accuracy(path: Path, spec: object, gain72: object, group_by: str) -> Dict[str, float]:
    values: Dict[str, List[float]] = {}
    for row in gain72.gain66.read_jsonl(path):
        if gain72.gain66.split_name(row) != "ambiguous":
            continue
        group = group_name(row, group_by)
        if not group:
            continue
        pred = gain72.gain66.lnqa_prediction(row, spec)
        if pred is None:
            val = float(bool(row.get("is_correct")))
        else:
            val = float(pred["is_correct"])
        values.setdefault(group, []).append(val)
    return {key: sum(vals) / len(vals) for key, vals in values.items() if vals}


def check_granular_ambiguous_gains(rows: List[Dict[str, object]], run_root: Path) -> None:
    try:
        gain72 = load_gold_gain_helpers()
    except Exception as exc:
        add_check(rows, "granular_ambiguous", "helper module loads", None, str(exc))
        return

    expected_counts = {"country": 17, "topic": 8}
    specs = [
        spec
        for spec in gain72.ROWS
        if spec.kind == "maple" and spec.track in {"base", "chat"}
    ]
    for spec in specs:
        try:
            plus_path = gain72.maple_path(run_root, spec, "tplus_eplus", 41, "target")
            minus_path = gain72.maple_path(run_root, spec, "tminus_eminus", 41, "target")
        except Exception as exc:
            add_check(rows, "granular_ambiguous", f"{spec.label} paths exist", None, str(exc))
            continue

        for group_by in ("country", "topic"):
            plus = ambiguous_group_accuracy(plus_path, spec, gain72, group_by)
            minus = ambiguous_group_accuracy(minus_path, spec, gain72, group_by)
            shared = sorted(set(plus).intersection(minus))
            expected = expected_counts[group_by]
            missing_count = len(shared) != expected
            deltas: List[Tuple[str, float]] = [(key, (plus[key] - minus[key]) * 100.0) for key in shared]
            non_positive = [(key, delta) for key, delta in deltas if delta <= 0.0]
            detail = (
                f"groups={len(shared)}/{expected}; "
                f"min_delta={min((d for _, d in deltas), default=float('nan')):.2f}; "
                f"non_positive={'; '.join(f'{k}:{d:.2f}' for k, d in non_positive[:8]) or 'none'}"
            )
            add_check(
                rows,
                "granular_ambiguous",
                f"{spec.label} ambiguous gain positive for every {group_by}",
                not missing_count and not non_positive,
                detail,
            )


def main() -> int:
    args = parse_args()
    run_root = args.run_root
    pretrained = read_optional(run_root / "plots/plot_8_pretrained_target_split_multiseed.csv")
    sft = read_optional(run_root / "plots/plot_8_sft_target_split_multiseed_gold.csv")
    gains = read_optional(run_root / "appendix_model_gain_tables/localnewsqa_model_gains_long.csv")
    switch = read_optional(run_root / "summaries/localnewsqa_locale_switch/summary.csv")
    sft_switch = read_optional(run_root / "summaries/localnewsqa_sft_locale_switch/summary.csv")

    rows: List[Dict[str, object]] = []
    check_accuracy_matrix(rows, "pretrained_accuracy", pretrained)
    check_accuracy_matrix(rows, "sft_accuracy", sft)
    check_sft_vs_pretrained(rows, pretrained, sft)
    check_pretrained_1b_score_health(rows, run_root)
    check_gain_ci(rows, gains)
    check_pretrained_switch(rows, switch)
    check_sft_switch(rows, sft_switch)
    check_granular_ambiguous_gains(rows, run_root)

    out_csv = args.out_csv or (run_root / "summaries/win_conditions.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(out_csv, index=False)

    counts = report["status"].value_counts().to_dict()
    print(f"[ok] wrote {out_csv}")
    print("status counts:", counts)
    failed = report[report["status"] == "fail"]
    if not failed.empty:
        print("\nFailed checks:")
        for _, row in failed.iterrows():
            print(f"- {row['group']}: {row['check']} ({row['detail']})")
        return 2
    missing = report[report["status"] == "missing"]
    if not missing.empty:
        print("\nMissing checks:")
        for _, row in missing.iterrows():
            print(f"- {row['group']}: {row['check']} ({row['detail']})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
