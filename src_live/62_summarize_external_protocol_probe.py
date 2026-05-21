#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


ROOT = Path("/path/to/metacul")
RESULT_ROOT = ROOT / "results" / "external_protocol_probes"

FAMILIES = ["maple_1b", "maple_3b", "llama32_1b", "llama32_3b", "gemma4_e2b", "gemma4_e4b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def read_summary_csv(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["variant"]] = row
    return out


def extract_metric(benchmark: str, family_dir: Path) -> tuple[str, dict[str, float]] | None:
    seed_dir = family_dir / f"seed_{args.seed}" / benchmark
    if not seed_dir.exists():
        return None
    if benchmark == "worldvaluebench":
        values = {}
        for variant in [
            "custom_tplus_eplus",
            "custom_tminus_eminus",
            "llama3_chat_with_metadata",
            "llama3_chat_without_metadata",
        ]:
            summary_json = seed_dir / variant / f"{benchmark}_{variant}_emd_summary.json"
            if summary_json.exists():
                values[variant] = json.loads(summary_json.read_text())["overall_emd"]
        return "emd", values
    values = {}
    for variant in [
        "custom_tplus_eplus",
        "custom_tminus_eminus",
        "llama3_chat_with_metadata",
        "llama3_chat_without_metadata",
    ]:
        summary_csv = seed_dir / variant / f"{benchmark}_eval_summary.csv"
        if not summary_csv.exists():
            continue
        summaries = read_summary_csv(summary_csv)
        row = summaries.get(variant)
        if row and row.get("status") == "ok":
            values[variant] = 100.0 * float(row["accuracy"])
    return "accuracy", values


def direction_ok(metric: str, better: float, worse: float) -> bool:
    return better < worse if metric == "emd" else better > worse


if __name__ == "__main__":
    args = parse_args()
    base = RESULT_ROOT / args.benchmark
    rows = []
    for protocol_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        protocol = protocol_dir.name
        row = {"protocol": protocol}
        ok_all = True
        for family in FAMILIES:
            family_dir = protocol_dir / family
            metric_info = extract_metric(args.benchmark, family_dir)
            if metric_info is None:
                row[family] = "missing"
                ok_all = False
                continue
            metric, values = metric_info
            if family.startswith("maple"):
                a = values.get("custom_tplus_eplus")
                b = values.get("custom_tminus_eminus")
            else:
                a = values.get("llama3_chat_with_metadata")
                b = values.get("llama3_chat_without_metadata")
            if a is None or b is None:
                row[family] = "missing"
                ok_all = False
                continue
            mark = "ok" if direction_ok(metric, a, b) else "fail"
            row[family] = f"{mark}:{a:.3f} vs {b:.3f}"
            ok_all = ok_all and (mark == "ok")
        row["all_required_ok"] = "yes" if ok_all else "no"
        rows.append(row)

    print("\t".join(["protocol", *FAMILIES, "all_required_ok"]))
    for row in rows:
        print("\t".join([row.get("protocol", ""), *[row.get(f, "") for f in FAMILIES], row.get("all_required_ok", "")]))
