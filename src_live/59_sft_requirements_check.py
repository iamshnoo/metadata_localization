#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ONE_B_CSV = Path(
    "/path/to/metacul/results/plots/plot8/plot_8_sft_target_split_multiseed_1b_altprompt.csv"
)
DEFAULT_BASELINE_CSV = Path(
    "/path/to/metacul/results/plots/plot8/plot_8_sft_target_split_multiseed.csv"
)

SPLITS = ("Overall", "Explicit", "Ambiguous")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether SFT results satisfy the target-split ordering requirements."
    )
    parser.add_argument("--one-b-csv", type=Path, default=DEFAULT_ONE_B_CSV)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    return parser.parse_args()


def load_accuracy_map(path: Path) -> dict[tuple[str, str], float]:
    df = pd.read_csv(path)
    required_cols = {"series", "split", "accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return {
        (str(row["series"]).strip(), str(row["split"]).strip()): float(row["accuracy"])
        for _, row in df.iterrows()
    }


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def fmt_delta(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def main() -> int:
    args = parse_args()
    if not args.one_b_csv.exists():
        print(f"[warn] missing 1B CSV: {args.one_b_csv}")
        return 0
    if not args.baseline_csv.exists():
        print(f"[warn] missing baseline CSV: {args.baseline_csv}")
        return 0

    one_b = load_accuracy_map(args.one_b_csv)
    baseline = load_accuracy_map(args.baseline_csv)

    rows = []
    all_pass = True
    for split in SPLITS:
        one_b_tplus = one_b[("1B T+/I+", split)]
        one_b_tminus = one_b[("1B T-/I-", split)]
        three_b_tplus = baseline[("3B T+/I+", split)]
        three_b_tminus = baseline[("3B T-/I-", split)]

        one_b_delta = one_b_tplus - one_b_tminus
        three_b_delta = three_b_tplus - three_b_tminus
        size_delta = three_b_tplus - one_b_tplus

        split_pass = one_b_delta > 0 and three_b_delta > 0 and size_delta > 0
        all_pass = all_pass and split_pass

        rows.append(
            {
                "split": split,
                "1B T+/I+": fmt_pct(one_b_tplus),
                "1B T-/I-": fmt_pct(one_b_tminus),
                "1B delta": fmt_delta(one_b_delta),
                "3B T+/I+": fmt_pct(three_b_tplus),
                "3B T-/I-": fmt_pct(three_b_tminus),
                "3B delta": fmt_delta(three_b_delta),
                "3B>1B": fmt_delta(size_delta),
                "pass": split_pass,
            }
        )

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print()
    if all_pass:
        print("[pass] all SFT target-split requirements are satisfied")
    else:
        print("[fail] at least one split violates the SFT target-split requirements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
