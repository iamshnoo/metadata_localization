#!/usr/bin/env python3
import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path("/path/to/metacul")
DEFAULT_RUN_ROOT = ROOT / "results/localnewsqa_gold_20260516"


def load_plot67():
    path = ROOT / "src/67_plot_localnewsqa_accuracy_switch_composite.py"
    spec = importlib.util.spec_from_file_location("plot67", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["plot67"] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the LocalNewsQA gold main accuracy/switch figure."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--accuracy-csv",
        type=Path,
        default=None,
        help="Seed-41 pretrained target accuracy CSV with bootstrap intervals.",
    )
    parser.add_argument(
        "--switch-csv",
        type=Path,
        default=None,
        help="Pretrained locale-switch summary CSV.",
    )
    parser.add_argument(
        "--gain-csv",
        type=Path,
        default=None,
        help="Gold LocalNewsQA metadata-gain long CSV.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--write-latex",
        action="store_true",
        help="Also overwrite the paper figure files under latex/figs/main.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root
    plot_root = args.out_dir or (run_root / "plots")
    plot67 = load_plot67()

    plot67.ACCURACY_CSV = args.accuracy_csv or (
        run_root / "plots/plot_8_pretrained_target_split_seed41_bootstrap.csv"
    )
    plot67.SWITCH_CSV = args.switch_csv or (
        run_root / "summaries/localnewsqa_locale_switch/summary.csv"
    )
    plot67.GAIN_CSV = args.gain_csv or (
        run_root / "appendix_model_gain_tables/localnewsqa_model_gains_long.csv"
    )
    plot67.OUT_DIR = plot_root

    if args.write_latex:
        plot67.LATEX_FIG = ROOT / "latex/figs/main/8_localnewsqa_accuracy_switch_composite.pdf"
        plot67.LATEX_TARGET_PANEL = ROOT / "latex/figs/main/8_localnewsqa_accuracy_target_panel.pdf"
        plot67.LATEX_SWITCH_PANEL = ROOT / "latex/figs/main/8_localnewsqa_accuracy_switch_panel.pdf"
        plot67.LATEX_ONECOL = ROOT / "latex/figs/main/8_localnewsqa_accuracy_switch_onecol.pdf"
    else:
        latex_shadow = plot_root / "latex_shadow"
        plot67.LATEX_FIG = latex_shadow / "8_localnewsqa_accuracy_switch_composite.pdf"
        plot67.LATEX_TARGET_PANEL = latex_shadow / "8_localnewsqa_accuracy_target_panel.pdf"
        plot67.LATEX_SWITCH_PANEL = latex_shadow / "8_localnewsqa_accuracy_switch_panel.pdf"
        plot67.LATEX_ONECOL = latex_shadow / "8_localnewsqa_accuracy_switch_onecol.pdf"

    plot67.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
