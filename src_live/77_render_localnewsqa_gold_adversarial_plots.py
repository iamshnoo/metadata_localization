#!/usr/bin/env python3
import importlib.util
import os
import shutil
from pathlib import Path


ROOT = Path("/path/to/metacul")
PLOT19_PATH = ROOT / "src/19_perplexity_plot.py"
RUN_ROOT = ROOT / "results/localnewsqa_gold_20260516"
SUMMARY_PATH = RUN_ROOT / "plots/adversarial_pretrained_summary_full_mismatch.csv"
PLOT_DIR = RUN_ROOT / "plots/adversarial"
LATEX_DIR = ROOT / "latex/figs/appendix"


def load_plot_module():
    spec = importlib.util.spec_from_file_location("plot19_module", PLOT19_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing gold adversarial summary: {SUMMARY_PATH}")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["ADVERSARIAL_PRETRAINED_SUMMARY_PATH"] = str(SUMMARY_PATH)
    os.environ["ADVERSARIAL_PLOT_OUTPUT_DIR"] = str(PLOT_DIR)

    module = load_plot_module()
    module.plot_adversarial_url_accuracy()
    module.plot_adversarial_url_accuracy(
        exclude_explicit=True,
        output_name="qa_adversarial_accuracy_noexplicit.pdf",
    )

    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in [
        ("qa_adversarial_accuracy.pdf", "16_qa_adversarial_accuracy.pdf"),
        ("qa_adversarial_accuracy_noexplicit.pdf", "17_qa_adversarial_accuracy_noexplicit.pdf"),
    ]:
        src = PLOT_DIR / src_name
        dst = LATEX_DIR / dst_name
        if not src.exists():
            raise FileNotFoundError(f"Missing generated adversarial figure: {src}")
        shutil.copy2(src, dst)
        print(f"[ok] copied {src} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
