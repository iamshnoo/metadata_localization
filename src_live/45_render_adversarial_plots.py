#!/usr/bin/env python3
import importlib.util
import shutil
from pathlib import Path


PLOT19_PATH = Path("/path/to/metacul/src/19_perplexity_plot.py")
PLOT9_DIR = Path("/path/to/metacul/results/plots/plot9")
LATEX_DIR = Path("/path/to/metacul/latex/figs/appendix")


def load_plot_module():
    spec = importlib.util.spec_from_file_location("plot19_module", PLOT19_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_plot_module()
    module.plot_adversarial_url_accuracy()
    module.plot_adversarial_url_accuracy(
        exclude_explicit=True,
        output_name="qa_adversarial_accuracy_noexplicit.pdf",
    )

    copy_specs = [
        ("qa_adversarial_accuracy.pdf", "16_qa_adversarial_accuracy.pdf"),
        ("qa_adversarial_accuracy_noexplicit.pdf", "17_qa_adversarial_accuracy_noexplicit.pdf"),
    ]
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in copy_specs:
        src = PLOT9_DIR / src_name
        dst = LATEX_DIR / dst_name
        if not src.exists():
            raise FileNotFoundError(f"Missing generated adversarial figure: {src}")
        shutil.copy2(src, dst)
        print(f"[ok] Copied {src} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
