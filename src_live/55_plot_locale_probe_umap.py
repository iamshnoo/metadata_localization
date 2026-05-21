#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None

from sklearn.manifold import TSNE


DEFAULT_INPUT_ROOT = Path("/path/to/metacul/results/mechanistic/locale_probe")
DEFAULT_OUTPUT_ROOT = Path("/path/to/metacul/results/mechanistic/locale_probe")
DEFAULT_VARIANTS = {
    "url_final": "URL",
    "country_final": "Country",
    "continent_final": "Continent",
    "none_final": "No metadata",
}
COLORS = {
    "Africa": "#d62728",
    "America": "#1f77b4",
    "Asia": "#2ca02c",
    "Europe": "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 2x2 UMAP/t-SNE panels for locale probe embeddings.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[f"{k}:{v}" for k, v in DEFAULT_VARIANTS.items()],
        help="Variant mapping in the form dir_name:Label",
    )
    return parser.parse_args()


def parse_variants(raw_variants) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in raw_variants:
        name, label = item.split(":", 1)
        mapping[name] = label
    return mapping


def reduce_2d(X: np.ndarray) -> np.ndarray:
    if umap is not None:
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=42,
        )
        return reducer.fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    return tsne.fit_transform(X)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    variants = parse_variants(args.variants)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    axes = axes.flatten()

    for ax, (variant_dir_name, label) in zip(axes, variants.items()):
        npz_path = args.input_root / variant_dir_name / "embeddings.npz"
        if not npz_path.exists():
            ax.set_axis_off()
            ax.set_title(f"{label}\nmissing")
            continue
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"]
        continents = data["continents"]
        coords = reduce_2d(X)
        for continent, color in COLORS.items():
            mask = continents == continent
            if np.any(mask):
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=10,
                    alpha=0.75,
                    color=color,
                    label=continent,
                )
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Locale probe representation geometry", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf_path = args.output_root / "locale_probe_umap_grid.pdf"
    fig.savefig(pdf_path)
    fig.savefig(pdf_path.with_suffix(".png"), dpi=220)
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
