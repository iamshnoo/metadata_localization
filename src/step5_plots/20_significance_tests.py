#!/usr/bin/env python3
"""
Bootstrap significance tests for plots 3, 4, 5, and 7 using per-sample losses.
"""

import argparse
import hashlib
import json
import os
import re

import numpy as np

DEFAULT_PER_SAMPLE_DIR = "/path/to/metacul/results/per_sample_losses"
DEFAULT_OUTPUT_DIR = "/path/to/metacul/results/significance"


def _slugify(value):
    value = value.strip().strip("/")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value or "unknown"


def per_sample_path(base_dir, model_path, test_set_path, split, max_samples, seed):
    key = f"{model_path}|{test_set_path}|{split}|{max_samples}|{seed}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    model_slug = _slugify(os.path.basename(model_path))
    test_slug = _slugify(os.path.basename(os.path.normpath(test_set_path)))
    filename = (
        f"{model_slug}__{test_slug}__{split}__max{max_samples}__seed{seed}__{digest}.jsonl"
    )
    return os.path.join(base_dir, filename)


def load_losses(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = row.get("index")
            if idx is None:
                continue
            loss_sum = row.get("loss_sum")
            token_count = row.get("token_count")
            if loss_sum is None or token_count is None:
                continue
            data[int(idx)] = (float(loss_sum), float(token_count))
    return data


def paired_bootstrap(loss_a, tok_a, loss_b, tok_b, iters, ci_level, rng):
    loss_a = np.asarray(loss_a, dtype=np.float64)
    tok_a = np.asarray(tok_a, dtype=np.float64)
    loss_b = np.asarray(loss_b, dtype=np.float64)
    tok_b = np.asarray(tok_b, dtype=np.float64)
    n = len(loss_a)
    if n == 0:
        return None

    mean_nll_a = float(loss_a.sum() / tok_a.sum())
    mean_nll_b = float(loss_b.sum() / tok_b.sum())
    mean_ppl_a = float(np.exp(mean_nll_a))
    mean_ppl_b = float(np.exp(mean_nll_b))

    boot_delta_nll = np.empty(iters, dtype=np.float64)
    boot_delta_ppl = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        loss_a_sum = loss_a[idx].sum()
        tok_a_sum = tok_a[idx].sum()
        loss_b_sum = loss_b[idx].sum()
        tok_b_sum = tok_b[idx].sum()
        nll_a = loss_a_sum / tok_a_sum
        nll_b = loss_b_sum / tok_b_sum
        boot_delta_nll[i] = nll_a - nll_b
        boot_delta_ppl[i] = np.exp(nll_a) - np.exp(nll_b)

    alpha = (1.0 - ci_level) / 2.0
    ci_low_nll = float(np.quantile(boot_delta_nll, alpha))
    ci_high_nll = float(np.quantile(boot_delta_nll, 1.0 - alpha))
    ci_low_ppl = float(np.quantile(boot_delta_ppl, alpha))
    ci_high_ppl = float(np.quantile(boot_delta_ppl, 1.0 - alpha))
    p_left = float(np.mean(boot_delta_nll <= 0))
    p_right = float(np.mean(boot_delta_nll >= 0))
    p_value = float(2.0 * min(p_left, p_right))

    return {
        "n_samples": int(n),
        "mean_nll_a": mean_nll_a,
        "mean_nll_b": mean_nll_b,
        "mean_ppl_a": mean_ppl_a,
        "mean_ppl_b": mean_ppl_b,
        "delta_nll": mean_nll_a - mean_nll_b,
        "delta_nll_ci_low": ci_low_nll,
        "delta_nll_ci_high": ci_high_nll,
        "delta_ppl": mean_ppl_a - mean_ppl_b,
        "delta_ppl_ci_low": ci_low_ppl,
        "delta_ppl_ci_high": ci_high_ppl,
        "p_value": p_value,
    }


def independent_bootstrap(loss_a, tok_a, loss_b, tok_b, iters, ci_level, rng):
    loss_a = np.asarray(loss_a, dtype=np.float64)
    tok_a = np.asarray(tok_a, dtype=np.float64)
    loss_b = np.asarray(loss_b, dtype=np.float64)
    tok_b = np.asarray(tok_b, dtype=np.float64)
    n_a = len(loss_a)
    n_b = len(loss_b)
    if n_a == 0 or n_b == 0:
        return None

    mean_nll_a = float(loss_a.sum() / tok_a.sum())
    mean_nll_b = float(loss_b.sum() / tok_b.sum())
    mean_ppl_a = float(np.exp(mean_nll_a))
    mean_ppl_b = float(np.exp(mean_nll_b))

    boot_delta_nll = np.empty(iters, dtype=np.float64)
    boot_delta_ppl = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        nll_a = loss_a[idx_a].sum() / tok_a[idx_a].sum()
        nll_b = loss_b[idx_b].sum() / tok_b[idx_b].sum()
        boot_delta_nll[i] = nll_a - nll_b
        boot_delta_ppl[i] = np.exp(nll_a) - np.exp(nll_b)

    alpha = (1.0 - ci_level) / 2.0
    ci_low_nll = float(np.quantile(boot_delta_nll, alpha))
    ci_high_nll = float(np.quantile(boot_delta_nll, 1.0 - alpha))
    ci_low_ppl = float(np.quantile(boot_delta_ppl, alpha))
    ci_high_ppl = float(np.quantile(boot_delta_ppl, 1.0 - alpha))
    p_left = float(np.mean(boot_delta_nll <= 0))
    p_right = float(np.mean(boot_delta_nll >= 0))
    p_value = float(2.0 * min(p_left, p_right))

    return {
        "n_samples_a": int(n_a),
        "n_samples_b": int(n_b),
        "mean_nll_a": mean_nll_a,
        "mean_nll_b": mean_nll_b,
        "mean_ppl_a": mean_ppl_a,
        "mean_ppl_b": mean_ppl_b,
        "delta_nll": mean_nll_a - mean_nll_b,
        "delta_nll_ci_low": ci_low_nll,
        "delta_nll_ci_high": ci_high_nll,
        "delta_ppl": mean_ppl_a - mean_ppl_b,
        "delta_ppl_ci_low": ci_low_ppl,
        "delta_ppl_ci_high": ci_high_ppl,
        "p_value": p_value,
    }


def compute_pair(model_a, model_b, test_path, args, rng):
    path_a = per_sample_path(
        args.per_sample_dir,
        model_a,
        test_path,
        args.split,
        args.max_samples,
        args.seed,
    )
    path_b = per_sample_path(
        args.per_sample_dir,
        model_b,
        test_path,
        args.split,
        args.max_samples,
        args.seed,
    )
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        return {
            "status": "missing",
            "missing_a": str(not os.path.exists(path_a)),
            "missing_b": str(not os.path.exists(path_b)),
        }
    data_a = load_losses(path_a)
    data_b = load_losses(path_b)
    common = sorted(set(data_a.keys()) & set(data_b.keys()))
    if not common:
        return {"status": "no_overlap"}
    loss_a = [data_a[i][0] for i in common]
    tok_a = [data_a[i][1] for i in common]
    loss_b = [data_b[i][0] for i in common]
    tok_b = [data_b[i][1] for i in common]
    stats = paired_bootstrap(
        loss_a, tok_a, loss_b, tok_b, args.bootstrap_iters, args.ci_level, rng
    )
    if stats is None:
        return {"status": "empty"}
    stats["status"] = "ok"
    return stats


def plot3_rows(args, rng):
    rows = []
    regions = ["africa", "america", "asia", "europe", "combined"]
    meta_order = ["with_metadata", "without_metadata"]
    for meta in meta_order:
        for region in regions:
            model_a = f"/path/to/metacul/models/combined_{meta}_1b"
            model_b = f"/path/to/metacul/models/combined_{meta}_500m"
            if region == "combined":
                test_path = (
                    f"/path/to/metacul/training_data/meco_datasets/combined/{meta}/"
                )
            else:
                test_path = (
                    f"/path/to/metacul/training_data/meco_datasets/continents/{region}/{meta}/"
                )
            stats = compute_pair(model_a, model_b, test_path, args, rng)
            row = {
                "plot": "plot3",
                "meta": meta,
                "region": region,
                "test_path": test_path,
                "model_a": model_a,
                "model_b": model_b,
            }
            row.update(stats)
            rows.append(row)
    return rows


def plot4_rows(args, rng):
    rows = []
    continents = ["africa", "america", "asia", "europe"]
    sizes = ["500m", "1b"]
    test_metas = ["with_metadata", "without_metadata"]
    for size in sizes:
        for train in continents:
            for test in continents:
                for test_meta in test_metas:
                    model_a = (
                        f"/path/to/metacul/models/{train}_without_metadata_{size}"
                    )
                    model_b = (
                        f"/path/to/metacul/models/{train}_with_metadata_{size}"
                    )
                    test_path = (
                        f"/path/to/metacul/training_data/meco_datasets/continents/{test}/{test_meta}/"
                    )
                    stats = compute_pair(model_a, model_b, test_path, args, rng)
                    row = {
                        "plot": "plot4",
                        "size": size,
                        "train_continent": train,
                        "test_continent": test,
                        "test_meta": test_meta,
                        "test_path": test_path,
                        "model_a": model_a,
                        "model_b": model_b,
                    }
                    row.update(stats)
                    rows.append(row)
    return rows


def plot5_rows(args, rng):
    rows = []
    continents = ["africa", "america", "asia", "europe"]
    sizes = ["500m", "1b"]
    for size in sizes:
        for train in continents:
            for test in continents:
                model_a = f"/path/to/metacul/models/{train}_with_metadata_{size}"
                model_b = f"/path/to/metacul/models/{test}_with_metadata_{size}"
                test_path_a = (
                    f"/path/to/metacul/training_data/meco_datasets/continents/{test}/with_metadata/"
                )
                test_path_b = (
                    f"/path/to/metacul/training_data/meco_datasets/continents/{train}/with_metadata/"
                )
                path_a = per_sample_path(
                    args.per_sample_dir,
                    model_a,
                    test_path_a,
                    args.split,
                    args.max_samples,
                    args.seed,
                )
                path_b = per_sample_path(
                    args.per_sample_dir,
                    model_b,
                    test_path_b,
                    args.split,
                    args.max_samples,
                    args.seed,
                )
                if not os.path.exists(path_a) or not os.path.exists(path_b):
                    stats = {
                        "status": "missing",
                        "missing_a": str(not os.path.exists(path_a)),
                        "missing_b": str(not os.path.exists(path_b)),
                    }
                else:
                    data_a = load_losses(path_a)
                    data_b = load_losses(path_b)
                    loss_a = [v[0] for v in data_a.values()]
                    tok_a = [v[1] for v in data_a.values()]
                    loss_b = [v[0] for v in data_b.values()]
                    tok_b = [v[1] for v in data_b.values()]
                    stats = independent_bootstrap(
                        loss_a, tok_a, loss_b, tok_b, args.bootstrap_iters, args.ci_level, rng
                    )
                    if stats is None:
                        stats = {"status": "empty"}
                    else:
                        stats["status"] = "ok"

                row = {
                    "plot": "plot5",
                    "size": size,
                    "train_continent": train,
                    "test_continent": test,
                    "test_path_a": test_path_a,
                    "test_path_b": test_path_b,
                    "model_a": model_a,
                    "model_b": model_b,
                }
                row.update(stats)
                rows.append(row)
    return rows


def plot7_rows(args, rng):
    rows = []
    continents = ["africa", "america", "asia", "europe"]
    steps = [2000, 4000, 8000, 10000]
    metas = ["with_metadata", "without_metadata"]
    for meta in metas:
        for step in steps:
            if step == 10000:
                combined_model = f"/path/to/metacul/models/combined_{meta}_1b"
            else:
                combined_model = (
                    "/path/to/metacul/models/ablation_intermediates/metadata/"
                    f"combined_{meta}_1b_step{step // 1000}k"
                )
            for cont in continents:
                if step == 10000:
                    loo_model = (
                        "/path/to/metacul/models/ablations/leave_one_out/"
                        f"combined_no_{cont}_{meta}_1b"
                    )
                else:
                    loo_model = (
                        "/path/to/metacul/models/ablation_intermediates/leave_one_out/"
                        f"combined_no_{cont}_{meta}_1b_step{step // 1000}k"
                    )
                left_out_test = (
                    f"/path/to/metacul/training_data/meco_datasets/continents/{cont}/{meta}/"
                )
                stats_left = compute_pair(loo_model, combined_model, left_out_test, args, rng)
                row_left = {
                    "plot": "plot7",
                    "meta": meta,
                    "step": step,
                    "continent": cont,
                    "test_scope": "left_out",
                    "test_path": left_out_test,
                    "model_a": loo_model,
                    "model_b": combined_model,
                }
                row_left.update(stats_left)
                rows.append(row_left)

                all_test = (
                    f"/path/to/metacul/training_data/meco_datasets/combined/{meta}/"
                )
                stats_all = compute_pair(loo_model, combined_model, all_test, args, rng)
                row_all = {
                    "plot": "plot7",
                    "meta": meta,
                    "step": step,
                    "continent": cont,
                    "test_scope": "all",
                    "test_path": all_test,
                    "model_a": loo_model,
                    "model_b": combined_model,
                }
                row_all.update(stats_all)
                rows.append(row_all)
    return rows


def plot4_avg_by_test(args, rng):
    rows = []
    continents = ["africa", "america", "asia", "europe"]
    sizes = ["500m", "1b"]
    test_metas = ["with_metadata", "without_metadata"]

    for size in sizes:
        for test_cont in continents:
            for test_meta in test_metas:
                deltas_loss = []
                deltas_tok = []
                for train_cont in continents:
                    model_a = (
                        f"/path/to/metacul/models/{train_cont}_without_metadata_{size}"
                    )
                    model_b = (
                        f"/path/to/metacul/models/{train_cont}_with_metadata_{size}"
                    )
                    test_path = (
                        f"/path/to/metacul/training_data/meco_datasets/continents/{test_cont}/{test_meta}/"
                    )
                    path_a = per_sample_path(
                        args.per_sample_dir,
                        model_a,
                        test_path,
                        args.split,
                        args.max_samples,
                        args.seed,
                    )
                    path_b = per_sample_path(
                        args.per_sample_dir,
                        model_b,
                        test_path,
                        args.split,
                        args.max_samples,
                        args.seed,
                    )
                    if not os.path.exists(path_a) or not os.path.exists(path_b):
                        continue
                    data_a = load_losses(path_a)
                    data_b = load_losses(path_b)
                    common = sorted(set(data_a.keys()) & set(data_b.keys()))
                    if not common:
                        continue
                    for idx in common:
                        loss_a, tok_a = data_a[idx]
                        loss_b, tok_b = data_b[idx]
                        deltas_loss.append(loss_a - loss_b)
                        deltas_tok.append(tok_a)

                if not deltas_loss:
                    stats = {"status": "missing"}
                else:
                    deltas_loss = np.asarray(deltas_loss, dtype=np.float64)
                    deltas_tok = np.asarray(deltas_tok, dtype=np.float64)
                    n = len(deltas_loss)
                    boot = np.empty(args.bootstrap_iters, dtype=np.float64)
                    for i in range(args.bootstrap_iters):
                        idx = rng.integers(0, n, size=n)
                        boot[i] = deltas_loss[idx].sum() / deltas_tok[idx].sum()
                    alpha = (1.0 - args.ci_level) / 2.0
                    ci_low = float(np.quantile(boot, alpha))
                    ci_high = float(np.quantile(boot, 1.0 - alpha))
                    mean_delta = float(deltas_loss.sum() / deltas_tok.sum())
                    p_left = float(np.mean(boot <= 0))
                    p_right = float(np.mean(boot >= 0))
                    p_value = float(2.0 * min(p_left, p_right))
                    stats = {
                        "status": "ok",
                        "n_samples": int(n),
                        "delta_nll": mean_delta,
                        "delta_nll_ci_low": ci_low,
                        "delta_nll_ci_high": ci_high,
                        "p_value": p_value,
                    }

                row = {
                    "plot": "plot4_avg_by_test",
                    "size": size,
                    "test_continent": test_cont,
                    "test_meta": test_meta,
                }
                row.update(stats)
                rows.append(row)

    return rows


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Paired bootstrap significance tests for plot differences."
    )
    parser.add_argument("--per-sample-dir", default=DEFAULT_PER_SAMPLE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument(
        "--plots",
        default="3,4,7",
        help="Comma-separated list of plots to compute (3,4,5,7).",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    plots = {p.strip() for p in args.plots.split(",") if p.strip()}

    if "3" in plots:
        rows = plot3_rows(args, rng)
        write_csv(os.path.join(args.output_dir, "plot3.csv"), rows)
    if "4" in plots:
        rows = plot4_rows(args, rng)
        write_csv(os.path.join(args.output_dir, "plot4.csv"), rows)
        avg_rows = plot4_avg_by_test(args, rng)
        write_csv(os.path.join(args.output_dir, "plot4_avg_by_test.csv"), avg_rows)
    if "5" in plots:
        rows = plot5_rows(args, rng)
        write_csv(os.path.join(args.output_dir, "plot5.csv"), rows)
    if "7" in plots:
        rows = plot7_rows(args, rng)
        write_csv(os.path.join(args.output_dir, "plot7.csv"), rows)


if __name__ == "__main__":
    main()
