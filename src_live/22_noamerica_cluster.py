#!/usr/bin/env python3
"""
Cluster the combined/with_metadata test split (NoAmerica analysis) using
sentence embeddings built from the same tokenizer as 18_perplexity_eval.py.
"""

import argparse
import hashlib
import json
import os
import re

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required for clustering. Install it in your env."
    ) from exc


DEFAULT_DATASET = "/scratch/amukher6/metacul/training_data/meco_datasets/combined/with_metadata/"
DEFAULT_OUTPUT_DIR = "/scratch/amukher6/metacul/results/noamerica_cluster"
DEFAULT_MODEL = "/scratch/amukher6/metacul/models/ablations/leave_one_out/combined_no_america_with_metadata_1b"
DEFAULT_TOKENIZER = "meta-llama/Llama-3.2-1B"
DEFAULT_COMBINED_MODEL = "/scratch/amukher6/metacul/models/combined_with_metadata_1b"
DEFAULT_LOO_MODEL_BASE = "/scratch/amukher6/metacul/models/ablations/leave_one_out"
DEFAULT_CONTINENT_TEST_BASE = "/scratch/amukher6/metacul/training_data/meco_datasets/continents"
DEFAULT_PER_SAMPLE_DIR = "/scratch/amukher6/metacul/results/per_sample_losses"


def load_split(dataset_path, split):
    split_path = f"{dataset_path.rstrip('/')}/{split}"
    if os.path.exists(split_path):
        return load_from_disk(split_path)
    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, dict) and split in dataset:
        return dataset[split]
    if hasattr(dataset, "keys") and split in dataset:
        return dataset[split]
    return dataset


def sample_indices(total, max_samples, rng):
    if max_samples is None or total <= max_samples:
        return list(range(total))
    return rng.choice(total, size=max_samples, replace=False).tolist()


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    masked = last_hidden * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


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
            loss_sum = row.get("loss_sum")
            token_count = row.get("token_count")
            if idx is None or loss_sum is None or token_count is None:
                continue
            data[int(idx)] = (float(loss_sum), float(token_count))
    return data


def _cluster_ppl(loss_map, cluster_to_indices):
    result = {}
    for cluster, indices in cluster_to_indices.items():
        loss_sum = 0.0
        tok_sum = 0.0
        used = 0
        for idx in indices:
            if idx not in loss_map:
                continue
            loss, toks = loss_map[idx]
            loss_sum += loss
            tok_sum += toks
            used += 1
        if tok_sum == 0.0:
            result[cluster] = {
                "n": 0,
                "mean_nll": None,
                "mean_ppl": None,
            }
        else:
            mean_nll = loss_sum / tok_sum
            result[cluster] = {
                "n": used,
                "mean_nll": float(mean_nll),
                "mean_ppl": float(np.exp(mean_nll)),
            }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Cluster combined test set for NoAmerica analysis."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--per-sample-dir", default=DEFAULT_PER_SAMPLE_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing outputs (do not skip if summary.json exists).",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        default=False,
        help="Run clustering for combined and leave-one-out models across specified test sets.",
    )
    parser.add_argument(
        "--combined-model",
        default=DEFAULT_COMBINED_MODEL,
        help="Combined with-metadata model path for run-all.",
    )
    parser.add_argument(
        "--loo-model-base",
        default=DEFAULT_LOO_MODEL_BASE,
        help="Base directory for leave-one-out models for run-all.",
    )
    parser.add_argument(
        "--continent-test-base",
        default=DEFAULT_CONTINENT_TEST_BASE,
        help="Base directory for continent test sets for run-all.",
    )
    parser.add_argument(
        "--baseline-model",
        default=DEFAULT_COMBINED_MODEL,
        help="Baseline model to compare against for per-cluster deltas.",
    )
    parser.add_argument(
        "--update-summary-only",
        action="store_true",
        default=False,
        help="Rebuild summary.json (including per-cluster deltas) using existing cluster outputs.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=False,
        help="Force tokenizer/model loading to use only local files.",
    )
    args = parser.parse_args()

    def run_single(dataset_path, model_path, output_dir, rng):
        summary_path = os.path.join(output_dir, "summary.json")
        if os.path.exists(summary_path) and not args.force and not args.update_summary_only:
            return
        if args.update_summary_only:
            cluster_assignments = os.path.join(output_dir, "cluster_assignments.jsonl")
            dropped_path = os.path.join(output_dir, "dropped_samples.jsonl")
            if not os.path.exists(cluster_assignments):
                raise SystemExit(
                    f"Missing {cluster_assignments}; cannot update summary only."
                )
            dataset = load_split(dataset_path, args.split)

            assigned = {}
            with open(cluster_assignments, "r") as f:
                for line in f:
                    row = json.loads(line)
                    idx = row.get("index")
                    label = row.get("cluster")
                    if idx is None or label is None:
                        continue
                    assigned[int(idx)] = int(label)

            cluster_to_indices = {}
            continent_counts = {}
            for idx, label in assigned.items():
                cluster_to_indices.setdefault(label, []).append(idx)
                text = dataset[int(idx)]["text"]
                continent = None
                for line in text.splitlines():
                    if line.startswith("CONTINENT:"):
                        continent = line.split("CONTINENT:", 1)[1].strip()
                        break
                continent = continent or "unknown"
                cluster_key = str(label)
                continent_counts.setdefault(cluster_key, {})
                continent_counts[cluster_key][continent] = (
                    continent_counts[cluster_key].get(continent, 0) + 1
                )

            dropped_samples = []
            if os.path.exists(dropped_path):
                with open(dropped_path, "r") as f:
                    for line in f:
                        row = json.loads(line)
                        idx = row.get("index")
                        if idx is None:
                            continue
                        dropped_samples.append(int(idx))

            labels = list(assigned.values())
            cluster_sizes = {
                str(i): int((np.array(labels) == i).sum())
                for i in sorted(set(labels))
            }

            per_cluster = {}
            delta_by_cluster = {}
            if args.per_sample_dir and args.baseline_model and model_path != args.baseline_model:
                model_path_a = per_sample_path(
                    args.per_sample_dir,
                    model_path,
                    dataset_path,
                    args.split,
                    args.max_samples,
                    args.seed,
                )
                model_path_b = per_sample_path(
                    args.per_sample_dir,
                    args.baseline_model,
                    dataset_path,
                    args.split,
                    args.max_samples,
                    args.seed,
                )
                if os.path.exists(model_path_a) and os.path.exists(model_path_b):
                    loss_a = load_losses(model_path_a)
                    loss_b = load_losses(model_path_b)
                    per_cluster_a = _cluster_ppl(loss_a, cluster_to_indices)
                    per_cluster_b = _cluster_ppl(loss_b, cluster_to_indices)
                    for cluster in sorted(cluster_to_indices.keys()):
                        key = str(cluster)
                        a = per_cluster_a.get(cluster, {})
                        b = per_cluster_b.get(cluster, {})
                        if a.get("mean_nll") is None or b.get("mean_nll") is None:
                            delta_by_cluster[key] = {
                                "delta_nll": None,
                                "delta_ppl": None,
                                "n_a": a.get("n", 0),
                                "n_b": b.get("n", 0),
                            }
                            continue
                        delta_nll = a["mean_nll"] - b["mean_nll"]
                        delta_ppl = a["mean_ppl"] - b["mean_ppl"]
                        delta_by_cluster[key] = {
                            "delta_nll": float(delta_nll),
                            "delta_ppl": float(delta_ppl),
                            "n_a": a.get("n", 0),
                            "n_b": b.get("n", 0),
                        }
                    per_cluster = {
                        "model": {str(k): v for k, v in per_cluster_a.items()},
                        "baseline": {str(k): v for k, v in per_cluster_b.items()},
                    }

            summary = {
                "n_samples": int(len(labels)),
                "n_clusters": int(len(cluster_sizes)),
                "silhouette": None,
                "cluster_sizes": cluster_sizes,
                "skipped_nonfinite": int(len(dropped_samples)),
                "continent_distribution": continent_counts,
                "per_cluster_ppl": per_cluster or None,
                "per_cluster_delta": delta_by_cluster or None,
                "dataset": dataset_path,
                "split": args.split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "model_path": model_path,
                "tokenizer_path": args.tokenizer_path,
                "baseline_model_path": args.baseline_model,
                "per_sample_dir": args.per_sample_dir,
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            return
        dataset = load_split(dataset_path, args.split)
        indices = sample_indices(len(dataset), args.max_samples, rng)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_path,
                local_files_only=True,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_path,
                local_files_only=args.local_files_only,
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            device_map=args.device,
            local_files_only=True,
        )
        model.eval()

        embeddings = []
        text_index = []
        skipped_nonfinite = 0
        dropped_samples = []
        for start in tqdm(range(0, len(indices), args.batch_size), desc="Embedding"):
            batch_idx = indices[start : start + args.batch_size]
            texts = [dataset[int(i)]["text"] for i in batch_idx]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                padding=True,
            )
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            last_hidden = outputs.hidden_states[-1].float()
            pooled = mean_pool(last_hidden, attention_mask).float()
            finite_mask = torch.isfinite(pooled).all(dim=1)
            if not finite_mask.all():
                skipped = (~finite_mask).sum().item()
                skipped_nonfinite += int(skipped)
            pooled = pooled[finite_mask]
            kept_idx = [idx for idx, keep in zip(batch_idx, finite_mask.tolist()) if keep]
            for idx, keep in zip(batch_idx, finite_mask.tolist()):
                if not keep:
                    dropped_samples.append(
                        {"index": int(idx), "reason": "non_finite_embedding"}
                    )
            if pooled.numel() > 0:
                embeddings.append(pooled.cpu().numpy())
                text_index.extend(kept_idx)

        if not embeddings:
            raise SystemExit("No finite embeddings were produced; cannot cluster.")
        X = np.vstack(embeddings)
        kmeans = KMeans(n_clusters=args.clusters, random_state=args.seed, n_init="auto")
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")

        os.makedirs(output_dir, exist_ok=True)
        cluster_path = os.path.join(output_dir, "clusters.jsonl")
        with open(cluster_path, "w") as f:
            for idx, label in zip(text_index, labels):
                text = dataset[int(idx)]["text"]
                f.write(
                    json.dumps({"index": int(idx), "cluster": int(label), "text": text})
                    + "\n"
                )

        dropped_path = os.path.join(output_dir, "dropped_samples.jsonl")
        with open(dropped_path, "w") as f:
            for row in dropped_samples:
                row = dict(row)
                row["text"] = dataset[int(row["index"])]["text"]
                f.write(json.dumps(row) + "\n")
        out_path = os.path.join(output_dir, "cluster_assignments.jsonl")
        with open(out_path, "w") as f:
            for idx, label in zip(text_index, labels):
                f.write(json.dumps({"index": int(idx), "cluster": int(label)}) + "\n")

        continent_counts = {}
        for idx, label in zip(text_index, labels):
            text = dataset[int(idx)]["text"]
            continent = None
            for line in text.splitlines():
                if line.startswith("CONTINENT:"):
                    continent = line.split("CONTINENT:", 1)[1].strip()
                    break
            continent = continent or "unknown"
            cluster_key = str(label)
            continent_counts.setdefault(cluster_key, {})
            continent_counts[cluster_key][continent] = (
                continent_counts[cluster_key].get(continent, 0) + 1
            )

        cluster_to_indices = {}
        for idx, label in zip(text_index, labels):
            cluster_to_indices.setdefault(int(label), []).append(int(idx))

        per_cluster = {}
        delta_by_cluster = {}
        if args.per_sample_dir and args.baseline_model and model_path != args.baseline_model:
            model_path_a = per_sample_path(
                args.per_sample_dir,
                model_path,
                dataset_path,
                args.split,
                args.max_samples,
                args.seed,
            )
            model_path_b = per_sample_path(
                args.per_sample_dir,
                args.baseline_model,
                dataset_path,
                args.split,
                args.max_samples,
                args.seed,
            )
            if os.path.exists(model_path_a) and os.path.exists(model_path_b):
                loss_a = load_losses(model_path_a)
                loss_b = load_losses(model_path_b)
                per_cluster_a = _cluster_ppl(loss_a, cluster_to_indices)
                per_cluster_b = _cluster_ppl(loss_b, cluster_to_indices)
                for cluster in sorted(cluster_to_indices.keys()):
                    key = str(cluster)
                    a = per_cluster_a.get(cluster, {})
                    b = per_cluster_b.get(cluster, {})
                    if a.get("mean_nll") is None or b.get("mean_nll") is None:
                        delta_by_cluster[key] = {
                            "delta_nll": None,
                            "delta_ppl": None,
                            "n_a": a.get("n", 0),
                            "n_b": b.get("n", 0),
                        }
                        continue
                    delta_nll = a["mean_nll"] - b["mean_nll"]
                    delta_ppl = a["mean_ppl"] - b["mean_ppl"]
                    delta_by_cluster[key] = {
                        "delta_nll": float(delta_nll),
                        "delta_ppl": float(delta_ppl),
                        "n_a": a.get("n", 0),
                        "n_b": b.get("n", 0),
                    }
                per_cluster = {
                    "model": {str(k): v for k, v in per_cluster_a.items()},
                    "baseline": {str(k): v for k, v in per_cluster_b.items()},
                }

        summary = {
            "n_samples": int(len(labels)),
            "n_clusters": int(args.clusters),
            "silhouette": float(sil),
            "cluster_sizes": {str(i): int((labels == i).sum()) for i in range(args.clusters)},
            "skipped_nonfinite": int(skipped_nonfinite),
            "continent_distribution": continent_counts,
            "per_cluster_ppl": per_cluster or None,
            "per_cluster_delta": delta_by_cluster or None,
            "dataset": dataset_path,
            "split": args.split,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "model_path": model_path,
            "tokenizer_path": args.tokenizer_path,
            "baseline_model_path": args.baseline_model,
            "per_sample_dir": args.per_sample_dir,
        }
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    rng = np.random.default_rng(args.seed)
    if not args.run_all:
        run_single(args.dataset, args.model_path, args.output_dir, rng)
        return

    models = {
        "combined": args.combined_model,
        "no_africa": os.path.join(args.loo_model_base, "combined_no_africa_with_metadata_1b"),
        "no_asia": os.path.join(args.loo_model_base, "combined_no_asia_with_metadata_1b"),
        "no_europe": os.path.join(args.loo_model_base, "combined_no_europe_with_metadata_1b"),
        "no_america": os.path.join(args.loo_model_base, "combined_no_america_with_metadata_1b"),
    }
    tests = {
        "combined": args.dataset,
        "africa": os.path.join(args.continent_test_base, "africa/with_metadata/"),
        "asia": os.path.join(args.continent_test_base, "asia/with_metadata/"),
        "europe": os.path.join(args.continent_test_base, "europe/with_metadata/"),
        "america": os.path.join(args.continent_test_base, "america/with_metadata/"),
    }

    run_matrix = {
        "combined": ["combined", "africa", "asia", "europe", "america"],
        "no_africa": ["combined", "africa"],
        "no_america": ["combined", "america"],
        "no_asia": ["combined", "asia"],
        "no_europe": ["combined", "europe"],
    }

    for model_name, test_names in run_matrix.items():
        model_path = models[model_name]
        for test_name in test_names:
            dataset_path = tests[test_name]
            output_dir = os.path.join(args.output_dir, model_name, test_name)
            run_single(dataset_path, model_path, output_dir, rng)


if __name__ == "__main__":
    main()
