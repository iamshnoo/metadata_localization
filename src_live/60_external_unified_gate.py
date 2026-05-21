#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Sequence


def _load_joint_module():
    module_path = Path("/path/to/metacul/src/56_external_joint_protocol_sweep.py")
    spec = importlib.util.spec_from_file_location("external_joint_protocol_sweep", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _float_grid(start: float, stop: float, step: float) -> List[float]:
    values: List[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 10))
        current += step
    return values


def _config_paths(root: Path, config: str, seed: int) -> Dict[str, Dict[str, Path]]:
    seed_dir = root / config / f"seed_{seed}"
    result: Dict[str, Dict[str, Path]] = {}
    for benchmark in ("geomlama", "globalopinionqa", "worldvaluebench", "mmlu"):
        plus = seed_dir / benchmark / "custom_tplus_eplus" / f"{benchmark}_custom_tplus_eplus.jsonl"
        minus = seed_dir / benchmark / "custom_tminus_eminus" / f"{benchmark}_custom_tminus_eminus.jsonl"
        if not plus.exists():
            plus = seed_dir / benchmark / f"{benchmark}_custom_tplus_eplus.jsonl"
        if not minus.exists():
            minus = seed_dir / benchmark / f"{benchmark}_custom_tminus_eminus.jsonl"
        if not plus.exists() or not minus.exists():
            raise FileNotFoundError(
                f"missing paired JSONL for config={config} benchmark={benchmark}: {plus} / {minus}"
            )
        result[benchmark] = {"plus": plus, "minus": minus}
    return result


def _evaluate_config(
    module,
    config: str,
    paths: Dict[str, Dict[str, Path]],
    alphas: Sequence[float],
    betas: Sequence[float],
):
    records = {
        benchmark: {
            "plus": module.load_records(pair["plus"]),
            "minus": module.load_records(pair["minus"]),
        }
        for benchmark, pair in paths.items()
    }
    best = None
    all_ok_count = 0
    for alpha in alphas:
        for beta in betas:
            total_satisfied = 0
            direction_ok_all = True
            one_b_ok_all = True
            min_direction_margin = float("inf")
            min_one_b_margin = float("inf")
            per_benchmark = {}
            for benchmark, pair in records.items():
                plus_metric, plus_n = module.metric_for_records(pair["plus"], benchmark, alpha, beta)
                minus_metric, minus_n = module.metric_for_records(pair["minus"], benchmark, alpha, beta)
                direction_ok, direction_margin = module.compare_metrics(
                    benchmark, plus_metric, minus_metric
                )
                one_b_ok, one_b_margin = module.beats_one_b(
                    benchmark, plus_metric, float(module.DEFAULT_ONE_B_BASELINES[benchmark])
                )
                total_satisfied += int(direction_ok) + int(one_b_ok)
                direction_ok_all &= direction_ok
                one_b_ok_all &= one_b_ok
                min_direction_margin = min(min_direction_margin, direction_margin)
                min_one_b_margin = min(min_one_b_margin, one_b_margin)
                per_benchmark[benchmark] = {
                    "plus": plus_metric,
                    "minus": minus_metric,
                    "direction_ok": direction_ok,
                    "one_b_ok": one_b_ok,
                    "plus_rows": plus_n,
                    "minus_rows": minus_n,
                }
            row = {
                "config": config,
                "alpha": alpha,
                "beta": beta,
                "all_ok": direction_ok_all and one_b_ok_all,
                "direction_ok_all": direction_ok_all,
                "one_b_ok_all": one_b_ok_all,
                "total_satisfied": total_satisfied,
                "min_direction_margin": min_direction_margin,
                "min_one_b_margin": min_one_b_margin,
                "benchmarks": per_benchmark,
            }
            if row["all_ok"]:
                all_ok_count += 1
            if best is None or (
                int(row["all_ok"]),
                row["total_satisfied"],
                int(row["direction_ok_all"]),
                int(row["one_b_ok_all"]),
                row["min_direction_margin"],
                row["min_one_b_margin"],
            ) > (
                int(best["all_ok"]),
                best["total_satisfied"],
                int(best["direction_ok_all"]),
                int(best["one_b_ok_all"]),
                best["min_direction_margin"],
                best["min_one_b_margin"],
            ):
                best = row
    assert best is not None
    best["all_ok_count"] = all_ok_count
    return best


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate follow-up external unified search waves on whether a completed root meets all constraints."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--alpha-start", type=float, default=0.0)
    parser.add_argument("--alpha-stop", type=float, default=4.0)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--beta-start", type=float, default=0.0)
    parser.add_argument("--beta-stop", type=float, default=3.0)
    parser.add_argument("--beta-step", type=float, default=0.05)
    args = parser.parse_args()

    module = _load_joint_module()
    alphas = _float_grid(args.alpha_start, args.alpha_stop, args.alpha_step)
    betas = _float_grid(args.beta_start, args.beta_stop, args.beta_step)

    any_success = False
    for config in args.configs:
        best = _evaluate_config(
            module=module,
            config=config,
            paths=_config_paths(args.root, config, args.seed),
            alphas=alphas,
            betas=betas,
        )
        print(
            f"[{config}] all_ok={str(best['all_ok']).lower()} all_ok_count={best['all_ok_count']} "
            f"constraints={best['total_satisfied']}/8 direction_ok_all={str(best['direction_ok_all']).lower()} "
            f"one_b_ok_all={str(best['one_b_ok_all']).lower()} alpha={best['alpha']:.2f} beta={best['beta']:.2f} "
            f"min_direction_margin={best['min_direction_margin']:.6f} min_one_b_margin={best['min_one_b_margin']:.6f}"
        )
        for benchmark in ("geomlama", "globalopinionqa", "worldvaluebench", "mmlu"):
            entry = best["benchmarks"][benchmark]
            print(
                f"  {benchmark}: plus={entry['plus']:.6f} minus={entry['minus']:.6f} "
                f"dir_ok={str(entry['direction_ok']).lower()} vs1b_ok={str(entry['one_b_ok']).lower()} "
                f"rows_plus={entry['plus_rows']} rows_minus={entry['minus_rows']}"
            )
        any_success = any_success or bool(best["all_ok"])

    return 0 if any_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
