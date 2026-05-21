#!/usr/bin/env python3
import argparse
import importlib.util
import math
import shlex
import subprocess
from itertools import cycle
from pathlib import Path


ROOT = Path("/path/to/metacul")
SUBMITTER = ROOT / "slurm/submit_final_benchmark_matrix.py"
WORKER = ROOT / "slurm/priority_dataset_worker_fullgpu.slurm"
WRAPPER = ROOT / "slurm/pretrained_external_eval_single.slurm"
LOG_DIR = Path("/path/to/logs/slurm_logs")
DEFAULT_TABLE8_FAMILIES = [
    "maple_1b",
    "maple_3b",
    "llama32_1b",
    "llama32_3b",
    "gemma4_e2b",
    "gemma4_e4b",
]


def load_submitter_module():
    spec = importlib.util.spec_from_file_location("submit_final_benchmark_matrix", SUBMITTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def shell_exports(env: dict[str, str]) -> str:
    parts = ["env"]
    for key, value in sorted(env.items()):
        parts.append(f"{key}={shlex.quote(str(value))}")
    return " ".join(parts)


def completed_job_names(dataset: str) -> set[str]:
    cmd = (
        f"sacct -S 2026-05-02 -u USER_NAME -n -P --format=JobName,State "
        f"| rg '^bm-{dataset}-'"
    )
    out = subprocess.check_output(["bash", "-lc", cmd], text=True)
    done: set[str] = set()
    for line in out.splitlines():
        if not line:
            continue
        name, state = line.split("|", 1)
        if state.startswith("COMPLETED"):
            done.add(name)
    return done


def pending_dataset_job_ids(dataset: str) -> list[str]:
    cmd = f"squeue -u USER_NAME -h -o '%i|%j|%T' | rg '^.*\\|bm-{dataset}-'"
    out = subprocess.check_output(["bash", "-lc", cmd], text=True)
    ids: list[str] = []
    for line in out.splitlines():
        if not line:
            continue
        job_id, _name, state = line.split("|")
        if state == "PENDING":
            ids.append(job_id)
    return ids


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit full-GPU worker chunks for one benchmark dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset key, e.g. blend.")
    parser.add_argument("--families", default=",".join(DEFAULT_TABLE8_FAMILIES))
    parser.add_argument("--tracks", default="base")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cancel-old-pending", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_submitter_module()

    family_filter = {x.strip() for x in args.families.split(",") if x.strip()}
    track_filter = {x.strip() for x in args.tracks.split(",") if x.strip()}
    rows = mod.load_rows(
        only_datasets={args.dataset},
        only_families=family_filter or None,
        only_tracks=track_filter or None,
    )
    done = completed_job_names(args.dataset)

    jobs: list[dict] = []
    for row in rows:
        for seed in mod.DEFAULT_SEEDS:
            job = mod.external_benchmark_job(row, seed)
            if job["job_name"] in done:
                continue
            jobs.append(job)

    jobs.sort(key=lambda j: j["job_name"])
    if not jobs:
        print(f"No remaining {args.dataset} jobs to submit.")
        return

    if args.cancel_old_pending:
        old_pending = pending_dataset_job_ids(args.dataset)
        for chunk in chunked(old_pending, 200):
            subprocess.run(["scancel", *chunk], check=False)

    ts = subprocess.check_output(["date", "+%Y%m%d_%H%M%S"], text=True).strip()
    run_dir = LOG_DIR / f"{args.dataset}_priority_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "manifest.tsv"

    worker_count = min(args.workers, len(jobs))
    per_worker = math.ceil(len(jobs) / worker_count)
    chunks = list(chunked(jobs, per_worker))

    with manifest.open("w") as mf:
        mf.write("worker_index\tjob_name\tbenchmark\tseed\tvariant\tout_root\n")
        for worker_index, jobs_chunk in enumerate(chunks, start=1):
            chunk_script = run_dir / f"worker_{worker_index:02d}.sh"
            lines = [
                "#!/bin/bash",
                "set -euo pipefail",
                "status=0",
            ]
            for idx, job in enumerate(jobs_chunk, start=1):
                env = job["env"]
                lines.append(f"echo '[{idx}/{len(jobs_chunk)}] {job['job_name']}'")
                lines.append(f"if ! {shell_exports(env)} bash {shlex.quote(str(WRAPPER))}; then")
                lines.append(f"  echo 'FAILED {job['job_name']}'")
                lines.append("  status=1")
                lines.append("fi")
                mf.write(
                    "\t".join(
                        [
                            str(worker_index),
                            job["job_name"],
                            env["BENCHMARK"],
                            env["EVAL_SEED"],
                            env["VARIANT"],
                            env["OUT_ROOT"],
                        ]
                    )
                    + "\n"
                )
            lines.append("exit $status")
            chunk_script.write_text("\n".join(lines) + "\n")
            chunk_script.chmod(0o755)

    lanes = cycle(
        [
            ("SLURM_ACCOUNT", "contrib-gpuq", "gpu"),
            ("cs678sp23", "contrib-gpuq", "gpu"),
            ("SLURM_ACCOUNT", "gpuq", "gpu"),
            ("cs678sp23", "gpuq", "gpu"),
        ]
    )

    submit_ids: list[tuple[int, str, str, str, str, str]] = []
    for worker_index, _jobs_chunk in enumerate(chunks, start=1):
        account, partition, qos = next(lanes)
        chunk_script = run_dir / f"worker_{worker_index:02d}.sh"
        job_name = f"{args.dataset}-w{worker_index:02d}"
        cmd = [
            "sbatch",
            "--parsable",
            "--account",
            account,
            "--partition",
            partition,
            "--qos",
            qos,
            "--gres",
            "gpu:A100.80gb:1",
            "--mem",
            "96GB",
            "--time",
            "08:00:00",
            "--job-name",
            job_name,
            "--export",
            f"ALL,CHUNK_SCRIPT={chunk_script}",
            str(WORKER),
        ]
        job_id = subprocess.check_output(cmd, text=True).strip().split(";")[0].strip()
        submit_ids.append((worker_index, job_id, job_name, account, partition, qos))

    print(f"run_dir={run_dir}")
    print(f"manifest={manifest}")
    print(f"remaining_jobs={len(jobs)}")
    print(f"worker_count={len(chunks)}")
    for worker_index, job_id, job_name, account, partition, qos in submit_ids:
        print(
            f"worker_{worker_index:02d}={job_id}\t{job_name}\taccount={account}\tpartition={partition}\tqos={qos}"
        )


if __name__ == "__main__":
    main()
