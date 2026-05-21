#!/usr/bin/env python3
import importlib.util
import math
import shlex
import subprocess
from pathlib import Path


ROOT = Path("/path/to/metacul")
SUBMITTER = ROOT / "slurm/submit_final_benchmark_matrix.py"
WORKER = ROOT / "slurm/geomlama_worker_fullgpu.slurm"
WRAPPER = ROOT / "slurm/pretrained_external_eval_single.slurm"
LOG_DIR = Path("/path/to/logs/slurm_logs")


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


def completed_job_names() -> set[str]:
    out = subprocess.check_output(
        [
            "bash",
            "-lc",
            "sacct -S 2026-05-02 -u USER_NAME -n -P --format=JobName,State | rg '^bm-geomlama-'",
        ],
        text=True,
    )
    done: set[str] = set()
    for line in out.splitlines():
        if not line:
            continue
        name, state = line.split("|", 1)
        if state.startswith("COMPLETED"):
            done.add(name)
    return done


def pending_geomlama_job_ids() -> list[str]:
    out = subprocess.check_output(
        [
            "bash",
            "-lc",
            "squeue -u USER_NAME -h -o '%i|%j|%T' | rg '^.*\\|bm-geomlama-'",
        ],
        text=True,
    )
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


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_submitter_module()

    rows = mod.load_rows(only_datasets={"geomlama"})
    done = completed_job_names()

    jobs: list[dict] = []
    for row in rows:
        for seed in mod.DEFAULT_SEEDS:
            job = mod.external_benchmark_job(row, seed)
            if job["job_name"] in done:
                continue
            jobs.append(job)

    jobs.sort(key=lambda j: j["job_name"])
    if not jobs:
        print("No remaining GeoMLaMA jobs to submit.")
        return

    old_pending = pending_geomlama_job_ids()
    if old_pending:
        for chunk in chunked(old_pending, 200):
            subprocess.check_call(["scancel", *chunk])

    ts = subprocess.check_output(["date", "+%Y%m%d_%H%M%S"], text=True).strip()
    run_dir = LOG_DIR / f"geomlama_priority_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "manifest.tsv"

    worker_count = min(8, len(jobs))
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
                lines.append(
                    f"if ! {shell_exports(env)} bash {shlex.quote(str(WRAPPER))}; then"
                )
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

    submit_ids: list[tuple[int, str, str]] = []
    for worker_index, _jobs_chunk in enumerate(chunks, start=1):
        chunk_script = run_dir / f"worker_{worker_index:02d}.sh"
        job_name = f"geomlama-a100-w{worker_index:02d}"
        cmd = [
            "sbatch",
            "--parsable",
            "--account",
            "SLURM_ACCOUNT",
            "--partition",
            "contrib-gpuq",
            "--qos",
            "cs_dept",
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
        submit_ids.append((worker_index, job_id, job_name))

    print(f"run_dir={run_dir}")
    print(f"manifest={manifest}")
    print(f"remaining_geomlama_jobs={len(jobs)}")
    print(f"worker_count={len(chunks)}")
    for worker_index, job_id, job_name in submit_ids:
        print(f"worker_{worker_index:02d}={job_id}\t{job_name}")


if __name__ == "__main__":
    main()
