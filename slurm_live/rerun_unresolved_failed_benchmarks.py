#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import submit_final_benchmark_matrix as sfm


LOG_DIR = Path("/path/to/logs/slurm_logs")
MANIFEST_GLOB = "final_benchmark_matrix*.tsv"
SUCCESS_STATES = ("COMPLETED", "PENDING", "RUNNING")


def normalize_variant_for_jobname(variant: str) -> str:
    return variant.replace("/", "")


def make_key(
    dataset: str,
    family: str,
    track: str,
    variant: str,
    seed: str,
    role: str,
) -> Tuple[str, str, str, str, str, str]:
    return (dataset, family, track, variant, seed, role)


def iter_manifest_rows() -> Iterable[Dict[str, str]]:
    for manifest in sorted(LOG_DIR.glob(MANIFEST_GLOB)):
        with manifest.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                yield {"manifest": str(manifest), **row}


def build_manifest_indexes() -> Tuple[Dict[str, Dict[str, str]], Dict[Tuple[str, str, str, str, str, str], Dict[str, str]]]:
    by_job_id: Dict[str, Dict[str, str]] = {}
    by_key: Dict[Tuple[str, str, str, str, str, str], Dict[str, str]] = {}
    for row in iter_manifest_rows():
        job_id = row.get("job_id") or row.get("new_job_id")
        if job_id:
            by_job_id[job_id] = row
        key = make_key(
            row["dataset"],
            row["family"],
            row["track"],
            row["variant"],
            row["seed"],
            row["role"],
        )
        by_key[key] = row
    return by_job_id, by_key


def sacct_rows(start_date: str) -> List[Dict[str, str]]:
    raw = subprocess.check_output(
        [
            "sacct",
            "-S",
            start_date,
            "-u",
            "USER_NAME",
            "--format=JobIDRaw,JobName%120,State,ExitCode",
            "-P",
        ],
        text=True,
    )
    rows: List[Dict[str, str]] = []
    for line in raw.splitlines()[1:]:
        if not line.strip():
            continue
        job_id, job_name, state, exit_code = line.split("|", 3)
        if "." in job_id:
            continue
        rows.append(
            {
                "job_id": job_id,
                "job_name": job_name,
                "state": state,
                "exit_code": exit_code,
            }
        )
    return rows


def queued_job_names() -> Set[str]:
    raw = subprocess.check_output(
        ["squeue", "-u", "USER_NAME", "-h", "-o", "%j"],
        text=True,
    )
    return {line.strip() for line in raw.splitlines() if line.strip()}


def row_lookup() -> Dict[Tuple[str, str, str, str], sfm.Row]:
    rows = sfm.load_rows()
    lookup: Dict[Tuple[str, str, str, str], sfm.Row] = {}
    for row in rows:
        lookup[(row.dataset_key, row.family_key, row.track, row.variant)] = row
    return lookup


def submit_exact_row(
    row: sfm.Row,
    seed: int,
    role: str,
) -> Tuple[str, str, str]:
    if row.dataset_key == "localnewsqa":
        jobs = list(sfm.localnewsqa_jobs(row, seed))
        selected = None
        for job in jobs:
            if job["manifest_role"] == role:
                selected = job
                break
        if selected is None:
            raise RuntimeError(f"Could not find localnewsqa role={role} for {row}")
        job_id = sfm.sbatch_submit(
            Path(selected["script"]),
            selected["env"],
            sfm.benchmark_sbatch_args(row) + ["--job-name", selected["job_name"]],
        )
        return job_id, selected["script"], selected["env"]["OUT_DIR"]

    job = sfm.external_benchmark_job(row, seed)
    job_id = sfm.sbatch_submit(
        Path(job["script"]),
        job["env"],
        sfm.benchmark_sbatch_args(row) + ["--job-name", job["job_name"]],
    )
    return job_id, job["script"], job["env"]["OUT_ROOT"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun unresolved failed benchmark jobs only.")
    parser.add_argument("--start-date", default="2026-05-02")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    by_job_id, by_key = build_manifest_indexes()
    row_by_spec = row_lookup()
    queued_names = queued_job_names()

    states_by_key: Dict[Tuple[str, str, str, str, str, str], Set[str]] = defaultdict(set)
    failed_job_ids_by_key: Dict[Tuple[str, str, str, str, str, str], List[str]] = defaultdict(list)

    for row in sacct_rows(args.start_date):
        if not row["job_name"].startswith("bm-"):
            continue
        meta = by_job_id.get(row["job_id"])
        if meta is None:
            continue
        key = make_key(
            meta["dataset"],
            meta["family"],
            meta["track"],
            meta["variant"],
            meta["seed"],
            meta["role"],
        )
        states_by_key[key].add(row["state"])
        if row["state"].startswith("FAILED"):
            failed_job_ids_by_key[key].append(row["job_id"])

    unresolved: List[Tuple[Tuple[str, str, str, str, str, str], str]] = []
    for key, failed_ids in failed_job_ids_by_key.items():
        states = states_by_key[key]
        row_meta = by_key.get(key)
        if row_meta is None:
            continue
        expected_job_name = None
        spec_key = (row_meta["dataset"], row_meta["family"], row_meta["track"], row_meta["variant"])
        row = row_by_spec.get(spec_key)
        if row is not None:
            seed = int(row_meta["seed"])
            if row.dataset_key == "localnewsqa":
                for job in sfm.localnewsqa_jobs(row, seed):
                    if job["manifest_role"] == row_meta["role"]:
                        expected_job_name = job["job_name"]
                        break
            else:
                expected_job_name = sfm.external_benchmark_job(row, seed)["job_name"]

        if any(state.startswith(prefix) for prefix in SUCCESS_STATES for state in states if state.startswith(prefix)):
            continue
        if expected_job_name and expected_job_name in queued_names:
            continue
        unresolved.append((key, failed_ids[-1]))

    ts = subprocess.check_output(["date", "+%Y%m%d_%H%M%S"], text=True).strip()
    manifest = LOG_DIR / f"final_benchmark_matrix_failed_rerun_{ts}.tsv"

    if args.dry_run:
        print(f"unresolved_failed_keys={len(unresolved)}")
        for key, failed_id in unresolved[:50]:
            print(failed_id, *key, sep="\t")
        print(f"manifest={manifest}")
        return

    with manifest.open("w") as f:
        f.write("old_job_id\tnew_job_id\tdataset\tfamily\ttrack\tvariant\tseed\trole\tscript\tout_root_or_dir\n")
        for key, failed_id in unresolved:
            dataset, family, track, variant, seed_str, role = key
            row = row_by_spec[(dataset, family, track, variant)]
            new_job_id, script, out_dir = submit_exact_row(row, int(seed_str), role)
            f.write(
                "\t".join(
                    [
                        failed_id,
                        new_job_id,
                        dataset,
                        family,
                        track,
                        variant,
                        seed_str,
                        role,
                        script,
                        out_dir,
                    ]
                )
                + "\n"
            )

    print(f"Wrote manifest: {manifest}")
    print(f"Submitted jobs: {len(unresolved)}")


if __name__ == "__main__":
    main()
