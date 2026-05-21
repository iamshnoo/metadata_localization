#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${MANIFEST:-/path/to/logs/slurm_logs/localnewsqa_gold_reruns_20260516_062858.tsv}
POLL_SECONDS=${POLL_SECONDS:-300}
MAX_SECONDS=${MAX_SECONDS:-86400}
LOG_PREFIX="[lnqa-retarget]"

deadline=$((SECONDS + MAX_SECONDS))

while true; do
  python - <<'PY'
import csv
import subprocess

manifest = "/path/to/logs/slurm_logs/localnewsqa_gold_reruns_20260516_062858.tsv"
rows = list(csv.DictReader(open(manifest), delimiter="\t"))
candidate_ids = [
    r["job_id"]
    for r in rows
    if r["job_id"].isdigit()
    and (r["group"].startswith("external") or r["group"].startswith("adversarial"))
]
if not candidate_ids:
    raise SystemExit

status = subprocess.run(
    ["sacct", "-j", ",".join(candidate_ids), "--format=JobID,State,Partition,QOS,Account", "-P"],
    text=True,
    capture_output=True,
    check=False,
).stdout
state = {}
part = {}
for line in status.splitlines()[1:]:
    fields = line.split("|")
    if len(fields) < 5:
        continue
    jid, st, partition, qos, account = fields[:5]
    if "." in jid:
        continue
    state[jid] = st
    part[jid] = partition

pending = [
    jid
    for jid in candidate_ids
    if state.get(jid) == "PENDING" and part.get(jid) != "gpuq"
]
accounts = ["SLURM_ACCOUNT", "cs678sp23"]
updated = 0
failed = 0
for i, jid in enumerate(pending):
    account = accounts[i % len(accounts)]
    cmd = [
        "scontrol",
        "update",
        f"JobId={jid}",
        "Partition=gpuq",
        "QOS=gpu",
        f"Account={account}",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode == 0:
        updated += 1
    else:
        failed += 1
print(f"candidates={len(pending)} updated={updated} failed={failed}")
PY

  if ((SECONDS >= deadline)); then
    printf '%s reached max runtime\n' "$LOG_PREFIX"
    break
  fi
  printf '%s sleeping %ss\n' "$LOG_PREFIX" "$POLL_SECONDS"
  sleep "$POLL_SECONDS"
done
