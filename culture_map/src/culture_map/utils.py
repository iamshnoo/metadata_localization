import json
import os
import re
from pathlib import Path


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_env_file(path):
    path = Path(path)
    if not path.exists():
        return {}
    loaded = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def slugify(text):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "value"


def write_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_jsonl(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
