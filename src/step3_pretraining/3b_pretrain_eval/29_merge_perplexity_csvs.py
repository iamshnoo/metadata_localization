#!/usr/bin/env python3
import argparse
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument('--base', required=True)
parser.add_argument('--incoming', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

default_fieldnames = [
    'model_path', 'test_set_path', 'mean_ppl', 'ci_low', 'ci_high',
    'skipped', 'ci_level', 'bootstrap_method', 'split', 'max_samples',
    'seed', 'bootstrap_iters'
]

def read_rows(path):
    if not os.path.exists(path):
        return {}, []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        seen_fieldnames = list(reader.fieldnames or [])
        rows = {}
        for row in reader:
            rows[(row['model_path'], row['test_set_path'])] = row
        return rows, seen_fieldnames

def merge_fieldnames(*field_lists):
    merged = []
    for field_list in field_lists:
        for field in field_list:
            if field and field not in merged:
                merged.append(field)
    return merged

base_rows, base_fields = read_rows(args.base)
incoming_rows, incoming_fields = read_rows(args.incoming)
rows = base_rows
rows.update(incoming_rows)
ordered = [rows[k] for k in sorted(rows.keys())]
fieldnames = merge_fieldnames(default_fieldnames, base_fields, incoming_fields)
with open(args.output, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(ordered)
print(f"Wrote {len(ordered)} rows to {args.output}")
