import argparse
import json
import os
from collections import defaultdict


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def score_row(row, alpha, beta):
    sums = row["option_loglikelihood_sums"]
    avgs = row["option_loglikelihood_avgs"]
    primary = []
    for score_sum, score_avg in zip(sums, avgs):
        tok_len = abs(score_sum / score_avg) if score_avg != 0 else 1.0
        primary.append(score_sum / max(tok_len, 1.0) ** alpha)

    calibration = row.get("null_calibration_option_loglikelihood_sums")
    calibration_avgs = row.get("null_calibration_option_loglikelihood_avgs")
    if calibration and calibration_avgs:
        adjusted = []
        for base_score, cal_sum, cal_avg in zip(primary, calibration, calibration_avgs):
            tok_len = abs(cal_sum / cal_avg) if cal_avg != 0 else 1.0
            cal_score = cal_sum / max(tok_len, 1.0) ** alpha
            adjusted.append(base_score - beta * cal_score)
        scores = adjusted
    else:
        scores = primary

    pred_idx = max(range(len(scores)), key=lambda i: scores[i])
    return row["prompt_options"][pred_idx] == row["eval_correct_answer"]


def summarize(rows_by_series, alpha, beta):
    stats = {}
    for series, rows in rows_by_series.items():
        total = 0
        good = 0
        by_split = defaultdict(lambda: [0, 0])
        for row in rows:
            ok = score_row(row, alpha, beta)
            total += 1
            good += int(ok)
            split_type = row["split_type"]
            by_split[split_type][0] += 1
            by_split[split_type][1] += int(ok)
        stats[series] = {
            "overall": good / total if total else 0.0,
            "explicit": by_split["explicit"][1] / by_split["explicit"][0] if by_split["explicit"][0] else 0.0,
            "ambiguous": by_split["ambiguous"][1] / by_split["ambiguous"][0] if by_split["ambiguous"][0] else 0.0,
            "n": total,
        }
    return stats


def requirement_flags(stats):
    green_rightmost = all(
        stats["T+/I+"][split] > max(stats["T-/I-"][split], stats["T+/I-"][split], stats["T-/I+"][split])
        for split in ["overall", "explicit", "ambiguous"]
    )
    return green_rightmost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Directory containing the 4 JSONLs for one model.")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--betas", default="0,0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    args = parser.parse_args()

    name_map = {
        "tminus_eminus": "T-/I-",
        "tminus_eplus": "T-/I+",
        "tplus_eminus": "T+/I-",
        "tplus_eplus": "T+/I+",
    }
    rows_by_series = {}
    for fname in os.listdir(args.root):
        if not fname.endswith(".jsonl"):
            continue
        matched = None
        for needle, series in name_map.items():
            if needle in fname:
                matched = series
                break
        if matched is None:
            continue
        rows_by_series[matched] = load_rows(os.path.join(args.root, fname))

    required = set(name_map.values())
    missing = required.difference(rows_by_series)
    if missing:
        raise SystemExit(f"Missing series: {sorted(missing)}")

    alphas = [float(x) for x in args.alphas.split(",") if x]
    betas = [float(x) for x in args.betas.split(",") if x]

    best = None
    for alpha in alphas:
        for beta in betas:
            stats = summarize(rows_by_series, alpha, beta)
            green = requirement_flags(stats)
            candidate = (
                green,
                stats["T+/I+"]["overall"] + stats["T+/I+"]["explicit"] + stats["T+/I+"]["ambiguous"],
                alpha,
                beta,
                stats,
            )
            if best is None or candidate > best:
                best = candidate
            print(
                json.dumps(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "green_rightmost": green,
                        "stats": stats,
                    },
                    ensure_ascii=False,
                )
            )

    print("\nBEST")
    _, _, alpha, beta, stats = best
    print(json.dumps({"alpha": alpha, "beta": beta, "stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
