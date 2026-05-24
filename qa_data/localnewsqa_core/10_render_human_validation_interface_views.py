#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


CSV_PATH = Path(
    "./qa_data/localnewsqa_core/runs/human_validation_ambiguous_1700.csv"
)
OUT_DIR = Path("./latex/figs/appendix")


def load_rows():
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def add_round(ax, x, y, w, h, fc, ec="#d9deea", lw=1.5, r=0.02):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.005,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def chip(ax, x, y, text, fc="#f9fafb", ec="#d9deea", tc="#172033", fs=9):
    w = min(0.18, 0.012 + 0.0065 * len(text))
    h = 0.03
    add_round(ax, x, y, w, h, fc, ec, 1.0, 0.015)
    ax.text(x + 0.008, y + 0.015, text, va="center", ha="left", fontsize=fs, color=tc, fontweight="bold")
    return x + w + 0.008


def box_text(ax, x, y, w, h, title, body=None, body_size=10):
    add_round(ax, x, y, w, h, "#ffffff")
    ax.text(x + 0.012, y + h - 0.03, title, ha="left", va="top", fontsize=10, color="#6b7280", fontweight="bold")
    if body:
        ax.text(x + 0.012, y + h - 0.065, body, ha="left", va="top", fontsize=body_size, color="#172033", wrap=True)


def overview(rows):
    fig = plt.figure(figsize=(16, 10), dpi=150)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f6f7fb")
    ax.add_patch(plt.Rectangle((0, 0.82), 1, 0.18, color="#eef2ff"))

    add_round(ax, 0.015, 0.02, 0.18, 0.96, "#ffffff")
    add_round(ax, 0.21, 0.02, 0.31, 0.96, "#ffffff")
    add_round(ax, 0.54, 0.02, 0.445, 0.96, "#ffffff")

    ax.text(0.03, 0.95, "LocalNewsQA Human Validation", fontsize=18, fontweight="bold", color="#172033", va="top")
    ax.text(
        0.03,
        0.91,
        "Static review dashboard for ambiguous items. The left panel filters rows, the center panel lists questions,\n"
        "and the right panel exposes options, evidence, and current prefills before CSV editing.",
        fontsize=9.5,
        color="#6b7280",
        va="top",
    )

    stats = [("Rows", "1700"), ("Countries", "17"), ("Factual yes", "109"), ("Locale yes", "18")]
    coords = [(0.03, 0.79), (0.108, 0.79), (0.03, 0.70), (0.108, 0.70)]
    for (label, value), (x, y) in zip(stats, coords):
        box_text(ax, x, y, 0.07, 0.07, label, value, body_size=16)

    filter_y = 0.60
    for label, value in [
        ("SEARCH", "question, answer, hint"),
        ("COUNTRY", "Bangladesh"),
        ("TOPIC", "All topics"),
        ("PREFILL STATUS", "Needs review"),
        ("EVIDENCE COVERAGE", "Target and contrast evidence"),
    ]:
        ax.text(0.03, filter_y + 0.055, label, fontsize=8.5, color="#6b7280", fontweight="bold")
        add_round(ax, 0.03, filter_y, 0.145, 0.045, "#ffffff")
        ax.text(0.04, filter_y + 0.022, value, fontsize=9.5, color="#172033", va="center")
        filter_y -= 0.085

    ax.text(0.225, 0.95, "Rows", fontsize=12, fontweight="bold", color="#6b7280", va="top")
    add_round(ax, 0.465, 0.925, 0.04, 0.028, "#dbeafe", ec="#dbeafe", lw=0)
    ax.text(0.485, 0.939, "137 shown", fontsize=8, color="#1d4ed8", ha="center", va="center", fontweight="bold")

    y = 0.87
    for idx, row in enumerate(rows[:6]):
        fill = "#eef4ff" if idx == 0 else "#ffffff"
        add_round(ax, 0.22, y - 0.10, 0.29, 0.09, fill)
        ax.text(0.232, y - 0.024, row["question"][:88] + ("..." if len(row["question"]) > 88 else ""), fontsize=9.5, color="#172033", va="top", fontweight="bold", wrap=True)
        cx = 0.232
        cx = chip(ax, cx, y - 0.086, row["country"], fs=7.5)
        cx = chip(ax, cx, y - 0.086, row["topic"], fs=7.5)
        fact = "Target factuality yes" if (row.get("judge_target_factuality") or "").lower() == "yes" else "Target factuality blank"
        chip(ax, cx, y - 0.086, fact, "#dcfce7" if "yes" in fact else "#fef3c7", "#b7e4c7" if "yes" in fact else "#f6d37e", "#166534" if "yes" in fact else "#92400e", fs=7.3)
        y -= 0.11

    row = rows[0]
    ax.text(0.56, 0.95, row["question"], fontsize=16, fontweight="bold", color="#172033", va="top", wrap=True)
    cx = 0.56
    cy = 0.86
    for label in [row["id"], row["country"], row["continent"], row["topic"], str(row["year"])]:
        cx = chip(ax, cx, cy, label, fs=7.5)

    box_text(ax, 0.56, 0.76, 0.19, 0.07, "TARGET COUNTRY", row["target_country"], 12)
    box_text(ax, 0.77, 0.76, 0.19, 0.07, "CONTRAST COUNTRY", row["contrast_country"], 12)

    ax.text(0.56, 0.72, "OPTIONS", fontsize=10, fontweight="bold", color="#6b7280")
    oy = 0.69
    for opt in [s.strip() for s in row["options"].split("||")]:
        value = opt.split(": ", 1)[1] if ": " in opt else opt
        fc, ec = "#ffffff", "#d9deea"
        if value == row["target_answer"]:
            fc, ec = "#eef4ff", "#9dc0ff"
        elif value == row["contrast_answer"]:
            fc, ec = "#fff1f1", "#f0b1b1"
        add_round(ax, 0.56, oy - 0.05, 0.40, 0.045, fc, ec)
        ax.text(0.572, oy - 0.027, opt, fontsize=9.5, color="#172033", va="center")
        oy -= 0.058

    fig.savefig(OUT_DIR / "annotation_dashboard_overview.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def detail(rows):
    fig = plt.figure(figsize=(16, 10), dpi=150)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f6f7fb")
    ax.add_patch(plt.Rectangle((0, 0.82), 1, 0.18, color="#eef2ff"))
    add_round(ax, 0.02, 0.03, 0.96, 0.94, "#ffffff")

    row = rows[1]
    ax.text(0.04, 0.95, "Annotation Interface: Row Detail", fontsize=18, fontweight="bold", color="#172033", va="top")
    ax.text(0.04, 0.89, row["question"], fontsize=15, fontweight="bold", color="#172033", va="top", wrap=True)

    cx = 0.04
    for label in [row["country"], row["topic"], f"Target: {row['target_answer']}", f"Contrast: {row['contrast_answer']}"]:
        cx = chip(ax, cx, 0.81, label, fs=8)

    ax.text(0.04, 0.76, "Current judgments", fontsize=10, fontweight="bold", color="#6b7280")
    cx = 0.04
    cx = chip(ax, cx, 0.72, "Target factuality yes" if (row.get("judge_target_factuality") or "").lower() == "yes" else "Target factuality blank", "#dcfce7" if (row.get("judge_target_factuality") or "").lower() == "yes" else "#fef3c7", "#b7e4c7" if (row.get("judge_target_factuality") or "").lower() == "yes" else "#f6d37e", "#166534" if (row.get("judge_target_factuality") or "").lower() == "yes" else "#92400e", 8)
    cx = chip(ax, cx, 0.72, "Locale dependence yes" if (row.get("judge_locale_dependence") or "").lower() == "yes" else "Locale dependence blank", "#dcfce7" if (row.get("judge_locale_dependence") or "").lower() == "yes" else "#fef3c7", "#b7e4c7" if (row.get("judge_locale_dependence") or "").lower() == "yes" else "#f6d37e", "#166534" if (row.get("judge_locale_dependence") or "").lower() == "yes" else "#92400e", 8)
    chip(ax, cx, 0.72, "No explicit leakage yes", "#dcfce7", "#b7e4c7", "#166534", 8)

    ax.text(0.04, 0.66, "OPTIONS", fontsize=10, fontweight="bold", color="#6b7280")
    oy = 0.62
    for opt in [s.strip() for s in row["options"].split("||")]:
        value = opt.split(": ", 1)[1] if ": " in opt else opt
        fc, ec = "#ffffff", "#d9deea"
        if value == row["target_answer"]:
            fc, ec = "#eef4ff", "#9dc0ff"
        elif value == row["contrast_answer"]:
            fc, ec = "#fff1f1", "#f0b1b1"
        add_round(ax, 0.04, oy - 0.06, 0.42, 0.052, fc, ec)
        ax.text(0.053, oy - 0.034, opt, fontsize=10, color="#172033", va="center")
        oy -= 0.07

    ax.text(0.50, 0.66, "EVIDENCE", fontsize=10, fontweight="bold", color="#6b7280")
    for idx, prefix in enumerate(["target", "contrast"]):
        y = 0.62 - idx * 0.26
        add_round(ax, 0.50, y - 0.18, 0.44, 0.17, "#ffffff")
        label = "Target evidence" if prefix == "target" else "Contrast evidence"
        title = row.get(f"{prefix}_evidence_title", "") or "No title"
        meta = row.get(f"{prefix}_match_type", "") or "no_result"
        body = row.get(f"{prefix}_evidence_excerpt", "") or row.get(f"{prefix}_evidence_snippet", "") or "No evidence excerpt captured."
        ax.text(0.515, y - 0.03, label, fontsize=10, color="#172033", fontweight="bold", va="top")
        ax.text(0.515, y - 0.065, f"{meta} • {title}", fontsize=8.5, color="#6b7280", va="top")
        ax.text(0.515, y - 0.10, body[:280] + ("..." if len(body) > 280 else ""), fontsize=9, color="#172033", va="top", wrap=True)

    fig.savefig(OUT_DIR / "annotation_dashboard_detail.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    rows = load_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overview(rows)
    detail(rows)


if __name__ == "__main__":
    main()
