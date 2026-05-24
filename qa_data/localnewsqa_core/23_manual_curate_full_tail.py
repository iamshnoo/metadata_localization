#!/usr/bin/env python3

import csv
import json
from pathlib import Path


INPUT = Path("./qa_data/localnewsqa_core/runs/human_validation_full_35874_web_certified_aggressive.csv")
OUTPUT = Path("./qa_data/localnewsqa_core/runs/human_validation_full_35874_web_certified_curated.csv")
SUMMARY = Path("./qa_data/localnewsqa_core/runs/human_validation_full_35874_web_certified_curated_summary.json")
TAIL = Path("./qa_data/localnewsqa_core/runs/human_validation_full_35874_web_uncertified_tail_curated.csv")
TAIL_SUMMARY = Path("./qa_data/localnewsqa_core/runs/human_validation_full_35874_web_uncertified_tail_curated_summary.json")


OVERRIDES = {
    "localnewsqa_explicit_0008507": {
        "target": {
            "url": "https://www.plkctslps.edu.hk/en/content.php?wid=49",
            "title": "Po Leung Kuk Camoes Tan Siu Lin Primary School",
            "snippet": "We have 5 classes for each class level from P1 to P6.",
        },
    },
    "localnewsqa_explicit_0008515": {
        "target": {
            "url": "https://www.edb.gov.hk/en/curriculum-development/curriculum-area/life-wide-learning/ole/time_arrangement.html",
            "title": "Suggested Time Allocation of OLE - Education Bureau",
            "snippet": "Schools are encouraged to have flexible planning of OLE for students throughout the three years of the SS education (S4 to S6).",
        },
    },
    "localnewsqa_explicit_0009678": {
        "target": {
            "url": "https://en.wiktionary.org/wiki/SJK_%28C%29",
            "title": "SJK (C) - Wiktionary, the free dictionary",
            "snippet": "(Malaysia, education) initialism of sekolah jenis kebangsaan (Cina) ('Chinese national-type school').",
        },
    },
    "localnewsqa_explicit_0013259": {
        "target": {
            "url": "https://en.wiktionary.org/wiki/bru",
            "title": "bru - Wiktionary, the free dictionary",
            "snippet": "(South Africa) bro; bra; term of address for a man.",
        },
    },
    "localnewsqa_explicit_0015333": {
        "target": {
            "url": "https://glis.edu.gh/junior-high-school/",
            "title": "Junior High School (J.H.S) - Ghana-Lebanon Islamic School",
            "snippet": "The school's policy does not allow any admission into JHS 3 (final year).",
        },
    },
    "localnewsqa_ambig_0020538": {
        "target": {
            "url": "https://www.amdsb.ca/apps/pages/kindergarten",
            "title": "Kindergarten - Elementary - Avon Maitland District School Board",
            "snippet": "Children who are 3 years old (JK) or 4 years old (SK) by December 31 are eligible to attend Full Day Kindergarten.",
        },
        "contrast": {
            "url": "https://www.newdurhamschool.org/resources/little-wildcats-pre-k-k",
            "title": "Little Wildcats (Pre-K & K) - New Durham Elementary School",
            "snippet": "Little Wildcats (Pre-K & K).",
        },
    },
    "localnewsqa_ambig_0025173": {
        "target": {
            "url": "https://mfa.gov.lk/en/mfa-consular-online/",
            "title": "Consular Affairs Division launches online authentication of GCE O/L & A/L Certificates",
            "snippet": "The service supports authentication of GCE Ordinary Level (O/L) and Advanced Level (A/L) certificates issued by Sri Lanka's Department of Examinations.",
        },
        "contrast": {
            "url": "https://en.wikipedia.org/wiki/Board_examination",
            "title": "Board examination",
            "snippet": "In India, board examinations are conducted at the completion of secondary (class 10) and senior secondary education (class 12).",
        },
    },
    "localnewsqa_ambig_0026382": {
        "target": {
            "url": "https://en.wikipedia.org/wiki/334_Scheme",
            "title": "334 Scheme",
            "snippet": "The 3-3-4 Scheme is the academic structure for senior secondary education and higher education in Hong Kong.",
        },
    },
    "localnewsqa_ambig_0029206": {
        "target": {
            "url": "https://nerdc.gov.ng/content_manager/jss1-3.html",
            "title": "Nigerian Educational Research and Development Council - NERDC Curriculum Junior Secondary 1 - 3",
            "snippet": "Junior Secondary School (JSS) 1-3 Basic Education Curriculum.",
        },
    },
    "localnewsqa_ambig_0029207": {
        "target": {
            "url": "https://nerdc.gov.ng/content_manager/jss1-3.html",
            "title": "Nigerian Educational Research and Development Council - NERDC Curriculum Junior Secondary 1 - 3",
            "snippet": "Junior Secondary School (JSS) 1-3 Basic Education Curriculum.",
        },
    },
    "localnewsqa_ambig_0029627": {
        "target": {
            "url": "https://nerdc.gov.ng/content_manager/jss1-3.html",
            "title": "Nigerian Educational Research and Development Council - NERDC Curriculum Junior Secondary 1 - 3",
            "snippet": "Junior Secondary School (JSS) 1-3 Basic Education Curriculum.",
        },
    },
    "localnewsqa_ambig_0031237": {
        "target": {
            "url": "https://www.kenyanews.go.ke/govt-implements-critical-reforms-to-fully-transition-to-cbc/",
            "title": "Govt implements critical reforms to fully transition to CBC - Kenya News Agency",
            "snippet": "As Kenya phases out the 8-4-4 education system, the government has implemented critical reforms to ensure the successful adoption of the Competency-Based Curriculum.",
        },
        "contrast": {
            "url": "https://www.gov.uk/national-curriculum",
            "title": "The national curriculum: Overview - GOV.UK",
            "snippet": "The national curriculum is organised into blocks of years called key stages (KS).",
        },
    },
    "localnewsqa_ambig_0031326": {
        "target": {
            "url": "https://uonbi.ac.ke/sites/default/files/JAB%202nd%20Revision.pdf",
            "title": "JOINT ADMISSIONS BOARD",
            "snippet": "This is to inform all the 2011 K.C.S.E examination candidates who met the JAB admission cut off point.",
        },
    },
    "localnewsqa_ambig_0031670": {
        "target": {
            "url": "https://en.wikipedia.org/wiki/Kenya_Police",
            "title": "Kenya Police",
            "snippet": "Officer Commanding Station (OCS) is in charge of a Police Station in a Ward and oversees all its Police Posts and Patrol Bases.",
        },
    },
    "localnewsqa_ambig_0032216": {
        "target": {
            "url": "https://www.ghanabusinessnews.com/2016/03/10/bank-of-ghana-denies-responsibility-for-dkmn-the-process-of-liquidating-dkm/",
            "title": "Bank of Ghana denies responsibility for DKM Microfinance - Ghana Business News",
            "snippet": "The Bank of Ghana said it was in the process of liquidating the assets of DKM Microfinance Limited to pay off depositors.",
        },
    },
    "localnewsqa_ambig_0032568": {
        "target": {
            "url": "https://www.nccegh.org/news/ncce-sekyere-central-addresses-core-values-and-environmental-governance-on-gbc-radio",
            "title": "NCCE Sekyere Central Addresses Core Values and Environmental Governance on GBC Radio",
            "snippet": "The NCCE in Sekyere Central engaged the public on vital topics at GBC Radio.",
        },
    },
    "localnewsqa_ambig_0034935": {
        "target": {
            "url": "https://www.friendsoftheearth.ie/press/pfg-our-best-chance-of-faster-and-fairer-climate-action-over-the-next-5-years/",
            "title": "PfG: Our best chance of faster and fairer climate action over the next 5 years | Friends of the Earth",
            "snippet": "The commitment to an average 7% annual reduction in polluting emissions is a huge step forward.",
        },
    },
}


REJECTS = {
    "localnewsqa_ambig_0026382": "Target-side 3-3-4 shorthand is real, but the UK contrast shorthand 4-2-3 could not be substantiated and appears non-canonical.",
    "localnewsqa_ambig_0031523": "KTN Leo appears to be an evening Swahili news bulletin, not the breakfast TV magazine show described in the question.",
    "localnewsqa_ambig_0032476": "The XYZ Show is a Kenyan satirical puppet show and does not match the Ghana/Joy Prime framing in the item.",
    "localnewsqa_ambig_0034935": "Ireland's 7% target is real, but the UK contrast answer 5% could not be substantiated for this question framing.",
}


def apply_side_override(row, side, payload):
    row[f"{side}_evidence_url"] = payload["url"]
    row[f"{side}_evidence_title"] = payload["title"]
    row[f"{side}_evidence_snippet"] = payload["snippet"]
    row[f"{side}_evidence_excerpt"] = payload["snippet"]
    row[f"{side}_match_type"] = "manual_curated_override"
    row[f"{side}_source_certified"] = "yes"
    row[f"{side}_cert_reason"] = "manual_curated_override"


def is_unresolved(row):
    target_ok = row.get("target_source_certified") == "yes"
    has_contrast = bool(str(row.get("contrast_answer", "")).strip())
    if not has_contrast:
        return not target_ok
    contrast_ok = row.get("contrast_source_certified") == "yes"
    return not (target_ok and contrast_ok)


def main():
    rows = list(csv.DictReader(INPUT.open("r", newline="", encoding="utf-8")))
    fieldnames = list(rows[0].keys())
    for field in ("manual_review_status", "manual_review_reason"):
        if field not in fieldnames:
            fieldnames.append(field)

    fixed = 0
    rejected = 0
    for row in rows:
        row_id = row["id"]
        note_parts = [str(row.get("annotator_notes", "")).strip()]
        unresolved_before = is_unresolved(row)

        if row_id in OVERRIDES:
            for side, payload in OVERRIDES[row_id].items():
                apply_side_override(row, side, payload)
            note_parts.append("manual curated source override applied")

        if row_id in REJECTS:
            row["manual_review_status"] = "manual_reject"
            row["manual_review_reason"] = REJECTS[row_id]
            rejected += 1
            note_parts.append(REJECTS[row_id])
        else:
            if unresolved_before and not is_unresolved(row):
                fixed += 1
                row["manual_review_status"] = "manual_fixed"
                row["manual_review_reason"] = "resolved by manual source override"
            elif not is_unresolved(row):
                row["manual_review_status"] = row.get("manual_review_status", "") or "resolved"
                row["manual_review_reason"] = row.get("manual_review_reason", "") or "source support available"
            else:
                row["manual_review_status"] = row.get("manual_review_status", "") or "unresolved"
                row["manual_review_reason"] = row.get("manual_review_reason", "") or "still unresolved after curation"

        row["annotator_notes"] = " | ".join([p for p in note_parts if p])
        row["judge_target_factuality"] = "yes" if row.get("target_source_certified") == "yes" else ""
        row["judge_locale_dependence"] = (
            "yes"
            if row.get("target_source_certified") == "yes"
            and row.get("contrast_source_certified") == "yes"
            and row.get("target_answer") != row.get("contrast_answer")
            else ""
        )

    tail_rows = [
        r for r in rows
        if is_unresolved(r) or r.get("manual_review_status") == "manual_reject"
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with TAIL.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(tail_rows)

    summary = {
        "rows": len(rows),
        "target_certified": sum(r.get("target_source_certified") == "yes" for r in rows),
        "contrast_certified": sum(
            bool(str(r.get("contrast_answer", "")).strip()) and r.get("contrast_source_certified") == "yes"
            for r in rows
        ),
        "both_certified": sum(
            bool(str(r.get("contrast_answer", "")).strip())
            and r.get("target_source_certified") == "yes"
            and r.get("contrast_source_certified") == "yes"
            for r in rows
        ),
        "judge_locale_dependence_yes": sum(r.get("judge_locale_dependence") == "yes" for r in rows),
        "manual_fixed_rows": fixed,
        "manual_rejected_rows": rejected,
        "remaining_tail_rows": len(tail_rows),
        "output_csv": str(OUTPUT),
        "tail_csv": str(TAIL),
    }
    tail_summary = {
        "rows": len(tail_rows),
        "manual_rejected_rows": sum(r.get("manual_review_status") == "manual_reject" for r in tail_rows),
        "target_certified": sum(r.get("target_source_certified") == "yes" for r in tail_rows),
        "contrast_certified": sum(
            bool(str(r.get("contrast_answer", "")).strip()) and r.get("contrast_source_certified") == "yes"
            for r in tail_rows
        ),
        "both_certified": sum(
            bool(str(r.get("contrast_answer", "")).strip())
            and r.get("target_source_certified") == "yes"
            and r.get("contrast_source_certified") == "yes"
            for r in tail_rows
        ),
        "output_csv": str(TAIL),
    }

    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    TAIL_SUMMARY.write_text(json.dumps(tail_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
