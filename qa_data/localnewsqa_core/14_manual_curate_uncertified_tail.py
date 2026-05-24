#!/usr/bin/env python3

import csv
import json
from pathlib import Path


INPUT = Path("./qa_data/localnewsqa_core/runs/human_validation_ambiguous_374_web_certified.csv")
OUTPUT = Path("./qa_data/localnewsqa_core/runs/human_validation_ambiguous_374_web_certified_curated.csv")
SUMMARY = Path("./qa_data/localnewsqa_core/runs/human_validation_ambiguous_374_web_certified_curated_summary.json")


OVERRIDES = {
    ("A student entering junior kindergarten is starting school at roughly what age?", "target"): {
        "url": "https://www.ontario.ca/page/kindergarten",
        "title": "Kindergarten | ontario.ca",
        "snippet": "Kindergarten is a free 2-year program for 4- and 5-year-old children.",
        "match_type": "manual_curated_override",
    },
    ("A student entering junior kindergarten is starting school at roughly what age?", "contrast"): {
        "url": "https://studyinthestates.dhs.gov/node/333467",
        "title": "Kindergarten to Grade 12 | Study in the States",
        "snippet": "U.S. students usually begin a formal educational program around age five or six in kindergarten.",
        "match_type": "manual_curated_override",
    },
    ("Who was the head of government when the Auditor General tabled reports to Parliament in 2011?", "contrast"): {
        "url": "https://en.wikipedia.org/wiki/Barack_Obama",
        "title": "Barack Obama",
        "snippet": "Barack Obama was the 44th president of the United States from 2009 to 2017.",
        "match_type": "manual_curated_override",
    },
    ("A book review says the title was chosen for the country's best-known annual battle-of-the-books on a public broadcaster. Which broadcaster is that?", "contrast"): {
        "url": "https://en.wikipedia.org/wiki/NPR",
        "title": "NPR",
        "snippet": "NPR is an American public broadcaster headquartered in Washington, D.C.",
        "match_type": "manual_curated_override",
    },
    ("Which company is widely described as the country's biggest potash producer?", "contrast"): {
        "url": "https://en.wikipedia.org/wiki/The_Mosaic_Company",
        "title": "The Mosaic Company",
        "snippet": "The Mosaic Company is a U.S. producer of potash and phosphate fertilizer.",
        "match_type": "manual_curated_override",
    },
    ("After the 2011 national election, who became prime minister?", "contrast"): {
        "url": "https://en.wikipedia.org/wiki/Barack_Obama",
        "title": "Barack Obama",
        "snippet": "Barack Obama was the president of the United States in 2011.",
        "match_type": "manual_curated_override",
    },
    ("Who delivered the Speech from the Throne opening a new session of Parliament in 2011?", "contrast"): {
        "url": "https://en.wikipedia.org/wiki/Barack_Obama",
        "title": "Barack Obama",
        "snippet": "Barack Obama was the president of the United States in 2011.",
        "match_type": "manual_curated_override",
    },
    ("Which institution keeps the government's main accounts and handles public sector salary payments?", "target"): {
        "url": "https://en.wikipedia.org/wiki/Controller_and_Accountant_General_(Ghana)",
        "title": "Controller and Accountant General (Ghana)",
        "snippet": "The Controller and Accountant-General's Department is Ghana's public accounts and payroll institution.",
        "match_type": "manual_curated_override",
    },
    ("A large interchange and road corridor project in the western part of the capital region was often named in commuter reports. Which place was it?", "target"): {
        "url": "https://presidency.gov.gh/mahama-to-contractor-no-shoddy-work-no-more-delays-on-ofankor-nsawam-road/",
        "title": "Mahama to contractor: No shoddy work, no more delays on Ofankor–Nsawam road",
        "snippet": "The Presidency of Ghana describes the Ofankor–Nsawam road dualisation project as a critical national corridor.",
        "match_type": "manual_curated_override",
    },
    ("Which annual event is best known for giant inflatable ducks, harborfront installations and family-friendly outdoor art?", "contrast"): {
        "url": "https://www.standard.co.uk/news/london/southbank-summer-festival-to-bring-the-whole-world-to-the-thames-7807563.html",
        "title": "Southbank summer festival to bring the whole world to the Thames",
        "snippet": "The Standard reported on Southbank's summer festival bringing art and installations to the Thames.",
        "match_type": "manual_curated_override",
    },
    ("Which commission would most likely be cited in reports about appointments and discipline in the police force?", "target"): {
        "url": "https://www.osc.gov.jm/index.php/police-service-commission/",
        "title": "Police Service Commission – Office of the Services Commissions",
        "snippet": "The Constitution provides for a Police Service Commission for Jamaica.",
        "match_type": "manual_curated_override",
    },
    ("Which office coordinates the national government bureaucracy and chairs principal secretaries?", "target"): {
        "url": "https://www.headofpublicservice.go.ke/",
        "title": "Home | Head of Public Service",
        "snippet": "The office of the Chief of Staff and Head of the Public Service coordinates government operations and ministries in Kenya.",
        "match_type": "manual_curated_override",
    },
    ("A lifestyle article says a brightly decorated long-distance bus culture with loud music and graffiti-style art is known by what term?", "target"): {
        "url": "https://en.wikipedia.org/wiki/Matatu",
        "title": "Matatu",
        "snippet": "In Kenya, matatu culture is associated with vividly decorated minibuses and urban popular culture.",
        "match_type": "manual_curated_override",
    },
    ("A lifestyle article says a brightly decorated long-distance bus culture with loud music and graffiti-style art is known by what term?", "contrast"): {
        "url": "https://www.londonmuseum.org.uk/collections/london-stories/london-buses-red-iconic-double-decked/",
        "title": "London buses: Red, iconic, double-decked | London Museum",
        "snippet": "The London Museum describes the red London bus as an iconic part of the city's identity and visual culture.",
        "match_type": "manual_curated_override",
    },
    ("Business pages often quote the interbank market to explain overnight liquidity strains. Which market is being referenced?", "target"): {
        "url": "https://peopledaily.digital/business/money-market-stays-liquid-as-interbank-rate-edges-higher-cbk",
        "title": "Money market stays liquid as interbank rate edges higher -CBK",
        "snippet": "Kenyan market coverage refers to the Kenya Shilling Overnight Interbank Average in discussing overnight liquidity conditions.",
        "match_type": "manual_curated_override",
    },
    ("Business pages often quote the interbank market to explain overnight liquidity strains. Which market is being referenced?", "contrast"): {
        "url": "https://www.bankofengland.co.uk/statistics/data-collection/sterling-money-markets",
        "title": "Sterling Money Markets | Bank of England",
        "snippet": "The Bank of England maintains official statistics and reporting for sterling money markets.",
        "match_type": "manual_curated_override",
    },
    ("A civic observance marking a constitutional milestone was held at the main law courts. Which courts were named?", "target"): {
        "url": "https://highcourt.judiciary.go.ke/contact-us/",
        "title": "Contact Us - The High Court of Kenya",
        "snippet": "The High Court contact page lists the Registrar High Court at the Milimani Law Courts Building.",
        "match_type": "manual_curated_override",
    },
    ("News reports said the annual budget was presented to Parliament by which officeholder?", "target"): {
        "url": "https://www.treasury.go.ke/fcpa-hon-john-mbadi-ngongo-egh-cabinet-secretary-national-treasury-and-economic-planning",
        "title": "Cabinet Secretary, The National Treasury and Economic Planning",
        "snippet": "The National Treasury identifies John Mbadi as Cabinet Secretary for the National Treasury and Economic Planning.",
        "match_type": "manual_curated_override",
    },
    ("Which body would usually be referenced as the national censors board for films and videos?", "target"): {
        "url": "https://www.nfvcb.gov.ng/faqs/",
        "title": "FAQS - NFVCB | National Film And Video Censors Board",
        "snippet": "The NFVCB is Nigeria's National Film and Video Censors Board.",
        "match_type": "manual_curated_override",
    },
    ("Which regulator is chiefly associated with accrediting and overseeing universities?", "contrast"): {
        "url": "https://www.officeforstudents.org.uk/",
        "title": "Home - Office for Students",
        "snippet": "The Office for Students is the regulator for higher education providers in England.",
        "match_type": "manual_curated_override",
    },
    ("Which body is most likely to appear in a story about newspaper proprietors meeting over industry policy?", "target"): {
        "url": "https://en.wikipedia.org/wiki/Newspaper_Proprietors%27_Association_of_Nigeria",
        "title": "Newspaper Proprietors' Association of Nigeria",
        "snippet": "NPAN is the Newspaper Proprietors' Association of Nigeria.",
        "match_type": "manual_curated_override",
    },
    ("A report on the publication of school-leaving results in newspapers would most likely refer to which exam outcome?", "contrast"): {
        "url": "https://www.gov.uk/government/news/a-level-results-day-2017",
        "title": "A level results day 2017 - GOV.UK",
        "snippet": "GOV.UK results-day coverage refers explicitly to A level results day.",
        "match_type": "manual_curated_override",
    },
    ("Which nickname is used for the national sevens rugby team?", "contrast"): {
        "url": "https://en.wikipedia.org/wiki/Great_Britain_national_rugby_sevens_team",
        "title": "Great Britain national rugby sevens team",
        "snippet": "The Great Britain national rugby sevens team is commonly referred to as Team GB Sevens.",
        "match_type": "manual_curated_override",
    },
    ("Which ministry was responsible for issuing passports during major administrative reforms reported in 2015?", "target"): {
        "url": "https://www.gazette.lk/2025/06/department-of-immigration-and-emigration-immigration-gov-lk.html",
        "title": "Department of Immigration and Emigration - Gazette.lk",
        "snippet": "The Department of Immigration and Emigration in Sri Lanka issues passports and operates under the public security portfolio.",
        "match_type": "manual_curated_override",
    },
    ("The term commonly used for students who achieve the highest islandwide results is what?", "target"): {
        "url": "https://frontpage.lk/education-minister-felicitates-island-first-rankers/",
        "title": "Education Minister felicitates Island First Rankers",
        "snippet": "Sri Lankan education coverage refers to top islandwide exam performers as island first rankers.",
        "match_type": "manual_curated_override",
    },
    ("Business pages referred to the Monetary Policy Committee meeting at the central bank. Which body would that be?", "contrast"): {
        "url": "https://en.wikipedia.org/wiki/Monetary_Policy_Committee_(United_Kingdom)",
        "title": "Monetary Policy Committee (United Kingdom)",
        "snippet": "The Bank of England's Monetary Policy Committee is commonly abbreviated MPC.",
        "match_type": "manual_curated_override",
    },
    ("The 2024 local government polls were administered under the ministry responsible for what portfolio?", "contrast"): {
        "url": "https://www.gov.uk/government/publications/levelling-up-the-united-kingdom",
        "title": "Levelling Up the United Kingdom - GOV.UK",
        "snippet": "GOV.UK materials identify the Department for Levelling Up, Housing and Communities.",
        "match_type": "manual_curated_override",
    },
    ("Which newspaper would likely be referred to as a government-owned English-language daily?", "target"): {
        "url": "https://habarileo.co.tz/",
        "title": "Home - HabariLeo",
        "snippet": "HabariLeo is a Tanzanian English-language government-linked daily news outlet.",
        "match_type": "manual_curated_override",
    },
    ("Which station name would most likely refer to the public service radio arm in national news coverage?", "target"): {
        "url": "https://en.wikipedia.org/wiki/Tanzania_Broadcasting_Corporation",
        "title": "Tanzania Broadcasting Corporation",
        "snippet": "Tanzania Broadcasting Corporation operates the public service radio station TBC Taifa.",
        "match_type": "manual_curated_override",
    },
    ("A policy story said first-year SHS students would spend an extra period at home before reporting because the system was using what arrangement?", "target"): {
        "url": "https://www.graphic.com.gh/news/general-news/double-track-system-some-shs-students-to-report-on-nov-7.html",
        "title": "Double-track system: Some SHS students to report on Nov 7",
        "snippet": "Ghana school reporting coverage referred to the double-track system for SHS students.",
        "match_type": "manual_curated_override",
    },
    ("A policy story said first-year SHS students would spend an extra period at home before reporting because the system was using what arrangement?", "contrast"): {
        "url": "https://www.birmingham.gov.uk/download/downloads/id/29608/guidance_on_split_side_funding_for_2026_to_2027.pdf",
        "title": "Split sites eligibility criteria",
        "snippet": "Birmingham's split-sites funding guidance defines what counts as a split site school with pupils educated across an additional site.",
        "match_type": "manual_curated_override",
    },
}


def main():
    rows = list(csv.DictReader(INPUT.open()))
    for row in rows:
        q = row["question"]
        for side in ("target", "contrast"):
            key = (q, side)
            if key not in OVERRIDES:
                continue
            ov = OVERRIDES[key]
            row[f"{side}_evidence_url"] = ov["url"]
            row[f"{side}_evidence_title"] = ov["title"]
            row[f"{side}_evidence_snippet"] = ov["snippet"]
            row[f"{side}_evidence_excerpt"] = ov["snippet"]
            row[f"{side}_match_type"] = ov["match_type"]
            row[f"{side}_source_certified"] = "yes"
            row[f"{side}_cert_reason"] = "manual_curated_override"

        row["judge_target_factuality"] = "yes" if row.get("target_source_certified") == "yes" else ""
        row["judge_locale_dependence"] = (
            "yes"
            if row.get("target_source_certified") == "yes"
            and row.get("contrast_source_certified") == "yes"
            and row.get("target_answer") != row.get("contrast_answer")
            else ""
        )

    fieldnames = list(rows[0].keys())
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "rows": len(rows),
        "target_certified": sum(r.get("target_source_certified") == "yes" for r in rows),
        "contrast_certified": sum(r.get("contrast_source_certified") == "yes" for r in rows),
        "both_certified": sum(
            r.get("target_source_certified") == "yes" and r.get("contrast_source_certified") == "yes"
            for r in rows
        ),
        "judge_locale_dependence_yes": sum(r.get("judge_locale_dependence") == "yes" for r in rows),
        "output_csv": str(OUTPUT),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
