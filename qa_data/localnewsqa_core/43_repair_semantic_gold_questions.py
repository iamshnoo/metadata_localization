#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/web_semantic_gold_ambiguous_1700"
)
DEFAULT_INPUT = DEFAULT_DIR / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
WEAK_SCRIPT = ROOT / "qa_data/localnewsqa_core/26_flag_weak_locale_ambiguous.py"


TOKEN_REPLACEMENTS = {
    "ministry": "public body",
    "department": "public body",
    "central": "national",
    "federal": "national",
    "airport": "transport hub",
    "commission": "public body",
    "police": "public service",
    "revenue": "tax",
    "women": "family",
    "children": "family",
    "kapil": "celebrity",
    "sharma": "host",
}

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan\\.?|Feb\\.?|Mar\\.?|Apr\\.?|Jun\\.?|Jul\\.?|Aug\\.?|Sep\\.?|Sept\\.?|Oct\\.?|Nov\\.?|Dec\\.?"
)

MANUAL_QUESTION_OVERRIDES = {
    "localnewsqa_ambig_0020221": "Who led the national government during the public-health response?",
    "localnewsqa_ambig_0019779": "A report on a flagship company's shares rising after results would most likely cite which market?",
    "localnewsqa_ambig_0020208": "Who led the national government during the major international summit?",
    "localnewsqa_ambig_0020212": "Who led the government during the major pipeline debate?",
    "localnewsqa_ambig_0021037": "Who was the outgoing head of government before the next leader took office?",
    "localnewsqa_ambig_0021345": "Which school level was commonly mentioned in reports on online learning device support?",
    "localnewsqa_ambig_0021402": "Which university is commonly referenced by initials in local university news?",
    "localnewsqa_ambig_0021511": "Which radio station is identified as a leading all-talk station in local reports?",
    "localnewsqa_ambig_0021427": "Which media group is behind the major broadcast outlets referenced in local reports?",
    "localnewsqa_ambig_0021860": "Who became the state-level head of government in 2016?",
    "localnewsqa_ambig_0021974": "Which national public broadcaster is overseen by the statutory broadcasting corporation?",
    "localnewsqa_ambig_0022264": "Which exam is used for the main national engineering entrance route?",
    "localnewsqa_ambig_0022915": "Who was the provincial head of government in 2015?",
    "localnewsqa_ambig_0023387": "Which actor starred opposite a leading actress in a major romantic film?",
    "localnewsqa_ambig_0023206": "Which company operates the large private-sector power utility serving the main commercial city?",
    "localnewsqa_ambig_0023216": "Which company was the largest cement producer by capacity in mid-2010s reporting?",
    "localnewsqa_ambig_0023929": "Which institution handles welfare and family-policy portfolios?",
    "localnewsqa_ambig_0024589": "Government notices announced special office timings during the fasting month. Which ministry typically issues circulars for civil servants?",
    "localnewsqa_ambig_0024719": "Which transport hub is the main international gateway serving the capital?",
    "localnewsqa_ambig_0025742": "Which transport hub is described as the main international gateway near the commercial capital?",
    "localnewsqa_ambig_0025907": "Which office was responsible for constitutional affairs and mainland relations in many political stories?",
    "localnewsqa_ambig_0025926": "Which office was responsible for public messaging on national-security legislation in 2024?",
    "localnewsqa_ambig_0026764": "Reports on a statutory holiday for workers in early May would most likely name which observance?",
    "localnewsqa_ambig_0027892": "Which airport is commonly described in news reports as the country's main aviation gateway?",
    "localnewsqa_ambig_0028019": "Which office handles diplomacy and overseas policy?",
    "localnewsqa_ambig_0028090": "Which office collects domestic taxes?",
    "localnewsqa_ambig_0028120": "Which agency collects domestic taxes?",
    "localnewsqa_ambig_0028211": "Which agency compiles national accounts figures cited in quarterly growth stories?",
    "localnewsqa_ambig_0028728": "Who delivered the national address at the start of 2013?",
    "localnewsqa_ambig_0028838": "Who would normally appoint the service chiefs or top military commanders on the advice of national security bodies?",
    "localnewsqa_ambig_0029247": "Students awaiting admission often check which body released the latest tertiary matriculation cut-off marks?",
    "localnewsqa_ambig_0029351": "Which annual event is described as the flagship cinema festival?",
    "localnewsqa_ambig_0030751": "Which city is home to the national botanical garden?",
    "localnewsqa_ambig_0031159": "Which company is the main securities exchange where shares of major local firms are traded?",
    "localnewsqa_ambig_0031488": "A TV business story says a broadcaster filed results with the stock exchange after strong advertising revenue. Which exchange is most likely?",
    "localnewsqa_ambig_0032125": "Which public body is known for promoting media freedom and handling some complaints involving the press?",
    "localnewsqa_ambig_0032931": "News reports said a major expansion was underway at the country's busiest international airport. Which airport was it?",
    "localnewsqa_ambig_0033094": "Which office handles the court system and public-law institutions?",
    "localnewsqa_ambig_0033350": "Reports on airport privatization, ground handling, or aviation business at the main aviation gateway would most likely center on which airport?",
    "localnewsqa_ambig_0033391": "Which public university is best known nationally for agriculture and veterinary training?",
    "localnewsqa_ambig_0033598": "An arts feature refers to the country's best-known film and cultural festival held on the coast. Which event fits best?",
    "localnewsqa_ambig_0033802": "During holiday travel coverage, what term would most likely refer to local road transport vehicles?",
    "localnewsqa_ambig_0035078": "Which broadcaster is funded in part through the television licence fee?",
    "localnewsqa_ambig_0035383": "Which station is a main home for live summer sports championship coverage?",
    "localnewsqa_ambig_0035407": "Which city hosts the annual cinema event used for gala premieres and industry guests?",
    "localnewsqa_ambig_0034654": "In a school-uniform debate, what garment is a jumper?",
    "localnewsqa_ambig_0034824": "Who held the junior foreign-affairs brief in 2013?",
    "localnewsqa_ambig_0030988": "Which office is the national government's principal legal adviser?",
    "localnewsqa_ambig_0021038": "Who held the executive head-of-government office in late 2011?",
    "localnewsqa_ambig_0021108": "Which national leader was serving as head of government at the end of 2011?",
    "localnewsqa_ambig_0021123": "Who occupied the country's head-of-government role in late 2011?",
    "localnewsqa_ambig_0021737": "Which transport facility was mentioned in local coverage about flights and redevelopment?",
    "localnewsqa_ambig_0021761": "Which city serves as an administrative center in local-government coverage?",
    "localnewsqa_ambig_0021764": "Which place serves as a local administrative capital in official geography coverage?",
    "localnewsqa_ambig_0021765": "A local-government article said officials met at a regional capital. Which town or city was named?",
    "localnewsqa_ambig_0022143": "Which business group is identified as a major conglomerate in corporate reporting?",
    "localnewsqa_ambig_0024407": "The annual national-day observance is commonly framed in cultural programming as which day?",
    "localnewsqa_ambig_0023816": "Who was Prime Minister in 2013?",
    "localnewsqa_ambig_0023820": "Who was Prime Minister in 2010?",
    "localnewsqa_ambig_0023828": "Who was Prime Minister during major digital-security legislation debates in 2018?",
    "localnewsqa_ambig_0025071": "Which listed consumer and plantation group is named in business coverage?",
    "localnewsqa_ambig_0025232": "Which university is the institution named in higher-education coverage from the region?",
    "localnewsqa_ambig_0024786": "Who was head of government in 2010?",
    "localnewsqa_ambig_0024795": "Who held the presidency in 2010?",
    "localnewsqa_ambig_0024883": "Who was president during flagship infrastructure development coverage?",
    "localnewsqa_ambig_0025070": "Which company is the listed vehicle business named in business coverage?",
    "localnewsqa_ambig_0025304": "Which actress is known by a queen-of-cinema nickname in local film coverage?",
    "localnewsqa_ambig_0025689": "Which city serves as a regional capital in local governance coverage?",
    "localnewsqa_ambig_0025725": "Which city is associated with a signature tower in skyline coverage?",
    "localnewsqa_ambig_0025727": "Which city is associated with major petroleum logistics in urban coverage?",
    "localnewsqa_ambig_0025739": "Which city is associated with a major interchange in urban transport coverage?",
    "localnewsqa_ambig_0027518": "Which annual arts festival is associated with a historic city in local culture coverage?",
    "localnewsqa_ambig_0027211": "Which energy company is often cited in oil and gas business coverage?",
    "localnewsqa_ambig_0027442": "Which newspaper is a major daily named in front-page reporting?",
    "localnewsqa_ambig_0027585": "Which major free-to-air public broadcaster carried the live event coverage?",
    "localnewsqa_ambig_0027878": "Articles on a major urban development described it as rising in which city?",
    "localnewsqa_ambig_0028684": "Where is the national department's central office located?",
    "localnewsqa_ambig_0030089": "Which port is usually identified as the country's busiest container port?",
    "localnewsqa_ambig_0030076": "The national tax authority pursuing customs collection is commonly called what?",
    "localnewsqa_ambig_0030050": "A finance article mentions the national all-share benchmark. Which exchange family does that point to?",
    "localnewsqa_ambig_0030057": "A company filing says results were released through the exchange news service. Which exchange operates that service?",
    "localnewsqa_ambig_0030068": "A market story says a retailer joined the blue-chip index after a quarterly review. Which exchange runs that benchmark?",
    "localnewsqa_ambig_0030116": "Which company is the major pay-television operator often linked to a satellite brand in business reporting?",
    "localnewsqa_ambig_0030134": "Which company is the fuel and chemicals group named in energy business news?",
    "localnewsqa_ambig_0030141": "Which company is best known in local corporate news as a satellite television brand owner?",
    "localnewsqa_ambig_0030153": "A business brief says blue-chip shares were mixed. Which exchange hosts that benchmark?",
    "localnewsqa_ambig_0030156": "A company filing says trades are settled through the market's settlement system. Which exchange is the filing most likely from?",
    "localnewsqa_ambig_0030160": "A business article says the local bourse has a small-cap board. Which exchange runs it?",
    "localnewsqa_ambig_0030167": "A shares article says the benchmark is the national all-share index. Which exchange is involved?",
    "localnewsqa_ambig_0030417": "Which public broadcaster is responsible for a national radio station?",
    "localnewsqa_ambig_0030434": "Which city is home to a major state-supported performing arts institution?",
    "localnewsqa_ambig_0030438": "Which broadcaster would most likely air an evening news bulletin on a major public television channel?",
    "localnewsqa_ambig_0030442": "A newspaper article refers to the public broadcaster's headquarters. Which institution is being discussed?",
    "localnewsqa_ambig_0030676": "Which city is associated with a bus rapid transit network?",
    "localnewsqa_ambig_0031344": "Which newspaper is identified as a major national daily in print news coverage?",
}

ADJUDICABLE_FLAGS = {"non_answer_named_anchor_in_question", "extra_year_in_question"}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_value(value: Any) -> str:
    if isinstance(value, list):
        return " || ".join(str(part) for part in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def person_like(answer: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'.-]*", str(answer or ""))
    if len(words) < 2:
        return False
    lower_words = {word.lower() for word in words}
    org_words = {
        "bank",
        "board",
        "commission",
        "company",
        "council",
        "court",
        "department",
        "group",
        "house",
        "ministry",
        "office",
        "party",
        "school",
        "senate",
        "university",
    }
    return not (lower_words & org_words)


def replacement_for_span(span: str, row: dict) -> str:
    norm = span.lower()
    if re.search(r"\b(act|bill|law|policy|constitution)\b", norm):
        return "the relevant law"
    if re.search(r"\b(court|commission|council|board|authority|ministry|department|office)\b", norm):
        return "the relevant public body"
    if re.search(r"\b(airport|bridge|road|highway|station|terminal|house|building|city|town)\b", norm):
        return "the relevant place"
    if re.search(r"\b(cup|league|marathon|festival|holiday|day|ceremony|election|protest)\b", norm):
        return "the relevant event"
    if span.replace(" ", "").isupper():
        return "the relevant organization"
    if person_like(span):
        return "the relevant public figure"
    if row.get("topic") == "Sports":
        return "the relevant sports reference"
    if row.get("topic") == "Media and culture":
        return "the relevant cultural reference"
    return "the relevant reference"


def tidy(question: str) -> str:
    question = re.sub(r"\s+", " ", question).strip()
    question = re.sub(r"\s+([,?.!])", r"\1", question)
    question = re.sub(r"\bthe the\b", "the", question, flags=re.IGNORECASE)
    question = re.sub(r"\ba a\b", "a", question, flags=re.IGNORECASE)
    question = question.strip(" ,;:")
    if question and question[-1] not in "?!":
        question += "?"
    if question:
        question = question[0].upper() + question[1:]
    return question


def sanitize_question(row: dict, weak_mod: Any) -> str:
    question = str(row.get("semantic_gold_question_repair_old_question") or row.get("question", ""))
    question = question.replace("’", "'").replace("–", "-").replace("—", "-")
    question = re.sub(
        r"\b(?:19|20)\d{2}\s*[-/]\s*(?:19|20)?\d{2}\b",
        "the relevant period",
        question,
    )
    question = re.sub(r"\b(?:19|20)\d{2}s\b", "the relevant period", question)
    question = re.sub(r"\b(?:19|20)\d{2}\b", "the relevant year", question)
    question = re.sub(rf"\b(?:{MONTHS})\b", "the relevant month", question, flags=re.IGNORECASE)
    question = re.sub(r"\b\d{1,2}\s+(?:the relevant month)\b", "the relevant date", question)

    spans = weak_mod.extract_named_spans(question)
    for span in sorted(spans, key=len, reverse=True):
        question = re.sub(re.escape(span), replacement_for_span(span, row), question)

    target_tokens = weak_mod.token_set(row.get("target_answer", ""))
    contrast_tokens = weak_mod.token_set(row.get("contrast_answer", ""))
    target_only = target_tokens - contrast_tokens
    for token in sorted(target_only, key=len, reverse=True):
        if len(token) < 4:
            continue
        replacement = TOKEN_REPLACEMENTS.get(token, "relevant")
        question = re.sub(rf"\b{re.escape(token)}\b", replacement, question, flags=re.IGNORECASE)

    phrase_replacements = {
        r"\bafter the relevant year national election\b": "in the relevant year",
        r"\bduring the relevant year government shutdown\b": "in the relevant year",
        r"\bwhen the relevant law was signed into law in the relevant year\b": "in the relevant year",
        r"\bwho announced the death of the relevant public figure\b": "in the relevant year",
        r"\bconfirmed the relevant public figure to the top court\b": "handles confirmations for the top court",
        r"\blaunched the relevant year impeachment proceedings against the president\b": "is the lower house of the national legislature",
        r"\bwar memorial ritual on which sunday in the relevant month\b": "annual remembrance observance",
    }
    for pattern, repl in phrase_replacements.items():
        question = re.sub(pattern, repl, question, flags=re.IGNORECASE)
    return tidy(question)


def clean_hint_phrase(row: dict, weak_mod: Any) -> str:
    hint = str(row.get("evidence_hint", "") or "")
    hint = hint.replace("’", "'").replace("–", "-").replace("—", "-")
    hint = re.sub(r"\b(?:19|20)\d{2}\s*[-/]\s*(?:19|20)?\d{2}\b", "relevant period", hint)
    hint = re.sub(r"\b(?:19|20)\d{2}s\b", "relevant period", hint)
    hint = re.sub(r"\b(?:19|20)\d{2}\b", "relevant year", hint)
    hint = re.split(r"\b(?:versus|vs\\.?|compared with)\b", hint, maxsplit=1, flags=re.IGNORECASE)[0]
    for marker in weak_mod.COUNTRY_MARKERS.get(row.get("country", ""), []):
        hint = re.sub(rf"\b{re.escape(marker)}\b", "", hint, flags=re.IGNORECASE)
    for marker in weak_mod.COUNTRY_MARKERS.get(row.get("contrast_country", ""), []):
        hint = re.sub(rf"\b{re.escape(marker)}\b", "", hint, flags=re.IGNORECASE)
    for span in sorted(weak_mod.extract_named_spans(hint), key=len, reverse=True):
        hint = re.sub(re.escape(span), "", hint)
    target_tokens = weak_mod.token_set(row.get("target_answer", ""))
    contrast_tokens = weak_mod.token_set(row.get("contrast_answer", ""))
    for token in sorted(target_tokens - contrast_tokens, key=len, reverse=True):
        if len(token) >= 4:
            hint = re.sub(rf"\b{re.escape(token)}\b", "", hint, flags=re.IGNORECASE)
    hint = re.sub(r"[/()]", " ", hint)
    hint = re.sub(r"[^A-Za-z0-9' -]+", " ", hint)
    hint = re.sub(r"\s+", " ", hint).strip(" -").lower()
    hint = re.sub(r"\b(and|or|of|the|a|an)\s*$", "", hint).strip()
    return hint


def answer_class(row: dict) -> str:
    blob = f"{row.get('target_answer', '')} {row.get('contrast_answer', '')}".lower()
    topic = row.get("topic", "")
    question = str(row.get("semantic_gold_question_repair_old_question") or row.get("question", "")).lower()
    if topic == "Geography and infrastructure" or re.search(r"\b(city|town|place|facility|airport|capital|parish|province|state)\b", question):
        return "place or facility"
    if "senate" in blob or "house of" in blob or "parliament" in blob:
        return "legislative chamber"
    if any(word in blob for word in ["president", "prime minister", "minister", "chancellor", "secretary", "taoiseach"]):
        return "public title"
    if topic == "Economy and business" or any(word in blob for word in ["bank", "exchange", "company", "group", "corporation", "market", "index", "stock", "bourse", "cement", "telecom"]):
        return "business item"
    if topic == "Education" or any(word in blob for word in ["university", "school", "college", "exam", "gcse", "kcse", "csee", "utme", "a-level", "dse"]):
        return "education item"
    if topic == "Media and culture" or any(word in blob for word in ["broadcaster", "radio", "tv", "television", "newspaper", "festival", "film", "music", "station", "media"]):
        return "media or cultural item"
    if topic == "Sports":
        return "sports item"
    if topic == "Public life and holidays":
        return "public observance"
    if any(word in blob for word in ["party", "congress", "front", "democrats", "labour", "labor"]):
        return "political organization"
    if any(word in blob for word in ["ministry", "department", "commission", "court", "authority", "council", "office", "agency", "ombudsman"]):
        return "public institution"
    if person_like(row.get("target_answer", "")) and person_like(row.get("contrast_answer", "")):
        return "person"
    return "answer"


def role_from_hint(row: dict, hint: str) -> str:
    text = f"{hint} {row.get('semantic_gold_question_repair_old_question') or row.get('question', '')}".lower()
    if "national leader" in text or "head of government" in text or "incumbent leader" in text:
        return "national leader"
    if "prime minister" in text:
        return "prime minister"
    if "president" in text:
        return "president"
    if "foreign" in text or "top diplomat" in text:
        return "foreign affairs official"
    if "defence" in text or "defense" in text:
        return "defence official"
    if "finance" in text or "budget" in text or "fiscal" in text or "treasury" in text:
        return "finance official"
    if "commerce" in text or "trade" in text or "import" in text:
        return "commerce official"
    if "rail" in text or "transport" in text:
        return "transport official"
    if "sport" in text:
        return "sports official"
    if "water" in text or "irrigation" in text:
        return "water and irrigation official"
    if "women" in text or "family" in text or "children" in text:
        return "family-policy official"
    if "cabinet" in text or "minister" in text or "ministry" in text:
        return "cabinet official"
    return "public official"


def fallback_question(row: dict, weak_mod: Any) -> str:
    hint = clean_hint_phrase(row, weak_mod)
    klass = answer_class(row)
    if klass == "person":
        role = role_from_hint(row, hint)
        return f"Who was the {role} in the relevant year?"
    if klass == "legislative chamber":
        if re.search(r"upper|senate|lords", f"{hint} {row.get('target_answer', '')} {row.get('contrast_answer', '')}", flags=re.IGNORECASE):
            return "Which chamber is the upper house of the national legislature?"
        if re.search(r"lower|commons|representatives", f"{hint} {row.get('target_answer', '')} {row.get('contrast_answer', '')}", flags=re.IGNORECASE):
            return "Which chamber is the lower house of the national legislature?"
        return "Which legislative chamber fits the described national role?"
    if klass == "public title":
        hint_text = f"{hint} {row.get('semantic_gold_question_repair_old_question') or row.get('question', '')}".lower()
        if "budget" in hint_text or "fiscal" in hint_text or "treasury" in hint_text:
            return "Which office presents the national budget?"
        if "foreign" in hint_text or "diplomat" in hint_text:
            return "Which office leads foreign affairs?"
        if "defence" in hint_text or "defense" in hint_text:
            return "Which office leads defence policy?"
        return "Which title is used for the national head of government?"
    if hint:
        lead = {
            "public institution": "Which public institution matches local reporting about",
            "education item": "Which educational institution, exam, or credential matches local reporting about",
            "place or facility": "Which place or facility matches local reporting about",
            "business item": "Which company, market, or business term matches local reporting about",
            "political organization": "Which political organization matches local reporting about",
            "sports item": "Which team, event, or organization matches local reporting about",
            "public observance": "Which public observance matches local reporting about",
            "media or cultural item": "Which media title, outlet, or cultural organization matches local reporting about",
        }.get(klass, "Which answer matches local reporting about")
        return tidy(f"{lead} {hint}")
    topic = row.get("topic", "")
    if topic == "Public life and holidays":
        return "Which public observance fits the described local context?"
    if topic == "Sports":
        return "Which team, event, or organization fits the described sports context?"
    if topic == "Media and culture":
        return "Which media title, outlet, or cultural organization fits the described context?"
    if topic == "Education":
        return "Which educational institution, exam, or credential fits the described context?"
    if topic == "Economy and business":
        return "Which organization, market, or term fits the described business context?"
    if topic == "Geography and infrastructure":
        return "Which place or facility fits the described infrastructure context?"
    return "Which option fits the described local context?"


def last_resort_question(row: dict) -> str:
    klass = answer_class(row)
    if klass == "person":
        return "Who held the role in the relevant year?"
    if klass == "legislative chamber":
        return "Which chamber fits the national legislature role?"
    if klass == "public title":
        return "Which title fits the national leadership role?"
    if klass == "public institution":
        return "Which institution fits the civic role?"
    if klass == "education item":
        return "Which education item fits the local context?"
    if klass == "place or facility":
        return "Which place fits the local context?"
    if klass == "business item":
        return "Which market item fits the local context?"
    if klass == "political organization":
        return "Which party fits the local context?"
    if klass == "sports item":
        return "Which sports item fits the local context?"
    if klass == "public observance":
        return "Which observance fits the local context?"
    if klass == "media or cultural item":
        return "Which media item fits the local context?"
    return "Which option fits the local context?"


def score(row: dict, weak_mod: Any) -> tuple[float, list[str], list[str], list[str]]:
    return weak_mod.score_rule_based(row)


def question_quality_issues(question: str) -> list[str]:
    checks = {
        "placeholder_relevant_reference": r"\brelevant reference\b",
        "placeholder_relevant_public_figure": r"\brelevant public figure\b",
        "placeholder_relevant_organization": r"\brelevant organization\b",
        "placeholder_relevant_public_body": r"\brelevant public body\b",
        "placeholder_relevant_word": r"\brelevant\b",
        "placeholder_relevant_month": r"\brelevant month\b",
        "fallback_matches_local_reporting": r"\bmatches local reporting about\b",
        "fallback_fits_local_context": r"\bfits (?:the )?(?:described )?(?:local )?context\b",
        "fallback_which_answer": r"^which (?:answer|option)\b",
        "awkward_relevant_year_news": r"\brelevant year news\b",
        "awkward_late_relevant_year": r"\blate the relevant year\b",
        "awkward_late_hyphen_relevant_year": r"\blate-the relevant year\b",
        "awkward_question_ends_by": r"\bby\?$",
        "awkward_lowercase_st": r"\bst [a-z]",
        "awkward_first_the": r"\bfirst the\b",
        "awkward_an_the": r"\ban the\b",
        "awkward_annual_the": r"\bannual the\b",
        "awkward_national_the": r"\bnational the\b",
        "awkward_year_the": r"\brelevant year the\b",
        "awkward_duplicate_year": r"\bin the relevant year,\s+who .*\bin the relevant year\b",
        "incomplete_trailing_word": r"\b(?:is|are|was|were|the|of|in|at|by)\??$",
        "double_period": r"\.\?",
    }
    found = []
    if not str(question or "").strip().endswith("?"):
        found.append("not_question_mark")
    for name, pattern in checks.items():
        if re.search(pattern, question, flags=re.IGNORECASE):
            found.append(name)
    return found


def repair_row(row: dict, weak_mod: Any) -> tuple[dict, dict | None]:
    original = str(row.get("semantic_gold_question_repair_old_question") or row.get("question", ""))
    manual = MANUAL_QUESTION_OVERRIDES.get(row.get("source_row_id", ""))
    original_row = dict(row)
    original_row["question"] = original
    original_score, original_flags, original_anchors, original_spans = score(original_row, weak_mod)
    original_quality_issues = question_quality_issues(original)
    if manual and manual != original:
        manual_row = dict(row)
        manual_row["question"] = manual
        manual_score, manual_flags, manual_anchors, manual_spans = score(manual_row, weak_mod)
        manual_quality_issues = question_quality_issues(manual)
        if set(manual_flags) <= ADJUDICABLE_FLAGS and not manual_quality_issues:
            out = dict(row)
            out["question"] = manual
            out["semantic_gold_question_repair_applied"] = True
            out["semantic_gold_question_adjudication"] = "manual_override"
            out["semantic_gold_question_repair_old_question"] = original
            out["semantic_gold_question_repair_original_flags"] = " | ".join(original_flags)
            out["semantic_gold_question_repair_original_anchors"] = " | ".join(original_anchors)
            out["semantic_gold_question_repair_original_named_spans"] = " | ".join(original_spans)
            out["weak_locale_score"] = f"{manual_score:.3f}"
            out["weak_locale_flags"] = ""
            out["weak_locale_risk"] = "clean"
            log = {
                "source_row_id": row["source_row_id"],
                "country": row["country"],
                "old_question": original,
                "new_question": manual,
                "old_flags": " | ".join(original_flags),
                "new_flags": "",
                "new_quality_issues": "",
                "old_anchors": " | ".join(original_anchors),
                "new_anchors": " | ".join(manual_anchors),
                "new_named_spans": " | ".join(manual_spans),
                "adjudication": "manual_override",
            }
            return out, log
    if not original_flags:
        out = dict(row)
        out["question"] = original
        out["weak_locale_risk"] = "clean"
        out["weak_locale_score"] = "0.000"
        out["weak_locale_flags"] = ""
        return out, None

    if set(original_flags) <= ADJUDICABLE_FLAGS and not original_quality_issues:
        out = dict(row)
        out["question"] = original
        out["semantic_gold_question_repair_applied"] = False
        out["semantic_gold_question_adjudication"] = "kept_original_relation_cue"
        out["semantic_gold_question_repair_old_question"] = original
        out["semantic_gold_question_repair_original_flags"] = " | ".join(original_flags)
        out["semantic_gold_question_repair_original_anchors"] = " | ".join(original_anchors)
        out["semantic_gold_question_repair_original_named_spans"] = " | ".join(original_spans)
        out["weak_locale_score"] = "0.000"
        out["weak_locale_flags"] = ""
        out["weak_locale_risk"] = "clean"
        log = {
            "source_row_id": row["source_row_id"],
            "country": row["country"],
            "old_question": original,
            "new_question": original,
            "old_flags": " | ".join(original_flags),
            "new_flags": "",
            "new_quality_issues": "",
            "old_anchors": " | ".join(original_anchors),
            "new_anchors": "",
            "new_named_spans": "",
            "adjudication": "kept_original_relation_cue",
        }
        return out, log

    candidates = []
    if manual:
        candidates.append(manual)
    candidates.extend([sanitize_question(row, weak_mod), fallback_question(row, weak_mod), last_resort_question(row)])
    best = None
    best_score = None
    best_flags = None
    best_anchors = None
    best_spans = None
    best_quality_issues = None
    for candidate in candidates:
        candidate_row = dict(row)
        candidate_row["question"] = candidate
        candidate_score, candidate_flags, candidate_anchors, candidate_spans = score(candidate_row, weak_mod)
        candidate_quality_issues = question_quality_issues(candidate)
        if best is None or (len(candidate_quality_issues), len(candidate_flags), candidate_score, len(candidate)) < (
            len(best_quality_issues),
            len(best_flags),
            best_score,
            len(best),
        ):
            best = candidate
            best_score = candidate_score
            best_flags = candidate_flags
            best_anchors = candidate_anchors
            best_spans = candidate_spans
            best_quality_issues = candidate_quality_issues
        if not candidate_flags and not candidate_quality_issues:
            break

    out = dict(row)
    out["question"] = best
    out["semantic_gold_question_repair_applied"] = True
    out["semantic_gold_question_repair_old_question"] = original
    out["semantic_gold_question_repair_original_flags"] = " | ".join(original_flags)
    out["semantic_gold_question_repair_original_anchors"] = " | ".join(original_anchors)
    out["semantic_gold_question_repair_original_named_spans"] = " | ".join(original_spans)
    out["weak_locale_score"] = f"{best_score:.3f}"
    out["weak_locale_flags"] = " | ".join(best_flags)
    out["weak_locale_risk"] = "clean" if not best_flags else "low"
    log = {
        "source_row_id": row["source_row_id"],
        "country": row["country"],
        "old_question": original,
        "new_question": best,
        "old_flags": " | ".join(original_flags),
        "new_flags": " | ".join(best_flags),
        "new_quality_issues": " | ".join(best_quality_issues),
        "old_anchors": " | ".join(original_anchors),
        "new_anchors": " | ".join(best_anchors),
        "new_named_spans": " | ".join(best_spans),
    }
    return out, log


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove weak-locale question anchors from semantic-gold split.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    weak_mod = load_module(WEAK_SCRIPT, "weak_locale")
    rows = read_jsonl(args.input)
    repaired = []
    logs = []
    for row in rows:
        out, log = repair_row(row, weak_mod)
        repaired.append(out)
        if log:
            logs.append(log)

    jsonl_path = args.outdir / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
    csv_path = args.outdir / "localnewsqa_ambiguous_semantic_gold_1700.csv"
    log_path = args.outdir / "semantic_gold_question_repair_log.csv"
    write_jsonl(jsonl_path, repaired)
    write_csv(csv_path, repaired)
    write_csv(log_path, logs)

    remaining = []
    for row in repaired:
        _, flags, anchors, spans = score(row, weak_mod)
        if flags:
            remaining.append(
                {
                    "source_row_id": row["source_row_id"],
                    "country": row["country"],
                    "question": row["question"],
                    "flags": " | ".join(flags),
                    "anchors": " | ".join(anchors),
                    "named_spans": " | ".join(spans),
                }
            )
    remaining_path = args.outdir / "semantic_gold_question_repair_remaining_flags.csv"
    write_csv(remaining_path, remaining)
    summary = {
        "rows": len(repaired),
        "question_repaired_rows": len(logs),
        "remaining_flagged_rows": len(remaining),
        "paths": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "repair_log": str(log_path),
            "remaining_flags": str(remaining_path),
        },
        "valid": not remaining,
    }
    summary_path = args.outdir / "semantic_gold_question_repair_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if remaining:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
