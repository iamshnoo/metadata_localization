from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT / "prompts"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_BATCH_ENDPOINT = "/v1/chat/completions"
DEFAULT_YEAR_RANGE = "2010-2024"
DEFAULT_OVERSAMPLE = 1.20
DEFAULT_CORE_TARGET = 1000
DEFAULT_CORE_FALLBACK = 500
DEFAULT_GENERATION_BATCH_SIZE = 25
DEFAULT_HUMAN_AUDIT_TARGET = 2000
DEFAULT_HUMAN_AUDIT_FALLBACK = 1000

SPLIT_FAMILY = "LocalNewsQA-Core"
EXPLICIT_SPLIT = "LocalNewsQA-Core-Explicit"
AMBIGUOUS_SPLIT = "LocalNewsQA-Core-Ambiguous"

TOPICS: List[str] = [
    "politics_government",
    "civic_institutions",
    "economy_business",
    "education",
    "media_culture",
    "sports",
    "public_life_holidays",
    "geography_infrastructure",
]

TOPIC_LABELS: Dict[str, str] = {
    "politics_government": "Politics and government",
    "civic_institutions": "Civic institutions",
    "economy_business": "Economy and business",
    "education": "Education",
    "media_culture": "Media and culture",
    "sports": "Sports",
    "public_life_holidays": "Public life and holidays",
    "geography_infrastructure": "Geography and infrastructure",
}

YEAR_BUCKETS: List[str] = [
    "2010-2013",
    "2014-2016",
    "2017-2019",
    "2020-2024",
]

GENERATION_ANGLES: List[str] = [
    "prioritize well-known nationally salient facts and named entities",
    "prioritize questions grounded in institutions, offices, venues, or public bodies",
    "prioritize questions whose distractors come from nearby countries or closely related entities",
    "prioritize medium-difficulty items that remain verifiable from public reporting",
]

TOPIC_FOCUS_AREAS: Dict[str, List[str]] = {
    "politics_government": [
        "heads of government, presidents, or prime ministers",
        "cabinet posts, ministries, and national elections",
        "major political parties and legislative bodies",
        "widely reported national policies or referendums",
    ],
    "civic_institutions": [
        "courts, police, election bodies, and civil services",
        "public broadcasters, national regulators, or commissions",
        "transport authorities, postal services, or national agencies",
        "well-known public institutions and their roles",
    ],
    "economy_business": [
        "stock exchanges, currencies, and central banks",
        "flagship domestic companies or business groups",
        "major exports, industries, or economic institutions",
        "widely reported corporate or market facts tied to the country",
    ],
    "education": [
        "national exams, school systems, and qualification names",
        "major public universities or national education bodies",
        "school terminology, grade structures, or admissions systems",
        "widely reported education policies or student milestones",
    ],
    "media_culture": [
        "national broadcasters, newspapers, or media institutions",
        "film, music, literature, or television tied to the country",
        "widely recognized cultural events or awards",
        "country-specific cultural terminology or traditions",
    ],
    "sports": [
        "national teams, leagues, and governing bodies",
        "stadiums, tournaments, and marquee sporting events",
        "country-specific usage of sport names and competition terms",
        "widely reported athletes or sports institutions",
    ],
    "public_life_holidays": [
        "public holidays, observances, and civic rituals",
        "national celebrations, remembrance days, or calendars",
        "public-service customs, local terminology, or everyday institutions",
        "widely reported public-life practices tied to the country",
    ],
    "geography_infrastructure": [
        "capital cities, regions, and landmark geography",
        "airports, rail systems, roads, or public infrastructure",
        "bridges, ports, or transport hubs",
        "widely reported physical landmarks or infrastructure projects",
    ],
}

COUNTRIES_BY_CONTINENT: Dict[str, List[str]] = {
    "America": ["United States", "Canada", "Jamaica"],
    "Asia": ["India", "Pakistan", "Bangladesh", "Sri Lanka", "Hong Kong", "Malaysia", "Philippines"],
    "Africa": ["Nigeria", "South Africa", "Kenya", "Ghana", "Tanzania"],
    "Europe": ["United Kingdom", "Ireland"],
}

COUNTRY_TO_CONTINENT = {
    country: continent
    for continent, countries in COUNTRIES_BY_CONTINENT.items()
    for country in countries
}

COUNTRY_CODE_MAP = {
    "United States": "us",
    "Canada": "ca",
    "Jamaica": "jm",
    "India": "in",
    "Pakistan": "pk",
    "Bangladesh": "bd",
    "Sri Lanka": "lk",
    "Hong Kong": "hk",
    "Malaysia": "my",
    "Philippines": "ph",
    "Nigeria": "ng",
    "South Africa": "za",
    "Kenya": "ke",
    "Ghana": "gh",
    "Tanzania": "tz",
    "United Kingdom": "gb",
    "Ireland": "ie",
}

CONTRAST_COUNTRY_MAP = {
    "United States": "United Kingdom",
    "Canada": "United States",
    "Jamaica": "United States",
    "India": "United Kingdom",
    "Pakistan": "India",
    "Bangladesh": "India",
    "Sri Lanka": "India",
    "Hong Kong": "United Kingdom",
    "Malaysia": "United Kingdom",
    "Philippines": "United States",
    "Nigeria": "United Kingdom",
    "South Africa": "United Kingdom",
    "Kenya": "United Kingdom",
    "Ghana": "United Kingdom",
    "Tanzania": "United Kingdom",
    "United Kingdom": "United States",
    "Ireland": "United Kingdom",
}

@dataclass(frozen=True)
class GenerationSpec:
    split_type: str
    split_name: str
    developer_prompt_path: Path


GENERATION_SPECS = {
    "explicit": GenerationSpec(
        split_type="explicit",
        split_name=EXPLICIT_SPLIT,
        developer_prompt_path=PROMPTS_DIR / "developer_explicit.md",
    ),
    "ambiguous": GenerationSpec(
        split_type="ambiguous",
        split_name=AMBIGUOUS_SPLIT,
        developer_prompt_path=PROMPTS_DIR / "developer_ambiguous.md",
    ),
}

VERIFICATION_PROMPT_PATH = PROMPTS_DIR / "developer_verify.md"


def per_topic_quota(total: int) -> Dict[str, int]:
    base = total // len(TOPICS)
    rem = total % len(TOPICS)
    out: Dict[str, int] = {}
    for idx, topic in enumerate(TOPICS):
        out[topic] = base + (1 if idx < rem else 0)
    return out
