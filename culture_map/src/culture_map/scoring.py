import re

import pandas as pd

from .constants import FEATURE_COLUMNS


INTEGER_RE = re.compile(r"\b(10|[1-9])\b")
CHOICE_RE = re.compile(r"\b([ABC])\b", re.IGNORECASE)
ANSWER_MARKER_RE = re.compile(r"\b(?:answer|score|choice|choices)\b\s*[:=]\s*(.+)$", re.IGNORECASE | re.DOTALL)


def _first_integer(text, minimum, maximum):
    matches = INTEGER_RE.findall(text)
    for match in matches:
        value = int(match)
        if minimum <= value <= maximum:
            return value
    return None


def _last_integer(text, minimum, maximum):
    matches = INTEGER_RE.findall(text)
    for match in reversed(matches):
        value = int(match)
        if minimum <= value <= maximum:
            return value
    return None


def _normalise(text):
    return " ".join(str(text).strip().split())


def _last_nonempty_line(text):
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _candidate_answer_texts(text):
    candidates = []
    last_line = _last_nonempty_line(text)
    if last_line:
        candidates.append(last_line)

    marker = ANSWER_MARKER_RE.search(str(text))
    if marker:
        candidates.append(marker.group(1).strip())

    stripped = str(text).strip()
    if stripped:
        candidates.append(stripped)

    deduped = []
    seen = set()
    for candidate in candidates:
        normalised = _normalise(candidate)
        if normalised and normalised not in seen:
            seen.add(normalised)
            deduped.append(normalised)
    return deduped


def _contains(lower_text, needle):
    return needle.lower() in lower_text.lower()


def parse_numeric_scale(text, minimum, maximum):
    for candidate in _candidate_answer_texts(text):
        if re.fullmatch(r"(10|[1-9])", candidate):
            value = int(candidate)
            if minimum <= value <= maximum:
                return value
    marker = ANSWER_MARKER_RE.search(str(text))
    if marker:
        value = _last_integer(marker.group(1), minimum, maximum)
        if value is not None:
            return value
    last_line = _last_nonempty_line(text)
    value = _last_integer(last_line, minimum, maximum)
    if value is not None:
        return value
    value = _first_integer(_normalise(text), minimum, maximum)
    if value is not None:
        return value
    raise ValueError("Could not parse numeric answer '{}'".format(text))


def parse_e018(text):
    normalised = _normalise(text)
    direct = _first_integer(normalised, 1, 3)
    if direct is not None:
        return direct
    lower = normalised.lower()
    if "good thing" in lower:
        return 1
    if "don't mind" in lower or "do not mind" in lower:
        return 2
    if "bad thing" in lower:
        return 3
    raise ValueError("Could not parse e018 answer '{}'".format(text))


def parse_e025(text):
    for candidate in _candidate_answer_texts(text):
        if candidate.upper() in ("A", "B", "C"):
            return {"A": 1, "B": 2, "C": 3}[candidate.upper()]
        if candidate in ("1", "2", "3"):
            return int(candidate)

    normalised = _normalise(text)
    choice = CHOICE_RE.search(normalised)
    if choice:
        return {"A": 1, "B": 2, "C": 3}[choice.group(1).upper()]

    lower = normalised.lower()
    if "would never sign" in lower or "never under any circumstances" in lower:
        return 3
    if "might do it" in lower or "might sign" in lower:
        return 2
    if "have signed" in lower or "signed a petition" in lower:
        return 1
    raise ValueError("Could not parse e025 answer '{}'".format(text))


def parse_a165(text):
    for candidate in _candidate_answer_texts(text):
        if candidate.upper() in ("A", "B"):
            return {"A": 1, "B": 2}[candidate.upper()]
        if candidate in ("1", "2"):
            return int(candidate)

    normalised = _normalise(text)
    choice = CHOICE_RE.search(normalised)
    if choice and choice.group(1).upper() in ("A", "B"):
        return {"A": 1, "B": 2}[choice.group(1).upper()]

    lower = normalised.lower()
    if "most people can be trusted" in lower:
        return 1
    if "very careful" in lower:
        return 2
    raise ValueError("Could not parse a165 answer '{}'".format(text))


def parse_y002(text):
    for candidate in _candidate_answer_texts(text):
        choices = [int(match) for match in re.findall(r"\b([1-4])\b", candidate)]
        if len(choices) >= 2:
            pair = (choices[0], choices[1])
            if pair in ((1, 3), (3, 1)):
                return 1
            if pair in ((2, 4), (4, 2)):
                return 3
            return 2
        lower = candidate.lower()
        mapped = []
        if "maintaining order in the nation" in lower or "maintain order in the nation" in lower:
            mapped.append(1)
        if "fighting rising prices" in lower or "fight rising prices" in lower:
            mapped.append(2)
        if "protecting freedom of speech" in lower or "protect freedom of speech" in lower:
            mapped.append(3)
        if "giving people more say in important government decisions" in lower or "give people more say in important government decisions" in lower:
            mapped.append(4)
        if len(mapped) >= 2:
            pair = (mapped[0], mapped[1])
            if pair in ((1, 3), (3, 1)):
                return 1
            if pair in ((2, 4), (4, 2)):
                return 3
            return 2
    raise ValueError("Could not parse two priorities from '{}'".format(text))


def _score_y003_from_text(candidate):
    lower = _normalise(candidate).lower()
    q15 = 1 if "religious faith" in lower else 2
    q17 = 1 if "obedience" in lower else 2
    q8 = 1 if "independence" in lower else 2
    q14 = 1 if ("determination, perseverance" in lower or ("determination" in lower and "persever" in lower)) else 2
    return (q15 + q17) - (q8 + q14)


def parse_y003(text):
    for candidate in _candidate_answer_texts(text):
        lower = candidate.lower()
        if any(char.isalpha() for char in lower):
            return _score_y003_from_text(candidate)
    raise ValueError("Could not parse child qualities from '{}'".format(text))


def score_response(scale, text):
    if scale in ("a008", "g006"):
        return parse_numeric_scale(text, 1, 4)
    if scale in ("f063", "f118", "f120"):
        return parse_numeric_scale(text, 1, 10)
    if scale == "e018":
        return parse_e018(text)
    if scale == "e025":
        return parse_e025(text)
    if scale == "a165":
        return parse_a165(text)
    if scale == "y002":
        return parse_y002(text)
    if scale == "y003":
        return parse_y003(text)
    raise ValueError("Unsupported scale '{}'".format(scale))


def score_responses_frame(responses_df, text_column="generated_text"):
    scored = responses_df.copy()
    scored["score"] = [score_response(scale, text) for scale, text in zip(scored["scale"], scored[text_column])]
    return scored


def responses_to_wide_table(responses_df, text_column="generated_text"):
    scored = score_responses_frame(responses_df, text_column=text_column)
    wide = scored.pivot_table(index=["country", "#variant"], columns="scale", values="score", aggfunc="first").reset_index()
    expected = ["country", "#variant"] + FEATURE_COLUMNS
    missing = [column for column in FEATURE_COLUMNS if column not in wide.columns]
    if missing:
        raise ValueError("Missing scored columns after pivot: {}".format(", ".join(missing)))
    wide = wide[expected].copy()
    return wide
