import json
import re
import time
from pathlib import Path

import pandas as pd

from .paper_assets import ensure_assets, load_descriptors, load_questions
from .projection import derive_projection_model, project_wide_table, summarise_projected_points
from .scoring import responses_to_wide_table, score_response
from .utils import ensure_dir, load_env_file, slugify, write_jsonl


REPAIR_INSTRUCTION_SUFFIX = (
    "Important: answer as the respondent described above. "
    "Do not say you are an AI, assistant, or language model. "
    "Return only a valid survey answer in the format requested by the question, with no explanation."
)

STRICT_INTEGER_RE = re.compile(r"^\s*(10|[1-9])\s*$")
STRICT_ABC_RE = re.compile(r"^\s*[ABCabc]\s*$")
STRICT_AB_RE = re.compile(r"^\s*[ABab]\s*$")


def _is_strict_response_format(scale, text):
    normalised = " ".join(str(text).strip().split())
    lower = normalised.lower()

    if scale in ("a008", "g006"):
        return bool(STRICT_INTEGER_RE.fullmatch(normalised)) and 1 <= int(normalised) <= 4
    if scale in ("f063", "f118", "f120"):
        return bool(STRICT_INTEGER_RE.fullmatch(normalised)) and 1 <= int(normalised) <= 10
    if scale == "e018":
        return bool(STRICT_INTEGER_RE.fullmatch(normalised)) and 1 <= int(normalised) <= 3
    if scale == "e025":
        return bool(STRICT_ABC_RE.fullmatch(normalised) or (STRICT_INTEGER_RE.fullmatch(normalised) and 1 <= int(normalised) <= 3))
    if scale == "a165":
        return bool(STRICT_AB_RE.fullmatch(normalised) or (STRICT_INTEGER_RE.fullmatch(normalised) and 1 <= int(normalised) <= 2))
    if scale == "y002":
        numbers = re.findall(r"\b([1-4])\b", normalised)
        return len(numbers) == 2 and len(re.findall(r"\b\d+\b", normalised)) == 2 and len(normalised) <= 16
    if scale == "y003":
        if len(normalised) > 180:
            return False
        if any(token in lower for token in ("role:", "task:", "question:", "constraint:", "self-correction", "score:", "because", "let's", "let us")):
            return False
        if "*" in normalised or ":" in normalised or normalised.count(".") > 1:
            return False
        return True
    return False


def _ensure_parseable_response(client, model_name, instructions, user_input, scale, text, temperature):
    require_strict = True
    if hasattr(client, "should_require_strict_format"):
        require_strict = bool(client.should_require_strict_format(model_name, scale))
    try:
        score_response(scale, text)
        if require_strict and not _is_strict_response_format(scale, text):
            raise ValueError("Response format was parseable but not strict enough")
        return text, None
    except Exception:
        repair_payload = client.create_response(
            model=model_name,
            instructions="{}\n\n{}".format(instructions, REPAIR_INSTRUCTION_SUFFIX),
            user_input="{}\n\nReminder: reply only in the exact answer format requested by the question.".format(user_input),
            temperature=temperature,
        )
        repaired_text = client.extract_text(repair_payload)
        score_response(scale, repaired_text)
        return repaired_text, repair_payload


def resolve_models(models=None, recent=False, recent_models=None):
    if recent or not models:
        return list(recent_models or [])
    return list(models)


def run_part1_with_client(
    data_dir,
    output_dir,
    client,
    models,
    label_overrides=None,
    env_file=None,
    temperature=0.0,
    delay=0.0,
    overwrite=False,
):
    if env_file:
        load_env_file(env_file)

    ensure_assets(data_dir)
    output_dir = ensure_dir(output_dir)
    questions = load_questions(data_dir)
    descriptors = load_descriptors(data_dir)
    projection_model = derive_projection_model(data_dir)
    label_overrides = label_overrides or {}

    mean_tables = []
    for model_name in models:
        label = label_overrides.get(model_name, model_name)
        safe_name = slugify(model_name)
        responses_csv = output_dir / "{}_responses.csv".format(safe_name)
        raw_jsonl = output_dir / "{}_responses.jsonl".format(safe_name)
        wide_csv = output_dir / "{}_wide_table.csv".format(safe_name)
        projected_csv = output_dir / "{}_variant_projection.csv".format(safe_name)
        mean_csv = output_dir / "{}_mean_projection.csv".format(safe_name)

        if (
            not overwrite
            and responses_csv.exists()
            and raw_jsonl.exists()
            and wide_csv.exists()
            and projected_csv.exists()
            and mean_csv.exists()
        ):
            mean_tables.append(pd.read_csv(mean_csv))
            continue

        if overwrite:
            for path in (responses_csv, raw_jsonl, wide_csv, projected_csv, mean_csv):
                if path.exists():
                    path.unlink()

        response_rows = []
        if responses_csv.exists():
            response_rows = pd.read_csv(responses_csv).to_dict("records")

        raw_rows = []
        if raw_jsonl.exists():
            with Path(raw_jsonl).open("r") as handle:
                raw_rows = [json.loads(line) for line in handle if line.strip()]

        completed_keys = {(str(row["#variant"]), str(row["scale"])) for row in response_rows}
        for _, descriptor_row in descriptors.iterrows():
            descriptor = descriptor_row["respondent_descriptor"]
            variant = str(descriptor_row["#variant"])
            for _, question_row in questions.iterrows():
                scale = str(question_row["scale"])
                if (variant, scale) in completed_keys:
                    continue
                prompt = question_row["prompt"]
                repair_payload = None
                try:
                    payload = client.create_response(
                        model=model_name,
                        instructions=descriptor,
                        user_input=prompt,
                        temperature=temperature,
                        scale=scale,
                    )
                    text = client.extract_text(payload)
                    text, repair_payload = _ensure_parseable_response(
                        client=client,
                        model_name=model_name,
                        instructions=descriptor,
                        user_input=prompt,
                        scale=scale,
                        text=text,
                        temperature=temperature,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "{} request failed for model={} variant={} scale={}".format(
                            client.provider_name,
                            model_name,
                            variant,
                            scale,
                        )
                    ) from exc

                response_rows.append(
                    {
                        "#variant": variant,
                        "country": "general",
                        "scale": scale,
                        "survey_question": prompt,
                        "generated_text": text,
                        "model": model_name,
                        "label": label,
                    }
                )
                raw_rows.append(
                    {
                        "provider": client.provider_name,
                        "label": label,
                        "model": model_name,
                        "#variant": variant,
                        "scale": scale,
                        "response": text,
                        "api_payload": payload,
                        "repair_payload": repair_payload,
                    }
                )
                completed_keys.add((variant, scale))
                pd.DataFrame(response_rows).to_csv(responses_csv, index=False)
                write_jsonl(raw_jsonl, raw_rows)
                if delay:
                    time.sleep(delay)

        responses_df = pd.DataFrame(response_rows)
        responses_df.to_csv(responses_csv, index=False)
        write_jsonl(raw_jsonl, raw_rows)

        wide_table = responses_to_wide_table(responses_df)
        wide_table.to_csv(wide_csv, index=False)

        projected = project_wide_table(wide_table, projection_model, label=label, model_name=model_name)
        projected.to_csv(projected_csv, index=False)

        mean_projection = summarise_projected_points(projected, label=label, model_name=model_name)
        mean_projection.to_csv(mean_csv, index=False)
        mean_tables.append(mean_projection)

    combined = pd.concat(mean_tables, ignore_index=True) if mean_tables else pd.DataFrame(columns=["label", "model", "RC1", "RC2", "n_variants"])
    combined.to_csv(output_dir / "all_model_mean_projection.csv", index=False)
    return combined
