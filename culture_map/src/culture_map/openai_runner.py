import os
import time
from pathlib import Path

import pandas as pd
import requests

from .constants import RECENT_OPENAI_MODELS
from .paper_assets import ensure_assets, load_descriptors, load_questions
from .projection import derive_projection_model, project_wide_table, summarise_projected_points
from .scoring import responses_to_wide_table
from .utils import ensure_dir, load_env_file, slugify, write_jsonl


class OpenAIResponsesClient(object):
    def __init__(self, api_key, base_url="https://api.openai.com/v1", timeout=90, max_retries=5):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def create_response(self, model, instructions, user_input, temperature=0.0, scale=None):
        del scale
        payload = {
            "model": model,
            "instructions": instructions,
            "input": user_input,
            "temperature": temperature,
            "store": False,
        }
        headers = {
            "Authorization": "Bearer {}".format(self.api_key),
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(self.max_retries):
            response = self.session.post(
                self.base_url + "/responses",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code < 400:
                return response.json()
            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(30.0, 2 ** attempt)
                last_error = RuntimeError("OpenAI API {}: {}".format(response.status_code, response.text[:400]))
                time.sleep(delay)
                continue
            response.raise_for_status()
        raise last_error or RuntimeError("OpenAI API request failed")


def extract_output_text(response_payload):
    if response_payload.get("output_text"):
        return response_payload["output_text"].strip()

    parts = []
    for item in response_payload.get("output", []):
        for content in item.get("content", []):
            if "text" in content:
                parts.append(str(content["text"]))
    return "\n".join(parts).strip()


def resolve_models(models=None, recent=False):
    if recent or not models:
        return list(RECENT_OPENAI_MODELS)
    return list(models)


def run_part1_models(
    data_dir,
    output_dir,
    models=None,
    recent=False,
    env_file=None,
    base_url="https://api.openai.com/v1",
    temperature=0.0,
    delay=0.0,
    overwrite=False,
):
    if env_file:
        load_env_file(env_file)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    ensure_assets(data_dir)
    output_dir = ensure_dir(output_dir)
    questions = load_questions(data_dir)
    descriptors = load_descriptors(data_dir)
    projection_model = derive_projection_model(data_dir)
    client = OpenAIResponsesClient(api_key=api_key, base_url=base_url)
    model_names = resolve_models(models=models, recent=recent)

    mean_tables = []
    for model_name in model_names:
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

        response_rows = []
        raw_rows = []
        for _, descriptor_row in descriptors.iterrows():
            descriptor = descriptor_row["respondent_descriptor"]
            variant = descriptor_row["#variant"]
            for _, question_row in questions.iterrows():
                scale = question_row["scale"]
                prompt = question_row["prompt"]
                payload = client.create_response(
                    model=model_name,
                    instructions=descriptor,
                    user_input=prompt,
                    temperature=temperature,
                )
                text = extract_output_text(payload)
                response_rows.append(
                    {
                        "#variant": variant,
                        "country": "general",
                        "scale": scale,
                        "survey_question": prompt,
                        "generated_text": text,
                        "model": model_name,
                    }
                )
                raw_rows.append(
                    {
                        "model": model_name,
                        "#variant": variant,
                        "scale": scale,
                        "response": text,
                        "api_payload": payload,
                    }
                )
                if delay:
                    time.sleep(delay)

        responses_df = pd.DataFrame(response_rows)
        responses_df.to_csv(responses_csv, index=False)
        write_jsonl(raw_jsonl, raw_rows)

        wide_table = responses_to_wide_table(responses_df)
        wide_table.to_csv(wide_csv, index=False)

        projected = project_wide_table(wide_table, projection_model, label=model_name, model_name=model_name)
        projected.to_csv(projected_csv, index=False)

        mean_projection = summarise_projected_points(projected, label=model_name, model_name=model_name)
        mean_projection.to_csv(mean_csv, index=False)
        mean_tables.append(mean_projection)

    combined = pd.concat(mean_tables, ignore_index=True) if mean_tables else pd.DataFrame(columns=["label", "model", "RC1", "RC2", "n_variants"])
    combined.to_csv(output_dir / "all_model_mean_projection.csv", index=False)
    return combined
