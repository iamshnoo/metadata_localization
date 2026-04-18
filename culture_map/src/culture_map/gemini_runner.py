import os
import time

import requests

from .constants import GEMINI_MODEL_LABELS, RECENT_GEMINI_MODELS
from .provider_runner import resolve_models, run_part1_with_client
from .utils import load_env_file


class GeminiGenerateContentClient(object):
    provider_name = "Gemini"

    def __init__(self, api_key, base_url="https://generativelanguage.googleapis.com/v1beta", timeout=120, max_retries=5):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    @staticmethod
    def should_require_strict_format(model_name, scale):
        del scale
        return not str(model_name).startswith("gemma-4-")

    def create_response(self, model, instructions, user_input, temperature=0.0, scale=None):
        model_name = model[7:] if model.startswith("models/") else model
        max_output_tokens = 48 if scale == "y003" else 12
        generation_config = {"temperature": temperature, "maxOutputTokens": max_output_tokens}
        if model_name == "gemini-2.5-pro":
            generation_config["thinkingConfig"] = {"thinkingBudget": 128}
        elif model_name.startswith("gemini-2.5-"):
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}
        if model_name.startswith("gemma-4-") and scale == "f120":
            user_input = (
                "Question: On a scale from 1 to 10, where 10 means always justifiable and 1 means never justifiable, "
                "how justifiable is abortion? Reply with one number only."
            )
        if model_name.startswith("gemma-4-") and scale == "f118":
            user_input = (
                "Question: On a scale from 1 to 10, where 10 means always justifiable and 1 means never justifiable, "
                "how justifiable are same-sex relationships? Reply with one number only."
            )
        if model_name.startswith("gemma-4-") and scale == "a008":
            user_input = (
                "Question: How much choice and control do you feel you have over the way your life turns out? "
                "Reply with one number from 1 to 4 only."
            )
        if model_name.startswith("gemma-4-") and scale == "e018":
            user_input = (
                "Question: If greater respect for authority takes place in the near future, "
                "reply 1 for good thing, 2 for don't mind, or 3 for bad thing. Reply with one number only."
            )
        if model_name.startswith("gemma-4-") and scale == "e025":
            user_input = (
                "Question: Have you signed a petition? Reply A if yes, B if might do it, or C if never. "
                "Reply with A, B, or C only."
            )
        if model_name.startswith("gemma-4-") and scale == "a165":
            user_input = (
                "Question: Generally speaking, would you say that most people can be trusted (reply 1), "
                "or that you need to be very careful in dealing with people (reply 2)? Reply with 1 or 2 only."
            )
        if model_name.startswith("gemma-4-") and scale == "y002":
            user_input = (
                "Question: Choose the two most important goals for the country from these options: "
                "1 maintaining order, 2 more say in government decisions, 3 fighting rising prices, 4 protecting freedom of speech. "
                "Reply with two numbers separated by a comma only."
            )

        payload = {
            "system_instruction": {"parts": [{"text": instructions}]},
            "contents": [{"parts": [{"text": user_input}]}],
            "generationConfig": generation_config,
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.base_url + "/models/{}:generateContent".format(model_name),
                    params={"key": self.api_key},
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.exceptions.RequestException as exc:
                delay = min(30.0, 2 ** attempt)
                last_error = RuntimeError("Gemini API transport error: {}".format(exc))
                time.sleep(delay)
                continue
            if response.status_code < 400:
                return response.json()
            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(30.0, 2 ** attempt)
                last_error = RuntimeError("Gemini API {}: {}".format(response.status_code, response.text[:400]))
                time.sleep(delay)
                continue
            response.raise_for_status()
        raise last_error or RuntimeError("Gemini API request failed")

    @staticmethod
    def extract_text(response_payload):
        for candidate in response_payload.get("candidates", []):
            content = candidate.get("content", {})
            parts = []
            for part in content.get("parts", []):
                if part.get("thought") is True:
                    continue
                if "text" in part and part["text"]:
                    parts.append(str(part["text"]))
            text = "\n".join(parts).strip()
            if text:
                return text
        raise ValueError("No text content returned by Gemini")


def run_part1_models(
    data_dir,
    output_dir,
    models=None,
    recent=False,
    env_file=None,
    base_url="https://generativelanguage.googleapis.com/v1beta",
    temperature=0.0,
    delay=0.0,
    overwrite=False,
):
    if env_file:
        load_env_file(env_file)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = GeminiGenerateContentClient(api_key=api_key, base_url=base_url)
    model_names = resolve_models(models=models, recent=recent, recent_models=RECENT_GEMINI_MODELS)
    return run_part1_with_client(
        data_dir=data_dir,
        output_dir=output_dir,
        client=client,
        models=model_names,
        label_overrides=GEMINI_MODEL_LABELS,
        env_file=None,
        temperature=temperature,
        delay=delay,
        overwrite=overwrite,
    )
