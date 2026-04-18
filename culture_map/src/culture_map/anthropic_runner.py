import os
import time

import requests

from .constants import ANTHROPIC_MODEL_LABELS, RECENT_ANTHROPIC_MODELS
from .provider_runner import resolve_models, run_part1_with_client
from .utils import load_env_file


class AnthropicMessagesClient(object):
    provider_name = "Anthropic"

    def __init__(self, api_key, base_url="https://api.anthropic.com/v1", timeout=90, max_retries=5):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def create_response(self, model, instructions, user_input, temperature=0.0, scale=None):
        del scale
        payload = {
            "model": model,
            "system": instructions,
            "messages": [{"role": "user", "content": user_input}],
            "max_tokens": 64,
        }
        if temperature and temperature > 0.0:
            payload["temperature"] = temperature
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        last_error = None
        for attempt in range(self.max_retries):
            response = self.session.post(
                self.base_url + "/messages",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code < 400:
                return response.json()
            if response.status_code in (429, 500, 502, 503, 504, 529):
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(30.0, 2 ** attempt)
                last_error = RuntimeError("Anthropic API {}: {}".format(response.status_code, response.text[:400]))
                time.sleep(delay)
                continue
            response.raise_for_status()
        raise last_error or RuntimeError("Anthropic API request failed")

    @staticmethod
    def extract_text(response_payload):
        parts = []
        for block in response_payload.get("content", []):
            if block.get("type") == "text" and block.get("text"):
                parts.append(str(block["text"]))
        text = "\n".join(parts).strip()
        if not text:
            raise ValueError("No text content returned by Anthropic")
        return text


def run_part1_models(
    data_dir,
    output_dir,
    models=None,
    recent=False,
    env_file=None,
    base_url="https://api.anthropic.com/v1",
    temperature=0.0,
    delay=0.0,
    overwrite=False,
):
    if env_file:
        load_env_file(env_file)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    client = AnthropicMessagesClient(api_key=api_key, base_url=base_url)
    model_names = resolve_models(models=models, recent=recent, recent_models=RECENT_ANTHROPIC_MODELS)
    return run_part1_with_client(
        data_dir=data_dir,
        output_dir=output_dir,
        client=client,
        models=model_names,
        label_overrides=ANTHROPIC_MODEL_LABELS,
        env_file=None,
        temperature=temperature,
        delay=delay,
        overwrite=overwrite,
    )
