import os
import time

import requests

from .constants import REQUESTED_TOGETHER_MODELS, TOGETHER_MODEL_LABELS
from .provider_runner import resolve_models, run_part1_with_client
from .utils import load_env_file


class TogetherChatCompletionsClient(object):
    provider_name = "Together"

    def __init__(self, api_key, base_url="https://api.together.xyz/v1", timeout=90, max_retries=5):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    @staticmethod
    def _max_tokens_for_model(model):
        if model == "MiniMaxAI/MiniMax-M2.7":
            return 256
        return 64

    @staticmethod
    def _max_retry_tokens_for_model(model):
        if model == "MiniMaxAI/MiniMax-M2.7":
            return 4096
        return 128

    @staticmethod
    def _has_text_content(response_payload):
        choices = response_payload.get("choices") or []
        if not choices:
            return False
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return True
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and str(item.get("text") or "").strip():
                    return True
        return False

    @staticmethod
    def _is_reasoning_only(response_payload):
        choices = response_payload.get("choices") or []
        if not choices:
            return False
        choice = choices[0] or {}
        message = choice.get("message") or {}
        return (
            not TogetherChatCompletionsClient._has_text_content(response_payload)
            and bool(str(message.get("reasoning") or "").strip())
            and str(choice.get("finish_reason") or "") == "length"
        )

    def create_response(self, model, instructions, user_input, temperature=0.0, scale=None):
        if model == "MiniMaxAI/MiniMax-M2.7" and scale == "f120":
            user_input = (
                "Question: On a scale from 1 to 10, where 10 means always justifiable and 1 means never justifiable, "
                "how justifiable is abortion? Reply with one number only."
            )
        if model == "MiniMaxAI/MiniMax-M2.7" and scale == "f118":
            user_input = (
                "Question: On a scale from 1 to 10, where 10 means always justifiable and 1 means never justifiable, "
                "how justifiable are same-sex relationships? Reply with one number only."
            )
        if model == "MiniMaxAI/MiniMax-M2.7" and scale == "y003":
            user_input = (
                "Question: Choose up to five especially important qualities for children from this list: "
                "Good manners; Independence; Hard work; Feeling of responsibility; Imagination; "
                "Tolerance and respect for other people; Thrift, saving money and things; "
                "Determination, perseverance; Religious faith; Not being selfish (unselfishness); Obedience. "
                "Reply with up to five qualities only, separated by commas."
            )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_input},
            ],
            "temperature": temperature,
            "max_tokens": self._max_tokens_for_model(model),
            "reasoning": {"enabled": False},
        }
        headers = {
            "Authorization": "Bearer {}".format(self.api_key),
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(self.max_retries):
            response = self.session.post(
                self.base_url + "/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code < 400:
                response_payload = response.json()
                if self._is_reasoning_only(response_payload):
                    next_max_tokens = min(self._max_retry_tokens_for_model(model), payload["max_tokens"] * 2)
                    if next_max_tokens > payload["max_tokens"]:
                        payload["max_tokens"] = next_max_tokens
                        time.sleep(min(4.0, 0.5 * (attempt + 1)))
                        continue
                return response_payload
            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(30.0, 2 ** attempt)
                last_error = RuntimeError("Together API {}: {}".format(response.status_code, response.text[:400]))
                time.sleep(delay)
                continue
            response.raise_for_status()
        raise last_error or RuntimeError("Together API request failed")

    @staticmethod
    def extract_text(response_payload):
        choices = response_payload.get("choices") or []
        if not choices:
            raise ValueError("No choices returned by Together")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
            text = "\n".join(parts).strip()
            if text:
                return text
        raise ValueError("No text content returned by Together")


def run_part1_models(
    data_dir,
    output_dir,
    models=None,
    recent=False,
    env_file=None,
    base_url="https://api.together.xyz/v1",
    temperature=0.0,
    delay=0.0,
    overwrite=False,
):
    if env_file:
        load_env_file(env_file)

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY is not set")

    client = TogetherChatCompletionsClient(api_key=api_key, base_url=base_url)
    model_names = resolve_models(models=models, recent=recent, recent_models=REQUESTED_TOGETHER_MODELS)
    return run_part1_with_client(
        data_dir=data_dir,
        output_dir=output_dir,
        client=client,
        models=model_names,
        label_overrides=TOGETHER_MODEL_LABELS,
        env_file=None,
        temperature=temperature,
        delay=delay,
        overwrite=overwrite,
    )
