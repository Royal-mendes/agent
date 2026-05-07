from __future__ import annotations

import os
import json
from typing import Any, Optional

from agent.schemas import AgentConfig


DEFAULT_OPENAI_COMPATIBLE_MODEL = "gpt-4o-mini"
DEFAULT_LOCAL_VLM_MODEL = "qwen2.5-vl-7b-instruct"
DEFAULT_LOCAL_VLM_BASE_URL = "http://127.0.0.1:18000/v1"


class VLMProviderError(RuntimeError):
    pass


class OpenAICompatibleVLMProvider:
    """OpenAI-compatible chat provider.

    This works with the official OpenAI API and with relay endpoints that expose
    the OpenAI Chat Completions protocol. Secrets are read from environment
    variables and are never stored by this class.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
        force_json_response_format: bool = False,
        max_tokens: Optional[int] = None,
        max_input_chars: Optional[int] = None,
        extra_body: Optional[dict] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.model = model or DEFAULT_OPENAI_COMPATIBLE_MODEL
        self.temperature = temperature
        self.force_json_response_format = force_json_response_format
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.max_input_chars = max_input_chars
        self.extra_body = extra_body or {}
        if client is not None:
            self.client = client
            return

        if not api_key:
            raise VLMProviderError("OpenAI-compatible VLM provider requires an API key env var.")
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise VLMProviderError("openai package is not installed in this environment.") from exc

        kwargs = {"api_key": api_key, "timeout": timeout_seconds, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data_url: Optional[str] = None,
        image_data_urls: Optional[list[str]] = None,
    ) -> str:
        user_prompt = self._fit_user_prompt(user_prompt)
        user_content: Any = user_prompt
        image_urls = list(image_data_urls or [])
        if image_data_url:
            image_urls.insert(0, image_data_url)
        if image_urls:
            user_content = [{"type": "text", "text": user_prompt}]
            user_content.extend(
                {"type": "image_url", "image_url": {"url": url}} for url in image_urls
            )
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "timeout": self.timeout_seconds,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if self.force_json_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _fit_user_prompt(self, user_prompt: str) -> str:
        """Bound local prompt size while preserving task constraints and output schema."""
        limit = self.max_input_chars
        if not limit or len(user_prompt) <= limit:
            return user_prompt
        marker = "\n...[middle of prompt truncated to fit local VLM context]...\n"
        head = max(1000, limit // 3)
        tail = max(1000, limit - head - len(marker))
        if head + tail + len(marker) >= len(user_prompt):
            return user_prompt
        return user_prompt[:head] + marker + user_prompt[-tail:]


def build_vlm_provider(cfg: AgentConfig) -> Optional[OpenAICompatibleVLMProvider]:
    provider = (cfg.vlm_provider or "mock").lower()
    if provider == "mock":
        return None
    if provider not in {"openai", "local"}:
        raise VLMProviderError(f"Unsupported VLM provider: {cfg.vlm_provider}")

    if provider == "openai":
        api_key = cfg.vlm_api_key or os.environ.get(cfg.vlm_api_key_env)
        base_url = cfg.vlm_base_url or os.environ.get(cfg.vlm_base_url_env)
        model = cfg.vlm_model or os.environ.get(cfg.vlm_model_env) or DEFAULT_OPENAI_COMPATIBLE_MODEL
    else:
        api_key = cfg.vlm_api_key or os.environ.get("LOCAL_VLM_API_KEY") or os.environ.get(cfg.vlm_api_key_env) or "local"
        base_url = cfg.vlm_base_url or os.environ.get("LOCAL_VLM_BASE_URL") or DEFAULT_LOCAL_VLM_BASE_URL
        model = cfg.vlm_model or os.environ.get("LOCAL_VLM_MODEL") or DEFAULT_LOCAL_VLM_MODEL

    if provider == "local":
        max_tokens = int(os.environ.get("LOCAL_VLM_MAX_TOKENS", "512"))
        max_input_chars = int(os.environ.get("LOCAL_VLM_MAX_INPUT_CHARS", "9000"))
        extra_body = {"repetition_penalty": 1.05}
        extra_body.update(_json_env("LOCAL_VLM_EXTRA_BODY_JSON"))
    else:
        max_tokens = _optional_int_env("VLM_MAX_TOKENS")
        max_input_chars = _optional_int_env("VLM_MAX_INPUT_CHARS")
        extra_body = {}
    extra_body.update(_json_env("VLM_EXTRA_BODY_JSON"))

    timeout_seconds = _effective_timeout_seconds(cfg.vlm_timeout_seconds)
    return OpenAICompatibleVLMProvider(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        temperature=cfg.vlm_temperature,
        force_json_response_format=cfg.vlm_force_json_response_format,
        max_tokens=max_tokens,
        max_input_chars=max_input_chars,
        extra_body=extra_body or None,
    )


def _optional_int_env(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _effective_timeout_seconds(configured: float) -> float:
    try:
        timeout = float(configured)
    except (TypeError, ValueError):
        timeout = 30.0
    try:
        cap = float(os.environ.get("VLM_HARD_TIMEOUT_SECONDS", "45"))
    except ValueError:
        cap = 45.0
    return max(1.0, min(timeout, cap))


def _json_env(name: str) -> dict:
    value = os.environ.get(name)
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise VLMProviderError(f"{name} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise VLMProviderError(f"{name} must decode to a JSON object.")
    return parsed
