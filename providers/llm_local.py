"""Local LLM via Ollama's OpenAI-compatible endpoint.

Ollama exposes `http://localhost:11434/v1/chat/completions` which is wire-
compatible with OpenAI's Chat Completions API. We use the official `openai`
Python client — no new SDK to learn, and the same code path works if you
later point at LiteLLM, vLLM, an OpenAI-compatible cloud API, etc.

Config (config.yaml → generator):
    base_url: http://localhost:11434/v1
    api_key:  ollama                # any string; Ollama ignores it
    model:    qwen3:35b-a3b
    system_prompt: ...
    temperature, max_tokens, timeout_s
"""
from __future__ import annotations

import logging

from openai import OpenAI

from core.interfaces import Generator


log = logging.getLogger(__name__)


class LocalGenerator(Generator):
    """Calls an OpenAI-compatible chat completions endpoint.

    Stateless across calls — no conversation memory, by design. The CLI is
    the only place that knows the system prompt and the user prompt.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        model: str = "qwen3:35b-a3b",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout_s: int = 120,
        system_prompt: str = "You are a careful assistant.",
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.system_prompt = system_prompt
        log.info("LocalGenerator → %s @ %s (model=%s)", base_url, "redacted", model)
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)

    def generate(self, prompt: str) -> str:
        """Send the user prompt (with the configured system prompt) and return
        the assistant's text. We hand-craft a two-message conversation so the
        system prompt stays at the system role and the user-supplied prompt
        is the only user content."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        # Defensive: tolerate streaming/empty responses.
        if not resp.choices:
            raise RuntimeError(f"empty response from {self.model}")
        return resp.choices[0].message.content or ""
