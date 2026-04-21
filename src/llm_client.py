"""Unified LLM client — OpenAI and Anthropic behind one `complete()` call.

Swap providers via config.LLM_PROVIDER without changing caller code.
"""

from __future__ import annotations

import json
from typing import Any

from config import (
    ANTHROPIC_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
)


def complete(
    system: str,
    user: str,
    json_mode: bool = False,
    model: str | None = None,
    temperature: float | None = None,
) -> Any:
    """Send a prompt and return either a raw string or parsed JSON."""
    model = model or LLM_MODEL
    temperature = LLM_TEMPERATURE if temperature is None else temperature

    if LLM_PROVIDER == "openai":
        return _complete_openai(system, user, json_mode, model, temperature)
    if LLM_PROVIDER == "anthropic":
        return _complete_anthropic(system, user, json_mode, model, temperature)
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def _complete_openai(
    system: str, user: str, json_mode: bool, model: str, temperature: float
) -> Any:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": LLM_MAX_TOKENS,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""
    return json.loads(content) if json_mode else content


def _complete_anthropic(
    system: str, user: str, json_mode: bool, model: str, temperature: float
) -> Any:
    from anthropic import Anthropic

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=model,
        system=system,
        max_tokens=LLM_MAX_TOKENS,
        temperature=temperature,
        messages=[{"role": "user", "content": user}],
    )
    content = msg.content[0].text if msg.content else ""
    return json.loads(content) if json_mode else content
