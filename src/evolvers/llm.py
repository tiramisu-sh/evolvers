"""LLM wrapper: unified call/batch with optional structured output."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from pydantic import BaseModel

Provider = Literal["anthropic", "openai"]


class LLM:
    """Thin wrapper around Anthropic + OpenAI-compatible clients.

    Detection: explicit `base_url` → OpenAI-compatible; model starts with "claude" → Anthropic;
    otherwise OpenAI. Pass `api_key` to override the SDK's env-var default.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: Provider | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.provider = provider or self._detect_provider(model, base_url)
        self._api_key = api_key
        self._client: Any = None  # lazy-init

    @staticmethod
    def _detect_provider(model: str, base_url: str | None) -> Provider:
        if base_url:
            return "openai"
        if model.lower().startswith("claude"):
            return "anthropic"
        return "openai"

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if self.provider == "anthropic":
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self._api_key) if self._api_key else Anthropic()
        else:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url=self.base_url,
            )

    def __call__(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        tools: list[Callable] | None = None,
        system: str | None = None,
        max_tokens: int = 32768,
    ) -> Any:
        """Single call. If schema is given, returns a parsed pydantic instance; else str."""
        if tools is not None:
            raise NotImplementedError("tool calling not in v0 PoC")
        self._ensure_client()
        if self.provider == "anthropic":
            return self._call_anthropic(prompt, schema, system, max_tokens)
        return self._call_openai(prompt, schema, system, max_tokens)

    def batch(self, prompts: list[str], *, max_workers: int = 8, **kwargs: Any) -> list[Any]:
        """Run prompts in parallel via threadpool."""
        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(prompts))) as ex:
            return list(ex.map(lambda p: self(p, **kwargs), prompts))

    def _call_anthropic(
        self,
        prompt: str,
        schema: type[BaseModel] | None,
        system: str | None,
        max_tokens: int,
    ) -> Any:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        if schema is not None:
            kwargs["tools"] = [
                {
                    "name": "respond",
                    "description": f"Provide a response matching the {schema.__name__} schema.",
                    "input_schema": schema.model_json_schema(),
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "respond"}
            resp = self._client.messages.create(**kwargs)
            for block in resp.content:
                if block.type == "tool_use" and block.name == "respond":
                    return schema(**block.input)
            raise RuntimeError(f"Anthropic returned no tool_use block; got {resp.content}")

        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text

    def _call_openai(
        self,
        prompt: str,
        schema: type[BaseModel] | None,
        system: str | None,
        max_tokens: int,
    ) -> Any:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if schema is not None:
            try:
                resp = self._client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=schema,
                    max_tokens=max_tokens,
                )
                parsed = resp.choices[0].message.parsed
                if parsed is None:
                    raise RuntimeError("openai parse returned None")
                return parsed
            except Exception:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        *messages[:-1],
                        {
                            "role": "user",
                            "content": (
                                f"{prompt}\n\nReply with a JSON object matching this schema:\n"
                                f"{json.dumps(schema.model_json_schema(), indent=2)}"
                            ),
                        },
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                )
                return schema.model_validate_json(resp.choices[0].message.content)

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def __repr__(self) -> str:
        return f"LLM(model={self.model!r}, provider={self.provider!r})"
