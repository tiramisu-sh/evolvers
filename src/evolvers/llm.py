"""LLM wrapper: async-primary unified call/batch with optional structured output."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel

Provider = Literal["anthropic", "openai"]


class LLM:
    """Thin async-primary wrapper around Anthropic + OpenAI-compatible clients.

    Concurrency is gated at the transport level via an internal `asyncio.Semaphore`
    sized by `max_concurrency`. Callers (Evolvable, Criterion) just fire as many
    coroutines as they like; coroutines that can't acquire a slot park cheaply on
    the event loop until one frees.

    Transient failures (429, 5xx, connection errors) are retried by the SDK with
    exponential backoff + jitter, honoring Retry-After; `max_retries` sizes that.

    Detection: explicit `base_url` → OpenAI-compatible; model starts with "claude"
    → Anthropic; otherwise OpenAI. Pass `api_key` to override the SDK's env-var
    default.

    Sync convenience wrappers (`call_sync`, `batch_sync`) are provided for use
    from sync codebases. They spin up an event loop via `asyncio.run` per call.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: Provider | None = None,
        # Conservative default: safe for rate-limited hosted APIs out of the box.
        # Raise it (64-256) for a dedicated endpoint like vLLM with more headroom.
        max_concurrency: int = 16,
        # Forwarded to the SDK client, which already does exponential backoff +
        # jitter and honors Retry-After. The SDK's own default of 2 is too low to
        # ride out a rate-limit window across a long training run.
        max_retries: int = 8,
    ):
        self.model = model
        self.base_url = base_url
        self.provider = provider or self._detect_provider(model, base_url)
        self._api_key = api_key
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self._sem = asyncio.Semaphore(max_concurrency)
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
            from anthropic import AsyncAnthropic

            self._client = (
                AsyncAnthropic(api_key=self._api_key, max_retries=self.max_retries)
                if self._api_key
                else AsyncAnthropic(max_retries=self.max_retries)
            )
        else:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url=self.base_url,
                max_retries=self.max_retries,
            )

    async def __call__(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        tools: list[Callable] | None = None,
        system: str | None = None,
        max_tokens: int = 32768,
    ) -> Any:
        """Single call (async). If schema is given, returns a parsed pydantic instance; else str."""
        if tools is not None:
            raise NotImplementedError("tool calling not in v0 PoC")
        self._ensure_client()
        async with self._sem:
            if self.provider == "anthropic":
                return await self._call_anthropic(prompt, schema, system, max_tokens)
            return await self._call_openai(prompt, schema, system, max_tokens)

    async def batch(self, prompts: list[str], **kwargs: Any) -> list[Any]:
        """Run prompts in parallel. Concurrency is gated by self.max_concurrency,
        not a per-call knob — that lives on the LLM, by design."""
        if not prompts:
            return []
        return await asyncio.gather(*(self(p, **kwargs) for p in prompts))

    # ── sync convenience wrappers ──────────────────────────────────────────────
    # For users in sync codebases. Each call spins up a fresh event loop via
    # asyncio.run. Concurrency gating across multiple sync calls from threads is
    # NOT guaranteed — if you want high concurrency, use the async path.

    def call_sync(self, prompt: str, **kwargs: Any) -> Any:
        return asyncio.run(self(prompt, **kwargs))

    def batch_sync(self, prompts: list[str], **kwargs: Any) -> list[Any]:
        return asyncio.run(self.batch(prompts, **kwargs))

    # ── async transport implementations ────────────────────────────────────────

    async def _call_anthropic(
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
            resp = await self._client.messages.create(**kwargs)
            for block in resp.content:
                if block.type == "tool_use" and block.name == "respond":
                    return schema(**block.input)
            raise RuntimeError(f"Anthropic returned no tool_use block; got {resp.content}")

        resp = await self._client.messages.create(**kwargs)
        return resp.content[0].text

    async def _call_openai(
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
                resp = await self._client.beta.chat.completions.parse(
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
                resp = await self._client.chat.completions.create(
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

        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def __repr__(self) -> str:
        return (
            f"LLM(model={self.model!r}, provider={self.provider!r}, "
            f"max_concurrency={self.max_concurrency}, max_retries={self.max_retries})"
        )
