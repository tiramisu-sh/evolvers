"""Criterion: judge (LLM-as-judge) and code (Python function) rubrics."""
from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

Kind = Literal["judge", "code"]


class _JudgeResponse(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    reasoning: str


@dataclass
class Criterion:
    """One rubric. Either a natural-language judge or a Python function."""
    name: str
    kind: Kind
    weight: float = 1.0
    question: str | None = None       # for kind="judge"
    fn: Callable[..., float] | None = None  # for kind="code"
    source_code: str | None = None    # captured for kind="code", to enable save/load

    def __post_init__(self) -> None:
        if self.kind == "judge" and not self.question:
            raise ValueError(f"judge criterion {self.name!r} needs a `question`")
        if self.kind == "code" and self.fn is None:
            raise ValueError(f"code criterion {self.name!r} needs a `fn`")


def judge(question: str, *, name: str | None = None, weight: float = 1.0) -> Criterion:
    """Define an LLM-as-judge criterion from a natural-language question.

    The judge scores in [-1, 1]. -1 = fails entirely; +1 = perfectly satisfies.
    """
    return Criterion(
        name=name or _slugify(question, max_len=40),
        kind="judge",
        weight=weight,
        question=question,
    )


def code(
    fn: Callable[..., float] | Callable[..., int] | Callable[..., bool],
    *,
    name: str | None = None,
    weight: float = 1.0,
) -> Criterion:
    """Define a code criterion from a callable.

    The callable's signature is introspected:
    - 1 arg → called with `output` only
    - 2 args → called with `(input, output)`
    Returns a float in [-1, 1]; ints/bools are auto-cast.
    """
    resolved_name = name or _resolve_callable_name(fn)
    src = _capture_source_as_def(fn, resolved_name)
    return Criterion(
        name=resolved_name,
        kind="code",
        weight=weight,
        fn=fn,
        source_code=src,
    )


def evaluate_criterion(
    c: Criterion,
    program_input: Any,
    program_output: Any,
    llm: Any,
) -> tuple[float, str]:
    """Run one criterion. Returns (score, reasoning). Score is clamped to [-1, 1]."""
    if c.kind == "code":
        return _evaluate_code(c, program_input, program_output)
    return _evaluate_judge(c, program_input, program_output, llm)


def _evaluate_code(c: Criterion, program_input: Any, program_output: Any) -> tuple[float, str]:
    assert c.fn is not None
    try:
        params = list(inspect.signature(c.fn).parameters.values())
    except (TypeError, ValueError):
        params = []
    try:
        if len(params) == 1:
            raw = c.fn(program_output)
        elif len(params) >= 2:
            raw = c.fn(program_input, program_output)
        else:
            raw = c.fn()
    except Exception as e:
        return -1.0, f"code criterion raised {type(e).__name__}: {e}"
    value = float(raw)
    return _clamp(value), f"code returned {value:.4f}"


def _evaluate_judge(
    c: Criterion,
    program_input: Any,
    program_output: Any,
    llm: Any,
) -> tuple[float, str]:
    prompt = (
        f"You are a strict but fair judge. Score the OUTPUT against the RUBRIC.\n\n"
        f"RUBRIC: {c.question}\n\n"
        f"INPUT:\n{program_input}\n\n"
        f"OUTPUT:\n{program_output}\n\n"
        f"Reply with score in [-1, 1] (-1 = fails entirely; 0 = neutral; +1 = perfectly satisfies) "
        f"and concise reasoning."
    )
    try:
        resp = llm(prompt, schema=_JudgeResponse)
    except Exception as e:
        return 0.0, f"judge LLM failed ({type(e).__name__}: {e}); neutral score"
    return _clamp(resp.score), resp.reasoning


def _clamp(x: float) -> float:
    if x != x:
        return 0.0
    return max(-1.0, min(1.0, x))


def _slugify(text: str, *, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len] or "criterion"


def _resolve_callable_name(fn: Callable) -> str:
    name = getattr(fn, "__name__", "code_criterion")
    if name == "<lambda>":
        return "code_criterion"
    return name


def _capture_source_as_def(fn: Callable, name: str) -> str | None:
    """Return source as a `def {name}(args): ...` string, converting lambdas.

    Falls back to None if no source is available.
    """
    try:
        raw = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    raw = raw.strip()

    if raw.lstrip().startswith("def "):
        return raw

    try:
        tree = ast.parse(raw)
    except SyntaxError:
        # source line may include surrounding context — try to find the lambda
        for chunk in (raw, raw.split("=", 1)[-1].strip() if "=" in raw else raw):
            try:
                tree = ast.parse(chunk, mode="eval")
                break
            except SyntaxError:
                continue
        else:
            return None

    lambda_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            lambda_node = node
            break
    if lambda_node is None:
        return None
    args_src = ast.unparse(lambda_node.args)
    body_src = ast.unparse(lambda_node.body)
    return f"def {name}({args_src}):\n    return {body_src}\n"
