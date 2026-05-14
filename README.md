# evolvers

[![PyPI](https://img.shields.io/pypi/v/evolvers.svg)](https://pypi.org/project/evolvers/)
[![CI](https://github.com/tiramisu-sh/evolvers/actions/workflows/ci.yml/badge.svg)](https://github.com/tiramisu-sh/evolvers/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/evolvers.svg)](https://pypi.org/project/evolvers/)
[![Downloads](https://img.shields.io/pypi/dm/evolvers.svg)](https://pypi.org/project/evolvers/)

Evolvable AI programs: define a Python function + criteria of success, the bound LLM iteratively rewrites the function body to maximize the criteria score.

> **Status: early PoC.** APIs will change.

## Idea

A program is a Python function with an injected `llm`. Criteria are either natural-language LLM judges or plain code, each scoring outputs in `[-1, 1]`. An LLM-driven optimizer proposes mutations to the function body; mutations that improve the weighted score are accepted, the rest are reverted. Artifacts are saved as a directory (`manifest.json` + `program.py` + `criteria/`) and loadable by URI.

## Quick look

```python
import evolvers as ev

def tldr(input_text: str, llm) -> str:
    return input_text[:130] + "..."  # naive baseline; the optimizer rewrites this

llm = ev.LLM(model="claude-opus-4-7")  # or any OpenAI-compatible endpoint

evo = ev.Evolvable(
    tldr,
    criteria=[
        ev.judge("Does it directly summarize the main points as a TLDR?"),
        ev.code(lambda output_text:
            max(-1.0, 1 - 2 * max(0, (len(output_text) - 140) / 140))),
    ],
    llm=llm,
)

evo.train(dataset, budget=10)
evo.save("you/tldr-v1:claude-opus-4-7")

reloaded = ev.Evolvable.load("you/tldr-v1:claude-opus-4-7")
print(reloaded("very long text"))
```

## Install (from source)

```bash
git clone https://github.com/tiramisu-sh/evolvers
cd evolvers
uv sync
```

## What works today

- `ev.LLM` against Anthropic and OpenAI-compatible endpoints (vLLM, Ollama, OpenAI, Azure)
- Structured output via pydantic schemas (`schema=`)
- `ev.judge(question)` + `ev.code(callable)` criteria; lambdas captured as `def` for round-tripping
- `Evolvable.train(budget=N)` — propose-test-accept-or-revert loop driven by the bound LLM
- `Evolvable.save("owner/name:variant")` / `Evolvable.load(...)` to/from `~/.cache/evolvers/`
- `Evolvable.clone().set_llm(other)` for variants

Validated end-to-end against a local Qwen3.5-27B reasoning model: naive truncation baseline → optimized LLM-using TLDR in one mutation attempt.

## License

[Apache-2.0](LICENSE).
