"""Evolvable: a function + criteria + LLM, with train/evaluate/save/load.

Async-primary: `train`, `evaluate`, `__call__` are coroutines. Sync convenience
wrappers (`train_sync`, `evaluate_sync`, `call_sync`) spin an event loop per call.

Concurrency is owned by the LLM (via its `max_concurrency`); Evolvable just
fires every (item, rubric) pair as a coroutine and `asyncio.gather`s them.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
import re
import textwrap
import time
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from ._logging import log
from .criterion import Criterion, evaluate_criterion
from .llm import LLM


class Evolvable:
    """A function whose body the optimizer rewrites to maximize criteria scores.

    The function may take an `llm` parameter; if present, the bound LLM is auto-injected
    on each call. Persistence is to a directory under EVOLVERS_CACHE (default ~/.cache/evolvers/).

    The user-supplied function may be either sync or async. Async is preferred when the
    function calls the LLM (use `await llm(prompt)` directly). Sync functions calling the
    LLM should use the sync wrappers (`llm.call_sync(...)`, `llm.batch_sync(...)`).
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        criteria: list[Criterion],
        llm: LLM,
        *,
        _source: str | None = None,
        _signature: inspect.Signature | None = None,
    ):
        self.llm = llm
        self.criteria = list(criteria)
        self._signature = _signature or inspect.signature(fn)
        self._source = _source if _source is not None else _get_source(fn)
        self._compiled = fn
        self._best_source = self._source
        self._best_score: float | None = None
        self.history: list[dict[str, Any]] = []

    @property
    def source(self) -> str:
        return self._best_source

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Async call. Awaits the compiled function if async; runs in a thread if sync.

        Either path returns the function's result. If the function declares an `llm` param,
        the bound LLM is auto-injected.
        """
        if "llm" in self._signature.parameters and "llm" not in kwargs:
            kwargs["llm"] = self.llm
        if asyncio.iscoroutinefunction(self._compiled):
            return await self._compiled(*args, **kwargs)
        return await asyncio.to_thread(self._compiled, *args, **kwargs)

    def call_sync(self, *args: Any, **kwargs: Any) -> Any:
        """Sync wrapper around __call__. Spins up an event loop via asyncio.run."""
        return asyncio.run(self(*args, **kwargs))

    def set_llm(self, llm: LLM) -> Evolvable:
        self.llm = llm
        return self

    def clone(self) -> Evolvable:
        new = Evolvable.__new__(Evolvable)
        new.llm = self.llm
        new.criteria = copy.deepcopy(self.criteria)
        new._signature = self._signature
        new._source = self._source
        new._best_source = self._best_source
        new._best_score = self._best_score
        new._compiled = _compile_fn(self._best_source)
        new.history = []
        return new

    async def evaluate(
        self,
        dataset: Iterable[Any],
        *,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Evaluate on dataset (async). Concurrency is gated by self.llm.max_concurrency."""
        return await self._run_eval(list(dataset), label="eval", show_progress=show_progress)

    def evaluate_sync(self, dataset: Iterable[Any], **kwargs: Any) -> dict[str, Any]:
        return asyncio.run(self.evaluate(dataset, **kwargs))

    async def train(
        self,
        dataset: Iterable[Any],
        *,
        num_train_epochs: int = 20,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Propose-test-accept-or-revert loop (async). Concurrency lives on self.llm.

        `num_train_epochs` (TRL convention): how many propose + full-dataset-eval cycles
        to run. Each epoch produces one new candidate function body, evaluates it across
        the full dataset, and accepts if the aggregate score improves over the best so far.
        """
        data = list(dataset)
        log.info(
            "train_start",
            num_train_epochs=num_train_epochs,
            dataset_size=len(data),
            criteria=[c.name for c in self.criteria],
        )

        log.info("baseline_eval_start")
        baseline = await self._run_eval(data, label="baseline", show_progress=show_progress)
        self._best_score = baseline["aggregate"]
        self._best_source = self._source
        self.history.append(
            {
                "epoch": 0,
                "source": self._source,
                "score": baseline["aggregate"],
                "per_criterion": baseline["per_criterion"],
                "result": baseline,
                "accepted": True,
                "kind": "baseline",
            }
        )

        iterator: Iterable[int] = range(1, num_train_epochs + 1)
        if show_progress:
            iterator = tqdm(iterator, desc="evolve", total=num_train_epochs)

        for epoch in iterator:
            entry: dict[str, Any] = {"epoch": epoch, "accepted": False}
            log.info("epoch_start", epoch=epoch, total=num_train_epochs)
            t_propose = time.perf_counter()
            try:
                new_source = await self._propose_mutation()
            except Exception as e:
                entry["error"] = f"propose failed: {type(e).__name__}: {e}"
                log.warning("propose_failed", epoch=epoch, error=f"{type(e).__name__}: {e}")
                self.history.append(entry)
                continue
            propose_elapsed = time.perf_counter() - t_propose
            log.info(
                "mutation_proposed",
                epoch=epoch,
                elapsed_s=round(propose_elapsed, 1),
                source_chars=len(new_source),
            )

            try:
                new_fn = _compile_fn(new_source)
            except Exception as e:
                entry["source"] = new_source
                entry["error"] = f"compile failed: {type(e).__name__}: {e}"
                log.warning("compile_failed", epoch=epoch, error=f"{type(e).__name__}: {e}")
                self.history.append(entry)
                continue

            prev_compiled = self._compiled
            self._compiled = new_fn
            log.info("candidate_eval_start", epoch=epoch)
            try:
                result = await self._run_eval(data, label=f"epoch {epoch}", show_progress=False)
                entry["score"] = result["aggregate"]
                entry["per_criterion"] = result["per_criterion"]
                entry["source"] = new_source
                entry["result"] = result
                best = self._best_score if self._best_score is not None else float("-inf")
                if result["aggregate"] > best:
                    self._best_score = result["aggregate"]
                    self._best_source = new_source
                    entry["accepted"] = True
                    log.info("epoch_accepted", epoch=epoch, aggregate=result["aggregate"])
                else:
                    self._compiled = prev_compiled
                    log.info(
                        "epoch_reverted",
                        epoch=epoch,
                        aggregate=result["aggregate"],
                        best=self._best_score,
                    )
            except Exception as e:
                self._compiled = prev_compiled
                tb = traceback.format_exc(limit=3)
                entry["source"] = new_source
                entry["error"] = f"eval failed: {type(e).__name__}: {e}"
                entry["traceback"] = tb
                log.warning(
                    "epoch_eval_crashed",
                    epoch=epoch,
                    error=f"{type(e).__name__}: {e}",
                    traceback=tb,
                )

            self.history.append(entry)

        self._compiled = _compile_fn(self._best_source)
        self._source = self._best_source
        log.info("train_done", best_score=self._best_score)
        return {
            "best_score": self._best_score,
            "best_source": self._best_source,
            "history": self.history,
        }

    def train_sync(self, dataset: Iterable[Any], **kwargs: Any) -> dict[str, Any]:
        return asyncio.run(self.train(dataset, **kwargs))

    def save(self, uri: str) -> Path:
        path = _cache_dir(uri)
        path.mkdir(parents=True, exist_ok=True)

        (path / "program.py").write_text(self._best_source)

        crit_dir = path / "criteria"
        crit_dir.mkdir(exist_ok=True)
        criteria_meta: list[dict[str, Any]] = []
        for c in self.criteria:
            entry = {"name": c.name, "kind": c.kind, "weight": c.weight}
            if c.kind == "judge":
                (crit_dir / f"{c.name}.judge.txt").write_text(c.question or "")
            else:
                src = c.source_code or _fallback_code_source(c)
                (crit_dir / f"{c.name}.code.py").write_text(src)
                entry["source_available"] = c.source_code is not None
            criteria_meta.append(entry)

        traces_dir = path / "traces"
        traces_dir.mkdir(exist_ok=True)
        if self.history:
            with (traces_dir / "train.jsonl").open("w") as f:
                for h in self.history:
                    f.write(json.dumps(_jsonable(h)) + "\n")

        manifest = {
            "uri": uri,
            "signature": str(self._signature),
            "function_name": _extract_def_name(self._best_source),
            "llm": {"model": self.llm.model, "provider": self.llm.provider, "base_url": self.llm.base_url},
            "variant": uri.split(":", 1)[1] if ":" in uri else None,
            "criteria": criteria_meta,
            "best_score": self._best_score,
            "saved_at": time.time(),
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info("saved", uri=uri, path=str(path))
        return path

    @classmethod
    def load(cls, uri: str, *, llm: LLM | None = None) -> Evolvable:
        path = _cache_dir(uri)
        if not path.exists():
            raise FileNotFoundError(f"No artifact at {path}")
        manifest = json.loads((path / "manifest.json").read_text())

        source = (path / "program.py").read_text()
        fn = _compile_fn(source)

        criteria: list[Criterion] = []
        for cmeta in manifest["criteria"]:
            name, kind, weight = cmeta["name"], cmeta["kind"], cmeta["weight"]
            if kind == "judge":
                q = (path / "criteria" / f"{name}.judge.txt").read_text()
                criteria.append(Criterion(name=name, kind="judge", weight=weight, question=q))
            else:
                src = (path / "criteria" / f"{name}.code.py").read_text()
                code_fn = _compile_fn(src)
                criteria.append(Criterion(name=name, kind="code", weight=weight, fn=code_fn, source_code=src))

        if llm is None:
            llm_meta = manifest.get("llm", {})
            llm = LLM(model=llm_meta.get("model", ""), base_url=llm_meta.get("base_url"))

        instance = cls.__new__(cls)
        instance.llm = llm
        instance.criteria = criteria
        instance._signature = inspect.signature(fn)
        instance._source = source
        instance._best_source = source
        instance._best_score = manifest.get("best_score")
        instance._compiled = fn
        instance.history = []
        log.info("loaded", uri=uri, best_score=manifest.get("best_score"))
        return instance

    async def _run_one_trial(self, row: Any, idx: int, total: int) -> dict[str, Any]:
        program_input, call_args, call_kwargs = _row_to_call(row, self._signature)
        log.debug(
            "trial_start",
            trial=idx + 1,
            total=total,
            input=_truncate(repr(program_input), 80),
        )
        t0 = time.perf_counter()
        try:
            output = await self(*call_args, **call_kwargs)
            err = None
        except Exception as e:
            output = None
            err = f"{type(e).__name__}: {e}"
        program_latency_ms = (time.perf_counter() - t0) * 1000

        per_criterion: dict[str, dict[str, Any]] = {}
        if output is None:
            for c in self.criteria:
                per_criterion[c.name] = {"score": -1.0, "reasoning": f"program failed: {err}"}
        else:
            # All criteria fire as coroutines; LLM's semaphore gates transport-layer concurrency.
            results = await asyncio.gather(
                *(evaluate_criterion(c, program_input, output, self.llm) for c in self.criteria)
            )
            for c, (score, reasoning) in zip(self.criteria, results, strict=True):
                per_criterion[c.name] = {"score": score, "reasoning": reasoning}

        scores_summary = {k: round(v["score"], 2) for k, v in per_criterion.items()}
        elapsed = time.perf_counter() - t0
        log.debug(
            "trial_done",
            trial=idx + 1,
            total=total,
            elapsed_s=round(elapsed, 1),
            output=_truncate(repr(output), 80),
            scores=scores_summary,
        )
        return {
            "input": program_input,
            "output": output,
            "error": err,
            "latency_ms": program_latency_ms,
            "per_criterion": per_criterion,
        }

    async def _run_eval(
        self,
        data: list[Any],
        *,
        label: str,
        show_progress: bool,
    ) -> dict[str, Any]:
        n = len(data)
        log.info("eval_start", label=label, rows=n, max_concurrency=self.llm.max_concurrency)
        t0 = time.perf_counter()

        # All trials fire concurrently; LLM's internal semaphore gates the actual transport.
        trials = await asyncio.gather(*(self._run_one_trial(row, idx, n) for idx, row in enumerate(data)))

        per_criterion_mean: dict[str, float] = {}
        total_weight = sum(c.weight for c in self.criteria) or 1.0
        for c in self.criteria:
            scores = [t["per_criterion"][c.name]["score"] for t in trials]
            per_criterion_mean[c.name] = sum(scores) / max(1, len(scores))
        aggregate = sum(per_criterion_mean[c.name] * c.weight for c in self.criteria) / total_weight

        elapsed = time.perf_counter() - t0
        log.info(
            "eval_done",
            label=label,
            elapsed_s=round(elapsed, 1),
            aggregate=aggregate,
            per_criterion=per_criterion_mean,
        )

        return {
            "aggregate": aggregate,
            "per_criterion": per_criterion_mean,
            "trials": trials,
            "label": label,
        }

    async def _propose_mutation(self) -> str:
        recent = [h for h in self.history if "score" in h][-3:]
        if not recent:
            recent = [{"epoch": 0, "source": self._best_source, "score": self._best_score or 0.0}]

        def _desc(c: Criterion) -> str:
            if c.kind == "judge":
                # Criterion.__post_init__ already enforces this; re-check here so
                # that any future deserialization path that bypasses the
                # constructor (e.g. __new__ / pickle) fails loudly instead of
                # embedding "None" into the mutation prompt. Raise (not assert)
                # because asserts are stripped under `python -O`.
                if not c.question:
                    raise ValueError(f"judge criterion {c.name!r} has no question")
                return c.question
            return c.source_code or "<code>"

        criteria_desc = "\n".join(
            f"- {c.name} (weight={c.weight:.2f}, kind={c.kind}): " + _desc(c) for c in self.criteria
        )

        history_lines = []
        for h in recent:
            score = h.get("score") if h.get("score") is not None else 0.0
            block = f"Epoch {h['epoch']} (score={score:.3f}):\n```python\n{h.get('source', '')}\n```"
            history_lines.append(block)
        history_block = "\n\n".join(history_lines)

        last_trials_block = ""
        for h in reversed(recent):
            result = h.get("result")
            if not isinstance(result, dict):
                continue
            trials = result.get("trials", [])
            if trials:
                lines = []
                for t in trials[:3]:
                    score_summary = {k: round(v["score"], 2) for k, v in t["per_criterion"].items()}
                    reasoning_summary = {
                        k: _truncate(v.get("reasoning", "") or "", 200) for k, v in t["per_criterion"].items()
                    }
                    lines.append(
                        f"- input: {_truncate(repr(t['input']), 200)}\n"
                        f"  output: {_truncate(repr(t['output']), 400)}\n"
                        f"  scores: {score_summary}\n"
                        f"  judge_reasoning: {reasoning_summary}"
                    )
                last_trials_block = "Sample trials from last evaluation:\n" + "\n".join(lines)
                break

        best_score = self._best_score if self._best_score is not None else 0.0
        prompt = textwrap.dedent(f"""\
            You are optimizing a Python function. Goal: maximize the weighted-mean criterion score (range [-1, 1]).

            Current best implementation (score={best_score:.3f}):
            ```python
            {self._best_source}
            ```

            Criteria:
            {criteria_desc}

            Recent epochs:
            {history_block}

            {last_trials_block}

            You may use the injected `llm` callable inside the function. It is async-primary.

            If your function is `async def`, await llm calls:
              await llm(prompt, *, schema=..., system=...)         # → str | BaseModel
              await llm.batch(prompts, **kwargs)                    # → list

            If your function is sync `def`, use the sync wrappers (slower; spins an event loop per call):
              llm.call_sync(prompt, *, schema=..., system=...)
              llm.batch_sync(prompts, **kwargs)

            Prefer `async def` when the function makes any LLM calls.

            IMPORTANT — token budgets:
            - The injected `llm` is a 2026-era reasoning model. It uses substantial tokens for internal
              thinking BEFORE producing visible content. Defaults are already tuned for this.
            - DO NOT pass `max_tokens` to `llm()`. The default is set to a high value (32k+).
              Passing a small `max_tokens` (e.g. 100, 500, 1024) will cause the reasoning preamble
              to consume the entire budget, leaving content empty and the function returning ''.

            Rules:
            - Preserve the function signature: {self._signature}.
            - Reply with ONLY the function source as a single Python code block (```python ... ```).
            - Do not include explanations outside the code block.
            """)

        response = await self.llm(
            prompt,
            system=(
                "You are an expert Python programmer iteratively refining a function under criteria. "
                "You target 2026-era reasoning LLMs and never artificially cap token budgets."
            ),
        )
        if not isinstance(response, str):
            response = str(response)
        return _extract_python(response)

    def __repr__(self) -> str:
        name = _extract_def_name(self._best_source) or "<evolvable>"
        return f"Evolvable({name}, criteria={[c.name for c in self.criteria]}, llm={self.llm.model})"


def _compile_fn(source: str) -> Callable[..., Any]:
    ns: dict[str, Any] = {}
    exec(compile(source, "<evolvable>", "exec"), ns)
    fns = [v for v in ns.values() if inspect.isfunction(v)]
    if not fns:
        raise ValueError("no function found in source")
    return fns[-1]


def _get_source(fn: Callable) -> str:
    try:
        return textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as e:
        raise ValueError(f"cannot get source for {fn}: {e}") from e


def _extract_python(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_def_name(source: str) -> str | None:
    m = re.search(r"^(?:async\s+)?def\s+(\w+)\s*\(", source, re.MULTILINE)
    return m.group(1) if m else None


def _row_to_call(row: Any, sig: inspect.Signature) -> tuple[Any, tuple, dict]:
    params = [p for p in sig.parameters.values() if p.name != "llm"]
    if isinstance(row, dict):
        kwargs = {k: v for k, v in row.items() if k in sig.parameters}
        first_param = params[0].name if params else None
        program_input = kwargs.get(first_param) if first_param else row
        return program_input, (), kwargs
    return row, (row,), {}


def _cache_dir(uri: str) -> Path:
    root = Path(os.environ.get("EVOLVERS_CACHE", "~/.cache/evolvers")).expanduser()
    return root / uri


def _fallback_code_source(c: Criterion) -> str:
    return f"# source not captured for {c.name!r}\ndef {c.name}(output):\n    return 0.0\n"


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        return repr(obj)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."
