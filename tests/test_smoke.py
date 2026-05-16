"""Smoke tests that don't require an LLM endpoint."""

from __future__ import annotations

import evolvers as ev


def test_imports():
    assert ev.LLM is not None
    assert ev.Evolvable is not None
    assert ev.Criterion is not None
    assert ev.judge is not None
    assert ev.code is not None


def test_judge_construction():
    c = ev.judge("Is it good?")
    assert c.kind == "judge"
    assert c.question == "Is it good?"
    assert c.weight == 1.0


def test_code_criterion_eval():
    c = ev.code(
        lambda output_text: max(-1.0, 1 - 2 * max(0, (len(output_text) - 140) / 140)),
        name="length",
    )
    assert c.kind == "code"
    assert c.fn is not None
    assert c.fn("short") == 1
    assert c.fn("x" * 300) == -1.0
    assert c.source_code is not None
    assert "def length" in c.source_code


def test_llm_provider_detection():
    assert ev.LLM(model="claude-opus-4-7").provider == "anthropic"
    assert ev.LLM(model="gpt-4").provider == "openai"
    assert ev.LLM(model="deepkek", base_url="http://localhost:8001/v1", api_key="x").provider == "openai"


def test_llm_max_retries():
    # default (8) is stored and forwarded to the SDK client — anthropic branch
    a = ev.LLM(model="claude-opus-4-7", api_key="x")
    assert a.max_retries == 8
    a._ensure_client()
    assert a._client.max_retries == 8

    # explicit value is stored and forwarded — openai branch
    o = ev.LLM(model="gpt-4", api_key="x", max_retries=3)
    assert o.max_retries == 3
    o._ensure_client()
    assert o._client.max_retries == 3


def test_evolvable_local_call():
    def tldr(input_text: str, llm) -> str:
        return input_text[:130] + "..."

    cr_len = ev.code(lambda output_text: 1.0 if len(output_text) <= 140 else -1.0)
    llm = ev.LLM(model="claude-opus-4-7")
    evo = ev.Evolvable(tldr, criteria=[cr_len], llm=llm)

    result = evo.call_sync("hello world " * 30)
    assert isinstance(result, str)
    assert len(result) <= 140


def test_evolvable_save_load(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLVERS_CACHE", str(tmp_path))

    def tldr(input_text: str, llm) -> str:
        return input_text[:130] + "..."

    cr_len = ev.code(
        lambda output_text: 1.0 if len(output_text) <= 140 else -1.0,
        name="length",
    )
    llm = ev.LLM(model="claude-opus-4-7")
    evo = ev.Evolvable(tldr, criteria=[cr_len], llm=llm)

    evo.save("test/tldr-v0:test")
    reloaded = ev.Evolvable.load("test/tldr-v0:test")

    assert reloaded.call_sync("hello world " * 30) == evo.call_sync("hello world " * 30)
    assert len(reloaded.criteria) == 1
    assert reloaded.criteria[0].name == "length"
    fn = reloaded.criteria[0].fn
    assert fn is not None
    assert fn("x" * 300) == -1.0


def test_evolvable_evaluate_with_dict_rows():
    """Regression: _row_to_call's dict-row path used to iterate sig.parameters
    (yielding str keys) and call .name on each — would AttributeError on any
    dict-row input. evaluate_sync exercises the real path end-to-end."""

    def tldr(input_text: str, llm) -> str:
        return input_text[:130] + "..."

    cr = ev.code(lambda output_text: 1.0, name="trivial")
    llm = ev.LLM(model="claude-opus-4-7")
    evo = ev.Evolvable(tldr, criteria=[cr], llm=llm)

    result = evo.evaluate_sync([{"input_text": "hello world"}])
    assert result["aggregate"] == 1.0


def test_evolvable_clone_and_set_llm():
    def tldr(input_text: str, llm) -> str:
        return input_text[:130] + "..."

    cr = ev.code(lambda output_text: 1.0, name="trivial")
    llm = ev.LLM(model="claude-opus-4-7")
    evo = ev.Evolvable(tldr, criteria=[cr], llm=llm)

    llm2 = ev.LLM(model="claude-haiku-4-5-20251001")
    evo2 = evo.clone().set_llm(llm2)

    assert evo2.llm.model == "claude-haiku-4-5-20251001"
    assert evo.llm.model == "claude-opus-4-7"
    assert evo2.criteria is not evo.criteria
