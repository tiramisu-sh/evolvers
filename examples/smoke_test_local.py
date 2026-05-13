"""Tight smoke test: 1 example, budget=1, against local vLLM (deepkek).

Run: uv run python -u examples/smoke_test_local.py 2>&1 | tee smoke_test_local.log
"""

import os
import tempfile

os.environ.setdefault("EVOLVERS_CACHE", tempfile.mkdtemp(prefix="evolvers_smoke_"))

import evolvers as ev

DATASET = [
    "A new study from Stanford and MIT (published in Nature, May 2026) found that "
    "ground-based gamma-ray observatories underestimate cosmic-ray hadronic backgrounds "
    "by a factor of 1.5 at PeV energies. The discrepancy traces to QGSJet-II.04 simulation "
    "tuning that pre-dates LHC-era hadronic cross-section measurements. The authors recommend "
    "retuning with EPOS-LHC and SIBYLL ensembles before claiming PeVatron detections.",
]


def tldr(input_text: str, llm) -> str:
    return input_text[:130] + "..."


def main() -> None:
    cr_essential = ev.judge(
        "Does it directly summarize the main points as a TLDR (concise, captures the key claim and its implication)?",
        name="essential",
    )
    cr_length = ev.code(
        lambda output_text: max(-1.0, 1 - 2 * max(0, (len(output_text) - 140) / 140)),
        name="length",
    )

    llm = ev.LLM(model="deepkek", base_url="http://localhost:8001/v1", api_key="dummy")
    evo = ev.Evolvable(tldr, criteria=[cr_essential, cr_length], llm=llm)

    print("=== train budget=1 ===", flush=True)
    result = evo.train(DATASET, budget=1, show_progress=False)
    print(f"best_score={result['best_score']:.3f}", flush=True)

    print("=== best source ===", flush=True)
    print(evo.source, flush=True)

    print("=== call best on new input ===", flush=True)
    new_input = (
        "OpenAI announced GPT-5.5 yesterday, focusing on faster inference (3x speedup "
        "over GPT-5.4) via mixture-of-experts routing improvements. Pricing drops to "
        "$0.50/M input tokens and $2/M output."
    )
    print(repr(evo(new_input)), flush=True)


if __name__ == "__main__":
    main()
