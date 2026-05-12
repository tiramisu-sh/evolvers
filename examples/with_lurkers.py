"""Cross-package demo: fetch a small dataset with `lurkers`, train an
`evolvers` TLDR program on it.

Prerequisites:
    pip install evolvers lurkers
    # an LLM endpoint reachable from this script (this example uses a local
    # OpenAI-compatible vLLM at http://localhost:8001/v1).

Run:
    uv run --with evolvers --with lurkers python -u examples/with_lurkers.py
"""

from __future__ import annotations

import lurkers
import evolvers as ev

HN_RSS = "https://news.ycombinator.com/rss"
YT_URLS = [
    "https://www.youtube.com/watch?v=jNQXAC9IVRw",
    "https://www.youtube.com/watch?v=9bZkp7q19f0",
]


def tldr(input_text: str, llm) -> str:
    """Naive baseline; the optimizer rewrites this body to use `llm`."""
    return input_text[:130] + "..."


def build_dataset() -> list[str]:
    print("fetching HN front page (top 3 entries)...", flush=True)
    hn = lurkers.feed(HN_RSS, limit=3)
    print(f"  got {len(hn)} HN articles", flush=True)

    print("fetching YouTube videos...", flush=True)
    yt = [lurkers.fetch(u) for u in YT_URLS]
    print(f"  got {len(yt)} YouTube documents", flush=True)

    items = [d.content for d in hn + yt if d.content and d.content.strip()]
    avg = sum(len(s) for s in items) // max(1, len(items))
    print(f"dataset: {len(items)} items, avg chars: {avg}", flush=True)
    return items


def main() -> None:
    dataset = build_dataset()
    if not dataset:
        print("no content fetched; aborting", flush=True)
        return

    cr_essential = ev.judge(
        "Does it directly summarize the main points of the input text as a TLDR?",
        name="essential",
    )
    cr_length = ev.code(
        lambda output_text: max(-1.0, 1 - 2 * max(0, (len(output_text) - 140) / 140)),
        name="length",
    )

    llm = ev.LLM(model="deepkek", base_url="http://localhost:8001/v1", api_key="dummy")
    evo = ev.Evolvable(tldr, criteria=[cr_essential, cr_length], llm=llm)

    print("\n=== training ===", flush=True)
    evo.train(dataset, budget=2, show_progress=False)

    print("\n=== best source ===", flush=True)
    print(evo.source, flush=True)

    print("\n=== sample invocations ===", flush=True)
    for d in dataset[:3]:
        preview = d[:120].replace("\n", " ")
        print(f"\nINPUT:  {preview}...", flush=True)
        print(f"TLDR:   {evo(d)}", flush=True)


if __name__ == "__main__":
    main()
