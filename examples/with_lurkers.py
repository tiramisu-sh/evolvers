"""Cross-package demo: fetch a small dataset with `lurkers`, train an
`evolvers` TLDR program on it, persist the dataset + the evolved program.

Prerequisites:
    pip install evolvers lurkers
    # an LLM endpoint reachable from this script (this example uses a local
    # OpenAI-compatible vLLM at http://localhost:8001/v1).

Run:
    uv run --with evolvers --with lurkers python -u examples/with_lurkers.py

After the run, two artifacts are persisted for later inspection:
    - dataset:          ~/lurkers_demo/dataset.jsonl
    - evolved program:  ~/.cache/evolvers/vvsotnikov/tldr-lurkers:deepkek/
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import lurkers

import evolvers as ev

HN_RSS = "https://news.ycombinator.com/rss"
YT_URLS = [
    "https://www.youtube.com/watch?v=jNQXAC9IVRw",
    "https://www.youtube.com/watch?v=9bZkp7q19f0",
]

DATASET_DIR = Path.home() / "lurkers_demo"
DATASET_PATH = DATASET_DIR / "dataset.jsonl"
ARTIFACT_URI = "vvsotnikov/tldr-lurkers:deepkek"


def tldr(input_text: str, llm) -> str:
    """Naive baseline; the optimizer rewrites this body to use `llm`."""
    return input_text[:130] + "..."


def build_dataset() -> tuple[list[str], list]:
    print("fetching HN front page (top 3 entries)...", flush=True)
    hn = lurkers.feed(HN_RSS, limit=3)
    print(f"  got {len(hn)} HN articles", flush=True)

    print("fetching YouTube videos...", flush=True)
    yt = [lurkers.fetch(u) for u in YT_URLS]
    print(f"  got {len(yt)} YouTube documents", flush=True)

    docs = [d for d in hn + yt if d.content and d.content.strip()]
    items = [d.content for d in docs]
    avg = sum(len(s) for s in items) // max(1, len(items))
    print(f"dataset: {len(items)} items, avg chars: {avg}", flush=True)
    return items, docs


def save_dataset(docs: list) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with DATASET_PATH.open("w") as f:
        for d in docs:
            f.write(d.model_dump_json() + "\n")
    print(f"saved {len(docs)} documents to {DATASET_PATH}", flush=True)


async def main() -> None:
    dataset, docs = build_dataset()
    if not dataset:
        print("no content fetched; aborting", flush=True)
        return

    save_dataset(docs)

    cr_essential = ev.judge(
        "Does it directly summarize the main points of the input text as a TLDR?",
        name="essential",
    )
    cr_length = ev.code(
        lambda output_text: max(-1.0, 1 - 2 * max(0, (len(output_text) - 140) / 140)),
        name="length",
    )

    llm = ev.LLM(model="deepkek", base_url="http://localhost:8001/v1", api_key="dummy", max_concurrency=32)
    evo = ev.Evolvable(tldr, criteria=[cr_essential, cr_length], llm=llm)

    print("\n=== training ===", flush=True)
    await evo.train(dataset, num_train_epochs=2, show_progress=False)

    print("\n=== best source ===", flush=True)
    print(evo.source, flush=True)

    saved_path = evo.save(ARTIFACT_URI)
    print(f"\nsaved evolved program to {saved_path}", flush=True)

    print("\n=== sample invocations ===", flush=True)
    for d in dataset[:3]:
        preview = d[:120].replace("\n", " ")
        print(f"\nINPUT:  {preview}...", flush=True)
        print(f"TLDR:   {await evo(d)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
