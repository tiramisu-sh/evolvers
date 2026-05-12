import os
import evolvers as ev

dataset = [  # the actual dataset shape should be huggingface-compatible; this one is just for example
    "very long text 1",
    "very long text 2",
    "very long text 3",
    "very long text 4",
    "very long text 5",
    "very long text 6",
    "very long text 7",
    "very long text 8",
    "very long text 9",
    "very long text 10",
]

eval_dataset = [
    "very long text 11",
    "very long text 12",
    "very long text 13",
    "very long text 14",
    "very long text 15",
]


# defining an evolvable function / program
# `llm` is auto-injected at call time from the bound LLM on the Evolvable.
def tldr_template(input_text: str, llm) -> str:
    return input_text[:130] + '...'  # naive baseline; the optimizer rewrites this body


# defining criteria for the evolvable function
# both factories return a Criterion value; judges materialize to .judge.txt, code to .code.py
cr_essential = ev.judge("Does it directly summarize the main points as a 'TLDR'?")
cr_length = ev.code(lambda output_text: max(-1.0, 1 - 2 * max(0, (len(output_text) - 140) / 140)))

llm_opus = ev.LLM(model="claude-opus-4-7", api_key=os.getenv("CLAUDE_API_KEY"))


tldr_opus = ev.Evolvable(
    tldr_template,
    criteria=[cr_essential, cr_length],  # each criterion returns a score between -1 and 1
    llm=llm_opus,
)

# training the evolvable function
metrics = tldr_opus.train(
    dataset=dataset,
    budget=10,
    show_progress=True,
)

# evaluating the evolvable function
metrics = tldr_opus.evaluate(
    dataset=eval_dataset,
    show_progress=True,
)

# saving the evolvable function
tldr_opus.save("vvsotnikov/tldr-v1:claude-opus-4-7")

# retrain with local LLM
llm_local = ev.LLM(model="qwen/qwen-3.6-32b", api_key="1234", base_url="http://localhost:8001/v1")

tldr_local = tldr_opus.clone().set_llm(llm_local)

metrics = tldr_local.train(
    dataset=dataset,
    budget=10,
    show_progress=True,
)

metrics = tldr_local.evaluate(
    dataset=eval_dataset,
    show_progress=True,
)

tldr_local.save("vvsotnikov/tldr-v1:qwen-3.6-32b")  # shares the same signature; optimized for a different LLM; docker image-inspired

# using the evolvable function
tldr_local = ev.Evolvable.load("vvsotnikov/tldr-v1:qwen-3.6-32b")

print(tldr_local("very long text 1"))