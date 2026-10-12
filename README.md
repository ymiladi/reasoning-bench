# Reasoning benchmark: Falcon-H1R-7B vs K2 Think V2

A resumable harness that runs the same reasoning tasks through two open-weight models
(served with vLLM, or any OpenAI-compatible endpoint), grades every answer
deterministically, and produces a report with confidence intervals, paired significance
tests, token efficiency and cost.

| Model | Size | Served as | Sampling (from each model's own card/docs) |
|---|---|---|---|
| `tiiuae/Falcon-H1R-7B` | 7.6B hybrid Transformer+Mamba2 | vLLM ≥ 0.11, `--reasoning-parser deepseek_r1` | temp 0.6, top_p 0.95 |
| `IFM/K2-Think-V2` | 73B dense | vLLM, bf16, TP=4 | temp 1.0, top_p 1.0, `reasoning_effort=high` |

Both share one **64k-token output budget**, so no model gets more room to think.
Truncation is reported per model rather than hidden.

## Task suite (`configs/suite.yaml`)

| Task | Items × samples | Grading | Why it's in |
|---|---|---|---|
| AIME + HMMT 2026 | 63 × 8 | exact / symbolic | **Headline maths.** Released after both open models, so the cleanest contamination-free signal |
| AIME 2025 (reference) | 30 × 8 | exact | Both model cards report it. Checks the setup reproduces published numbers (Falcon 83.1, K2 90.4) |
| MATH-500 (150) | 150 × 2 | `math-verify` | Floor check, stratified by level |
| GPQA Diamond | 198 × 4 | option letter | Graduate-level science. Options are shuffled once per item, identically for every model |
| LiveCodeBench 2025 | 120 × 2 | hidden unit tests | Code reasoning, stratified by difficulty |
| ZebraLogic (grid) | 120 × 2 | all cells correct | Pure deduction, stratified by grid size. Cell accuracy is also reported |
| *custom* (optional) | yours | math / mcq / exact | Your own unpublished problems: the best defence against contamination |

**Answer extraction rules**
- Answers are read only from the model's **final response**, never from its reasoning trace.
- If a model runs out of budget mid-thought, the sample scores as `truncated`.

## Running it

You need:
- A Linux GPU box. The default layout is **8× H100 80GB**: K2 on GPUs 0–3, Falcon on 4–7. A 4-GPU box also works if you run the two models one after the other.
- A Hugging Face token, and you must accept the terms of the two gated datasets:
  [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) and
  [ZebraLogic](https://huggingface.co/datasets/allenai/ZebraLogicBench-private).

```bash
export HF_TOKEN=hf_...
bash scripts/setup_gpu_box.sh     # venv, vLLM, weights (~300 GB disk), freezes items into data/
python -m bench estimate          # forecast tokens and GPU hours before spending anything
bash scripts/serve_models.sh      # both vLLM servers, waits until healthy
bash scripts/run_benchmark.sh     # smoke test (2 items/task), then full run, grading, report
```

Rough budget with defaults (`python -m bench estimate`): about 3 GPU-box hours for Falcon
and 7 for K2. The throughput figures behind the estimate are assumptions. The report uses measured wall-clock time instead.

**Everything is resumable.** Re-running `python -m bench run` skips finished samples and retries failed ones.

Step by step, if you prefer to run each stage yourself:

```bash
python -m bench prepare                    # download, sample, freeze -> data/<task>.jsonl
python -m bench inspect                    # sanity-check item counts and prompts
python -m bench run --models k2-think-v2 --tasks aime_2025 --limit 3
python -m bench grade                      # re-runnable; regenerating isn't needed if a grader changes
python -m bench report                     # results/summary.json + results/report.html
```

## Reading the results

- **pass@1**: mean accuracy over the k samples for each item, then averaged over items. The 95% CI comes from bootstrapping over items.
- **maj@k / pass@k**: majority-vote accuracy, and whether any of the k samples was correct. Together they show how much the model gains from repeated sampling.
- **Head-to-head**: a paired difference over the same items, with a sign-flip permutation p-value. "No clear winner" means p ≥ 0.05.
- **Outcome bars**: break down how answers fail: wrong answer, no final answer, truncated, or (for code) compile error, runtime error or timeout.
- **Cost**: GPU $/hour × measured wall-clock time. Set your actual rate in `configs/models.yaml`.

## Safety and data handling

- **Untrusted code.** LiveCodeBench grading runs model-written code. `scripts/grade_sandboxed.sh` runs it in a Docker container with no network and resource limits, in addition to the harness's own per-test timeouts and memory limits.
- **Hidden test cases.** LiveCodeBench ships these as a pickled string. They are decoded with an unpickler that refuses any object other than plain data.
- **Gated dataset text.** GPQA's terms forbid publishing its examples. The report never shows GPQA text, and `data/` and `results/` are git-ignored.

## Adding your own problems

Add one JSON object per line to a file such as `custom/my_tasks.jsonl` (see `custom/example.jsonl`):

```json
{"id": "q1", "prompt": "...", "answer": "42", "grader": "math"}
{"id": "q2", "prompt": "... End with 'Answer: X'.", "answer": "C", "grader": "mcq"}
```

Then uncomment the `custom` block in `configs/suite.yaml`. Problems you write yourself and
have never published are the only fully contamination-proof part of any benchmark.

## Offline dry run (no GPU, no API)

```bash
PYTHONPATH=. python tests/test_graders.py                      # extractor, grader and sandbox tests
PYTHONPATH=. python tests/make_fixtures.py /tmp/dry/data
python -m bench run    --models-config configs/models.mock.yaml --data /tmp/dry/data --out /tmp/dry/results
python -m bench grade  --data /tmp/dry/data --out /tmp/dry/results
python -m bench report --models-config configs/models.mock.yaml --data /tmp/dry/data --out /tmp/dry/results
```

## Caveats worth stating in any write-up

- **Contamination.** Only the 2026 competition set and your custom set post-date both models. Treat the others as "may have been seen".
- **Different sampling settings.** Each model uses its own recommended settings: the comparison is each model at its documented best, not at identical temperature.
- **Single-run variance.** k samples per item shrink variance but don't remove it. Read the CIs and p-values before calling a winner on a small task.
