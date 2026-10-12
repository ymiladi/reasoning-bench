"""Benchmark task definitions: download, sample, and freeze items to data/<task>.jsonl.

Freezing matters: every model sees byte-identical prompts (including the GPQA
option order), and the exact item set is recorded for reproducibility.
"""
from __future__ import annotations

import base64
import io
import json
import pickle
import random
import zlib
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------- prompts

MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

GPQA_TMPL = (
    "Answer the following multiple-choice question. Think step by step, then end your "
    "response with a final line of the form 'Answer: X' where X is one of A, B, C, D.\n\n"
    "{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)

LCB_SYS = ("You will be given a question (problem specification) and will generate a correct "
           "Python program that matches the specification and passes all tests.")
LCB_STDIN = (LCB_SYS + "\n\nQuestion: {q}\n\nRead the inputs from stdin, solve the problem and write "
             "the answer to stdout (do not directly test on the sample inputs). Enclose your code "
             "within delimiters as follows. Ensure that when the python program runs, it reads the "
             "inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```")
LCB_FUNC = (LCB_SYS + "\n\nQuestion: {q}\n\nYou will use the following starter code to write the "
            "solution to the problem and enclose your code within delimiters.\n```python\n{starter}\n```")

ZEBRA_TMPL = (
    "Solve the following logic grid puzzle. Every attribute value is used exactly once.\n\n"
    "{puzzle}\n\n"
    "Think it through, then give your final answer as a single JSON object in a ```json code "
    "block with exactly this structure (fill in every value, using the spellings from the puzzle):\n"
    "```json\n{template}\n```"
)


# ---------------------------------------------------------------- helpers

def _hf_load(repo, name=None, split=None):
    from datasets import load_dataset
    return load_dataset(repo, name, split=split)


def _stratified_sample(rows, key, n, seed):
    """Sample n rows spreading evenly across strata defined by key(row)."""
    if n is None or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in rows:
        buckets[key(r)].append(r)
    for b in buckets.values():
        rng.shuffle(b)
    out, keys = [], sorted(buckets)
    while len(out) < n:
        for k in keys:
            if buckets[k] and len(out) < n:
                out.append(buckets[k].pop())
    return out


class _StrOnlyUnpickler(pickle.Unpickler):
    """LiveCodeBench pickles a plain JSON string; refuse anything that needs globals."""
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def _decode_lcb_private(s: str):
    try:
        return json.loads(s)
    except Exception:
        raw = zlib.decompress(base64.b64decode(s.encode("utf-8")))
        return json.loads(_StrOnlyUnpickler(io.BytesIO(raw)).load())


# ---------------------------------------------------------------- loaders
# Each returns a list of item dicts: {id, prompt, gold, grader, meta}

def load_competition_math(cfg):
    items = []
    for src in cfg["sources"]:
        ds = _hf_load(src["repo"], split=src.get("split", "train"))
        for r in ds:
            items.append({
                "id": f"{src['label']}-{r['problem_idx']}",
                "prompt": r["problem"] + MATH_SUFFIX,
                "gold": str(r["answer"]),
                "grader": "math",
                "meta": {"source": src["label"]},
            })
    return items


def load_math500(cfg):
    ds = list(_hf_load("HuggingFaceH4/MATH-500", split="test"))
    rows = _stratified_sample(ds, lambda r: r["level"], cfg.get("n_items"), cfg.get("seed", 0))
    return [{
        "id": r["unique_id"],
        "prompt": r["problem"] + MATH_SUFFIX,
        "gold": r["answer"],
        "grader": "math",
        "meta": {"level": r["level"], "subject": r["subject"]},
    } for r in rows]


def load_gpqa(cfg):
    ds = list(_hf_load("Idavidrein/gpqa", "gpqa_diamond", split="train"))
    rows = _stratified_sample(ds, lambda r: r.get("High-level domain", ""), cfg.get("n_items"),
                              cfg.get("seed", 0))
    items = []
    for i, r in enumerate(rows):
        rid = r.get("Record ID") or f"gpqa-{i}"
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"],
                r["Incorrect Answer 3"]]
        opts = [o.strip() for o in opts]
        rng = random.Random(f"{cfg.get('seed', 0)}-{rid}")   # fixed per item, same for all models
        perm = list(range(4))
        rng.shuffle(perm)
        letters = "ABCD"
        shown = {letters[k]: opts[perm[k]] for k in range(4)}
        gold = letters[perm.index(0)]
        items.append({
            "id": rid,
            "prompt": GPQA_TMPL.format(question=r["Question"].strip(), **shown),
            "gold": gold,
            "grader": "mcq",
            "meta": {"domain": r.get("High-level domain", ""),
                     "subdomain": r.get("Subdomain", ""), "sensitive": True},
        })
    return items


def load_livecodebench(cfg):
    from huggingface_hub import hf_hub_download
    rows = []
    for fn in cfg.get("files", ["test5.jsonl", "test6.jsonl"]):
        path = hf_hub_download("livecodebench/code_generation_lite", fn, repo_type="dataset")
        with open(path) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    lo, hi = cfg.get("date_from", "0000"), cfg.get("date_to", "9999")
    rows = [r for r in rows if lo <= r["contest_date"][:10] <= hi]
    rows = _stratified_sample(rows, lambda r: r["difficulty"], cfg.get("n_items"), cfg.get("seed", 0))
    items = []
    for r in rows:
        public = json.loads(r["public_test_cases"])
        private = _decode_lcb_private(r["private_test_cases"])
        meta = json.loads(r["metadata"]) if r.get("metadata") else {}
        functional = bool(r.get("starter_code", "").strip())
        prompt = (LCB_FUNC.format(q=r["question_content"], starter=r["starter_code"]) if functional
                  else LCB_STDIN.format(q=r["question_content"]))
        items.append({
            "id": r["question_id"],
            "prompt": prompt,
            "gold": {"tests": public + private, "func_name": meta.get("func_name")},
            "grader": "code",
            "meta": {"difficulty": r["difficulty"], "platform": r["platform"],
                     "contest_date": r["contest_date"][:10], "n_tests": len(public) + len(private)},
        })
    return items


def load_zebralogic(cfg):
    ds = list(_hf_load("allenai/ZebraLogicBench-private", "grid_mode", split="test"))

    def bucket(r):
        h, w = (int(x) for x in r["size"].split("*"))
        cells = h * w
        return "small" if cells < 10 else "medium" if cells < 20 else "large" if cells < 30 else "xl"

    rows = _stratified_sample(ds, bucket, cfg.get("n_items"), cfg.get("seed", 0))
    items = []
    for r in rows:
        header, sol_rows = r["solution"]["header"], r["solution"]["rows"]
        template = {f"House {i + 1}": {c: "___" for c in header[1:]} for i in range(len(sol_rows))}
        items.append({
            "id": r["id"],
            "prompt": ZEBRA_TMPL.format(puzzle=r["puzzle"], template=json.dumps(template, indent=2)),
            "gold": {"header": header, "rows": sol_rows},
            "grader": "zebra",
            "meta": {"size": r["size"], "bucket": bucket(r)},
        })
    return items


def load_custom(cfg):
    """Your own held-out problems: JSONL with id, prompt, answer, grader (math|mcq|exact)."""
    items = []
    with open(cfg["path"]) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            g = r.get("grader", "math")
            prompt = r["prompt"] + (MATH_SUFFIX if g == "math" and "\\boxed" not in r["prompt"] else "")
            items.append({"id": str(r["id"]), "prompt": prompt, "gold": r["answer"], "grader": g,
                          "meta": r.get("meta", {})})
    return items


LOADERS = {
    "competition_math": load_competition_math,
    "math500": load_math500,
    "gpqa": load_gpqa,
    "livecodebench": load_livecodebench,
    "zebralogic": load_zebralogic,
    "custom": load_custom,
}


def prepare(task_name: str, cfg: dict, data_dir: Path) -> Path:
    items = LOADERS[cfg["loader"]](cfg)
    if cfg.get("max_items"):
        items = items[: cfg["max_items"]]
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"{task_name}.jsonl"
    with open(out, "w") as f:
        for it in items:
            it["task"] = task_name
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return out


def load_items(task_name: str, data_dir: Path) -> list[dict]:
    path = data_dir / f"{task_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m bench prepare` first")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
