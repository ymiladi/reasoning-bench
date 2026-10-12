"""Grade raw samples -> results/graded/<model>/<task>.jsonl. Re-runnable without re-generating."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .graders import grade
from .tasks import load_items


def _grade_one(args):
    item, sample = args
    try:
        g = grade(item, sample)
    except Exception as e:  # noqa: BLE001
        g = {"correct": False, "score": 0.0, "pred": None, "outcome": f"grader_error:{type(e).__name__}"}
    content = sample.get("content") or ""
    return {
        "model": sample["model"], "task": sample["task"], "item_id": sample["item_id"],
        "sample_idx": sample["sample_idx"], **g,
        "output_tokens": sample.get("output_tokens") or 0,
        "input_tokens": sample.get("input_tokens") or 0,
        "latency_s": sample.get("latency_s"), "finish_reason": sample.get("finish_reason"),
        "error": sample.get("error"),
        "reasoning_chars": len(sample.get("reasoning") or ""),
        "content_tail": content[-600:],
    }


def grade_all(out_dir: Path, data_dir: Path, workers: int = 8):
    raw_root = out_dir / "raw"
    for model_dir in sorted(p for p in raw_root.glob("*") if p.is_dir()):
        for raw in sorted(model_dir.glob("*.jsonl")):
            task = raw.stem
            items = {it["id"]: it for it in load_items(task, data_dir)}
            with open(raw) as f:
                samples = [json.loads(line) for line in f if line.strip()]
            jobs = [(items[s["item_id"]], s) for s in samples if s["item_id"] in items]
            if items and next(iter(items.values()))["grader"] == "code" and workers > 1:
                with ProcessPoolExecutor(workers) as ex:
                    graded = list(ex.map(_grade_one, jobs, chunksize=4))
            else:
                graded = [_grade_one(j) for j in jobs]
            out = out_dir / "graded" / model_dir.name / f"{task}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                for g in graded:
                    f.write(json.dumps(g, ensure_ascii=False) + "\n")
            acc = sum(g["correct"] for g in graded) / max(1, len(graded))
            print(f"graded {model_dir.name}/{task}: {len(graded)} samples, raw acc {acc:.3f}")
