"""Generation runner: resumable, concurrent, writes raw samples to results/raw/<model>/<task>.jsonl."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from .backends import make_backend
from .tasks import load_items


def samples_for(model_cfg, task_name, task_cfg):
    return model_cfg.get("samples_override", {}).get(task_name, task_cfg.get("samples", 1))


def _done_keys(path: Path):
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("error"):          # errored samples are retried on resume
                    done.add((r["item_id"], r["sample_idx"]))
    return done


def _compact(path: Path):
    """Keep only the latest record per (item, sample) - drops superseded error rows."""
    if not path.exists():
        return
    latest = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r["item_id"], r["sample_idx"])
            if key not in latest or latest[key].get("error"):
                latest[key] = r
    with open(path, "w") as f:
        for r in latest.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def run_task(model_cfg, task_name, task_cfg, data_dir: Path, out_dir: Path,
                   limit=None):
    items = load_items(task_name, data_dir)
    if limit:
        items = items[:limit]
    k = samples_for(model_cfg, task_name, task_cfg)
    path = out_dir / "raw" / model_cfg["name"] / f"{task_name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(path)
    jobs = [(it, s) for it in items for s in range(k) if (it["id"], s) not in done]
    print(f"[{model_cfg['name']}] {task_name}: {len(items)} items x {k} samples, "
          f"{len(done)} done, {len(jobs)} to run")
    if not jobs:
        return
    backend = make_backend(model_cfg)
    t_start = time.time()
    totals = {"n": 0, "errors": 0, "out_tok": 0, "in_tok": 0}
    fout = open(path, "a")

    def write(item, s, rec):
        row = {"model": model_cfg["name"], "task": task_name, "item_id": item["id"],
               "sample_idx": s, **rec, "ts": time.time()}
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()
        totals["n"] += 1
        totals["errors"] += bool(rec.get("error"))
        totals["out_tok"] += rec.get("output_tokens") or 0
        totals["in_tok"] += rec.get("input_tokens") or 0

    sem = asyncio.Semaphore(backend.concurrency)
    progress = {"n": 0}

    async def one(item, s):
        async with sem:
            rec = await backend.generate(item, s)
        write(item, s, rec)
        progress["n"] += 1
        if progress["n"] % max(1, len(jobs) // 20) == 0 or progress["n"] == len(jobs):
            el = time.time() - t_start
            print(f"  [{model_cfg['name']}] {task_name}: {progress['n']}/{len(jobs)} "
                  f"({el / 60:.1f} min, {totals['out_tok'] / max(el, 1):.0f} out tok/s, "
                  f"{totals['errors']} errors)")

    await asyncio.gather(*(one(it, s) for it, s in jobs))
    fout.close()
    _compact(path)
    wall = time.time() - t_start
    with open(out_dir / "runs.jsonl", "a") as f:
        f.write(json.dumps({"model": model_cfg["name"], "task": task_name, "wall_s": wall,
                            **totals, "ts": time.time()}) + "\n")
    print(f"[{model_cfg['name']}] {task_name}: finished in {wall / 60:.1f} min "
          f"({totals['errors']} errors)")
