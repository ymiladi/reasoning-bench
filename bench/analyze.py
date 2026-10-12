"""Statistics: accuracy with item-level bootstrap CIs, paired model comparisons, efficiency and cost."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from .tasks import load_items

B = 5000            # bootstrap / permutation resamples
RNG = np.random.default_rng(12345)


def _boot_ci(vals: np.ndarray, b=B):
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(vals), size=(b, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired(a: np.ndarray, b: np.ndarray):
    """Mean diff a-b over shared items, bootstrap CI, and sign-flip permutation p-value."""
    d = a - b
    lo, hi = _boot_ci(d)
    obs = abs(d.mean())
    signs = RNG.choice([-1.0, 1.0], size=(B, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    p = float((np.sum(perm >= obs - 1e-12) + 1) / (B + 1))
    return {"diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "p_value": p, "n_items": int(len(d))}


def _majority_correct(rows):
    """Majority vote over predictions (math/mcq); None if not applicable."""
    preds = [(r.get("pred"), r["correct"]) for r in rows if r.get("pred") is not None]
    if not preds:
        return 0.0
    counts = Counter(p for p, _ in preds)
    top = counts.most_common(1)[0][0]
    return float(next(c for p, c in preds if p == top))


def load_graded(out_dir: Path):
    rows = []
    for f in sorted((out_dir / "graded").glob("*/*.jsonl")):
        with open(f) as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def analyze(out_dir: Path, data_dir: Path, models_cfg: dict, suite_cfg: dict):
    rows = load_graded(out_dir)
    if not rows:
        raise SystemExit("no graded results - run `python -m bench grade` first")
    models = [m["name"] for m in models_cfg["models"] if any(r["model"] == m["name"] for r in rows)]
    mcfg = {m["name"]: m for m in models_cfg["models"]}
    tasks = [t for t in suite_cfg["tasks"] if any(r["task"] == t for r in rows)]

    # item-level scores: by[model][task][item] = list of sample rows
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        by[r["model"]][r["task"]][r["item_id"]].append(r)

    items_meta = {t: {it["id"]: it for it in load_items(t, data_dir)} for t in tasks}

    runs = []
    if (out_dir / "runs.jsonl").exists():
        with open(out_dir / "runs.jsonl") as f:
            runs = [json.loads(line) for line in f if line.strip()]

    summary = {"models": models, "tasks": tasks, "per_task": {}, "overall": {}, "pairwise": {},
               "breakdowns": {}, "agreement": {}, "cost": {}, "examples": {},
               "task_info": {t: {"label": suite_cfg["tasks"][t].get("label", t),
                                 "description": suite_cfg["tasks"][t].get("description", "")}
                             for t in tasks},
               "model_info": {m: {"label": mcfg[m].get("label", m), "notes": mcfg[m].get("notes", "")}
                              for m in models}}

    item_scores = {}   # (model, task) -> {item: mean score}
    for m in models:
        for t in tasks:
            its = by[m][t]
            if not its:
                continue
            ids = sorted(its)
            scores = np.array([np.mean([r["score"] for r in its[i]]) for i in ids])
            item_scores[(m, t)] = dict(zip(ids, scores))
            all_s = [r for i in ids for r in its[i]]
            out_tok = np.array([r["output_tokens"] for r in all_s], dtype=float)
            lat = np.array([r["latency_s"] for r in all_s if r.get("latency_s")], dtype=float)
            n_correct = sum(r["correct"] for r in all_s)
            outcomes = Counter(r["outcome"] for r in all_s)
            lo, hi = _boot_ci(scores)
            entry = {
                "n_items": len(ids), "n_samples": len(all_s),
                "k": round(len(all_s) / len(ids), 2),
                "acc": float(scores.mean()), "ci_lo": lo, "ci_hi": hi,
                "pass_at_k": float(np.mean([any(r["correct"] for r in its[i]) for i in ids])),
                "maj_at_k": float(np.mean([_majority_correct(its[i]) for i in ids]))
                if _grader_of(items_meta[t]) in ("math", "mcq", "exact") else None,
                "mean_out_tokens": float(out_tok.mean()),
                "median_out_tokens": float(np.median(out_tok)),
                "out_tokens_per_correct": float(out_tok.sum() / n_correct) if n_correct else None,
                "latency_p50": float(np.percentile(lat, 50)) if len(lat) else None,
                "latency_p90": float(np.percentile(lat, 90)) if len(lat) else None,
                "truncation_rate": outcomes.get("truncated", 0) / len(all_s),
                "error_rate": outcomes.get("api_error", 0) / len(all_s),
                "outcomes": dict(outcomes),
            }
            if any("cell_acc" in r for r in all_s):
                entry["cell_acc"] = float(np.mean([r.get("cell_acc", 0.0) for r in all_s]))
            summary["per_task"].setdefault(t, {})[m] = entry

            # breakdowns by meta field
            meta_keys = suite_cfg["tasks"][t].get("breakdown_by", [])
            for key in meta_keys:
                groups = defaultdict(list)
                for i in ids:
                    v = items_meta[t].get(i, {}).get("meta", {}).get(key)
                    groups[str(v)].append(item_scores[(m, t)][i])
                summary["breakdowns"].setdefault(t, {}).setdefault(key, {})[m] = {
                    g: {"acc": float(np.mean(v)), "n": len(v)} for g, v in sorted(groups.items())}

            # failure examples (never for sensitive datasets such as GPQA)
            sensitive = suite_cfg["tasks"][t].get("sensitive", False)
            if not sensitive:
                fails = [r for r in all_s if not r["correct"] and r["outcome"] != "api_error"][:3]
                summary["examples"].setdefault(t, {})[m] = [
                    {"item_id": r["item_id"], "outcome": r["outcome"], "pred": r.get("pred"),
                     "gold": _short_gold(items_meta[t].get(r["item_id"])),
                     "tail": r.get("content_tail", "")[-400:]} for r in fails]

    # overall macro average across tasks (equal task weight), CI by bootstrapping each task
    for m in models:
        tks = [t for t in tasks if (m, t) in item_scores]
        if not tks:
            continue
        point = float(np.mean([np.mean(list(item_scores[(m, t)].values())) for t in tks]))
        boots = np.zeros(B)
        for t in tks:
            v = np.array(list(item_scores[(m, t)].values()))
            boots += v[RNG.integers(0, len(v), size=(B, len(v)))].mean(axis=1)
        boots /= len(tks)
        summary["overall"][m] = {"macro_acc": point, "ci_lo": float(np.percentile(boots, 2.5)),
                                 "ci_hi": float(np.percentile(boots, 97.5)), "n_tasks": len(tks)}

    # pairwise paired comparisons per task + agreement
    for a, b in combinations(models, 2):
        key = f"{a}__vs__{b}"
        summary["pairwise"][key] = {}
        summary["agreement"][key] = {}
        for t in tasks:
            sa, sb = item_scores.get((a, t)), item_scores.get((b, t))
            if not sa or not sb:
                continue
            common = sorted(set(sa) & set(sb))
            va = np.array([sa[i] for i in common])
            vb = np.array([sb[i] for i in common])
            summary["pairwise"][key][t] = _paired(va, vb)
            solved_a, solved_b = va >= 0.5, vb >= 0.5
            summary["agreement"][key][t] = {
                "both": int((solved_a & solved_b).sum()), "only_a": int((solved_a & ~solved_b).sum()),
                "only_b": int((~solved_a & solved_b).sum()), "neither": int((~solved_a & ~solved_b).sum())}

    # cost
    for m in models:
        c = mcfg[m].get("cost", {})
        tot_in = sum(r["input_tokens"] for r in rows if r["model"] == m)
        tot_out = sum(r["output_tokens"] for r in rows if r["model"] == m)
        n_correct = sum(r["correct"] for r in rows if r["model"] == m)
        entry = {"input_tokens": tot_in, "output_tokens": tot_out}
        if "usd_per_mtok_in" in c:
            usd = tot_in / 1e6 * c["usd_per_mtok_in"] + tot_out / 1e6 * c["usd_per_mtok_out"]
            entry["usd_estimated"] = usd
            entry["basis"] = "API list price"
        if "gpu_usd_per_hour" in c:
            wall_h = sum(r["wall_s"] for r in runs if r["model"] == m) / 3600
            entry["gpu_hours_wall"] = wall_h
            entry["usd_estimated"] = wall_h * c["gpu_usd_per_hour"]
            entry["basis"] = f"{c.get('gpu_desc', 'GPU')} at ${c['gpu_usd_per_hour']}/h x wall-clock"
        if entry.get("usd_estimated") is not None and n_correct:
            entry["usd_per_100_correct"] = entry["usd_estimated"] / n_correct * 100
        summary["cost"][m] = entry

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _grader_of(items):
    return next(iter(items.values()))["grader"] if items else None


def _short_gold(item):
    if not item:
        return None
    g = item["gold"]
    if isinstance(g, (str, int, float)):
        return str(g)[:80]
    return None
