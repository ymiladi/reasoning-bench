"""Deterministic graders. Each returns {"correct": bool, "score": float, "pred": ..., "outcome": str}.

outcome is one of:
  correct | wrong_answer | no_final_answer | truncated | api_error
  code: compile_error | runtime_error | timeout | wrong_output | memory_error | crashed
"""
from __future__ import annotations

from .extract import code_block, last_boxed, last_json_object, mcq_letter, strip_think


def _base(sample):
    """Shared handling for failed calls / budget exhaustion."""
    if sample.get("error"):
        return {"correct": False, "score": 0.0, "pred": None, "outcome": "api_error"}
    return None


def _no_answer(sample):
    outcome = "truncated" if sample.get("finish_reason") in ("length", "max_tokens") else "no_final_answer"
    return {"correct": False, "score": 0.0, "pred": None, "outcome": outcome}


def grade_math(item, sample):
    if (b := _base(sample)):
        return b
    text = strip_think(sample.get("content"))
    pred = last_boxed(text)
    if pred is None:
        return _no_answer(sample)
    gold = str(item["gold"])
    ok = False
    # Integer answers (AIME): compare exactly after normalisation.
    if gold.strip().lstrip("-").isdigit():
        p = pred.replace(",", "").replace("\\!", "").replace("$", "").strip()
        p = p.removeprefix("\\text{").removesuffix("}") if p.startswith("\\text{") else p
        try:
            ok = int(float(p)) == int(gold) and float(p) == int(float(p))
        except ValueError:
            ok = False
    if not ok:
        from math_verify import parse, verify
        try:
            g = parse(gold if "$" in gold or "\\boxed" in gold else f"${gold}$")
            p = parse(f"\\boxed{{{pred}}}")
            ok = bool(g) and bool(p) and verify(g, p)
        except Exception:
            ok = False
    return {"correct": ok, "score": float(ok), "pred": pred,
            "outcome": "correct" if ok else "wrong_answer"}


def grade_mcq(item, sample):
    if (b := _base(sample)):
        return b
    pred = mcq_letter(strip_think(sample.get("content")))
    if pred is None:
        return _no_answer(sample)
    ok = pred == item["gold"]
    return {"correct": ok, "score": float(ok), "pred": pred,
            "outcome": "correct" if ok else "wrong_answer"}


def grade_exact(item, sample):
    if (b := _base(sample)):
        return b
    text = strip_think(sample.get("content"))
    pred = last_boxed(text) or (text.strip().splitlines()[-1] if text.strip() else None)
    if pred is None:
        return _no_answer(sample)
    ok = pred.strip().lower() == str(item["gold"]).strip().lower()
    return {"correct": ok, "score": float(ok), "pred": pred,
            "outcome": "correct" if ok else "wrong_answer"}


def grade_zebra(item, sample):
    """Puzzle solved only if every cell matches (case-insensitive). score = cell accuracy."""
    if (b := _base(sample)):
        return b
    obj = last_json_object(strip_think(sample.get("content")))
    if obj is None:
        return _no_answer(sample)
    sol = obj.get("solution", obj) if isinstance(obj, dict) else None
    header, rows = item["gold"]["header"], item["gold"]["rows"]
    total = correct = 0
    for i, row in enumerate(rows):
        house = sol.get(f"House {i + 1}", {}) if isinstance(sol, dict) else {}
        if not isinstance(house, dict):
            house = {}
        for j in range(1, len(header)):
            total += 1
            pv = house.get(header[j])
            if isinstance(pv, list):
                pv = pv[0] if pv else None
            if pv is not None and str(pv).strip().lower() == str(row[j]).strip().lower():
                correct += 1
    ok = total > 0 and correct == total
    return {"correct": ok, "score": float(ok), "cell_acc": correct / total if total else 0.0,
            "pred": None, "outcome": "correct" if ok else "wrong_answer"}


def grade_code(item, sample):
    if (b := _base(sample)):
        return b
    code = code_block(strip_think(sample.get("content")))
    if not code:
        return _no_answer(sample)
    from .code_exec import run_tests
    res = run_tests(code, item["gold"]["tests"], item["gold"].get("func_name"))
    ok = res["status"] == "passed"
    detail = res.get("detail", "")
    outcome = "correct" if ok else (detail.split(":")[0] or "wrong_output")
    return {"correct": ok, "score": float(ok), "pred": None, "outcome": outcome,
            "tests_passed": res.get("passed", 0), "tests_total": res.get("total", 0)}


GRADERS = {"math": grade_math, "mcq": grade_mcq, "exact": grade_exact,
           "zebra": grade_zebra, "code": grade_code}


def grade(item, sample):
    return GRADERS[item["grader"]](item, sample)
