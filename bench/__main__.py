"""CLI:  python -m bench <command> [options]

  prepare   download + sample + freeze items into data/
  inspect   show item counts and one (redacted) prompt per task
  estimate  forecast token volume and GPU cost before running
  run       generate samples (resumable; re-run to continue or retry errors)
  grade     grade raw samples (re-runnable)
  report    compute statistics and write results/report.html
  all       prepare -> run -> grade -> report
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml

from . import tasks as T


def load_cfg(args):
    models = yaml.safe_load(open(args.models_config))
    suite = yaml.safe_load(open(args.suite))
    return models, suite


def pick(all_names, wanted):
    if not wanted:
        return list(all_names)
    bad = set(wanted) - set(all_names)
    if bad:
        raise SystemExit(f"unknown: {sorted(bad)}; available: {list(all_names)}")
    return wanted


def cmd_prepare(args, models, suite):
    for t in pick(suite["tasks"], args.tasks):
        cfg = suite["tasks"][t]
        try:
            path = T.prepare(t, cfg, Path(args.data))
            n = sum(1 for _ in open(path))
            print(f"prepared {t}: {n} items -> {path}")
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {t}: {type(e).__name__}: {e}")
            if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
                print("  -> gated dataset: accept its terms on huggingface.co and set HF_TOKEN")


def cmd_inspect(args, models, suite):
    for t in pick(suite["tasks"], args.tasks):
        try:
            items = T.load_items(t, Path(args.data))
        except FileNotFoundError as e:
            print(e)
            continue
        cfg = suite["tasks"][t]
        graders = {it["grader"] for it in items}
        print(f"\n== {t}: {len(items)} items, grader={graders}, samples={cfg.get('samples', 1)}")
        if cfg.get("sensitive"):
            print("   (prompt hidden: dataset terms forbid displaying examples)")
        else:
            print("   " + items[0]["prompt"][:600].replace("\n", "\n   "))


def cmd_estimate(args, models, suite):
    from .runner import samples_for
    print(f"{'model':<20}{'task':<24}{'calls':>7}{'est out tok':>14}{'est $':>10}")
    for m in [x for x in models["models"] if x["name"] in pick([y["name"] for y in models["models"]], args.models)]:
        tot_calls = tot_tok = tot_usd = 0
        for t in pick(suite["tasks"], args.tasks):
            cfg = suite["tasks"][t]
            try:
                n = len(T.load_items(t, Path(args.data)))
            except FileNotFoundError:
                known = {"MathArena/aime_2026": 30, "MathArena/aime_2025": 30, "MathArena/hmmt_feb_2026": 33}
                n = cfg.get("n_items") or (
                    sum(known.get(s["repo"], 30) for s in cfg.get("sources", [])) if cfg["loader"] == "competition_math"
                    else {"gpqa": 198}.get(cfg["loader"], 200))
            k = samples_for(m, t, cfg)
            calls, tok = n * k, n * k * cfg.get("est_out_tokens", 8000)
            c = m.get("cost", {})
            usd = tok / 1e6 * c.get("usd_per_mtok_out", 0) + calls * 400 / 1e6 * c.get("usd_per_mtok_in", 0)
            if c.get("gpu_usd_per_hour") and c.get("est_throughput_tok_s"):
                usd = tok / c["est_throughput_tok_s"] / 3600 * c["gpu_usd_per_hour"]
            tot_calls, tot_tok, tot_usd = tot_calls + calls, tot_tok + tok, tot_usd + usd
            print(f"{m['name']:<20}{t:<24}{calls:>7}{tok:>14,}{usd:>10.0f}")
        line = f"{m['name']:<20}{'TOTAL':<24}{tot_calls:>7}{tot_tok:>14,}{tot_usd:>10.0f}"
        c = m.get("cost", {})
        if c.get("est_throughput_tok_s"):
            line += (f"   (~{tot_tok / c['est_throughput_tok_s'] / 3600:.1f} GPU-box hours at an assumed "
                     f"{c['est_throughput_tok_s']:,} tok/s on {c.get('gpu_desc', 'GPU')})")
        print(line + "\n")
    print("Token figures are rough priors (est_out_tokens in suite.yaml); real usage is logged per run.")


def cmd_run(args, models, suite):
    from .runner import run_task
    mnames = pick([m["name"] for m in models["models"]], args.models)
    tnames = pick(suite["tasks"], args.tasks)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    async def main():
        for m in [x for x in models["models"] if x["name"] in mnames]:
            for t in tnames:
                await run_task(m, t, suite["tasks"][t], Path(args.data), out,
                               limit=args.limit)
    asyncio.run(main())


def cmd_grade(args, models, suite):
    from .grade import grade_all
    grade_all(Path(args.out), Path(args.data), workers=args.workers)


def cmd_report(args, models, suite):
    from .analyze import analyze
    from .report import render
    s = analyze(Path(args.out), Path(args.data), models, suite)
    path = render(Path(args.out), title=args.title)
    print("\nOverall (macro avg over tasks):")
    for m, v in sorted(s["overall"].items(), key=lambda kv: -kv[1]["macro_acc"]):
        print(f"  {m:<22}{100 * v['macro_acc']:6.1f}%  [{100 * v['ci_lo']:.1f}, {100 * v['ci_hi']:.1f}]")
    print(f"\nreport -> {path}\nsummary -> {Path(args.out) / 'summary.json'}")


def main():
    ap = argparse.ArgumentParser(prog="bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "inspect", "estimate", "run", "grade", "report", "all"])
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--suite", default="configs/suite.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="*", help="subset of model names")
    ap.add_argument("--tasks", nargs="*", help="subset of task names")
    ap.add_argument("--limit", type=int, help="only the first N items per task (smoke tests)")
    ap.add_argument("--workers", type=int, default=8, help="parallel code-grading processes")
    ap.add_argument("--title", default="Falcon-H1R-7B vs K2 Think V2: reasoning benchmark")
    args = ap.parse_args()
    models, suite = load_cfg(args)
    if args.command == "all":
        for c in (cmd_prepare, cmd_run, cmd_grade, cmd_report):
            c(args, models, suite)
    else:
        globals()[f"cmd_{args.command}"](args, models, suite)


if __name__ == "__main__":
    main()
