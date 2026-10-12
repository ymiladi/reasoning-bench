"""Render results/summary.json into a single self-contained HTML report (no external assets)."""
from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

PALETTE = ["#2f6fdb", "#d9822b", "#1f9e74", "#a35bc4", "#c94f4f"]


def esc(x):
    return html.escape(str(x)) if x is not None else "—"


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def num(x, d=0):
    return "—" if x is None else f"{x:,.{d}f}"


def bar_chart(groups, models, colors, values, cis=None, width=760, row_h=22, label_w=190):
    """Horizontal grouped bar chart, 0-100%, optional CI whiskers. values[g][m] in [0,1]."""
    plot_w = width - label_w - 60
    group_h = row_h * len(models) + 14
    h = group_h * len(groups) + 30
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img">']
    for tick in range(0, 101, 20):
        x = label_w + plot_w * tick / 100
        out.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{h - 22}" class="grid"/>'
                   f'<text x="{x}" y="{h - 6}" class="tick" text-anchor="middle">{tick}%</text>')
    for gi, g in enumerate(groups):
        y0 = gi * group_h + 4
        out.append(f'<text x="{label_w - 10}" y="{y0 + group_h / 2}" class="glabel" '
                   f'text-anchor="end" dominant-baseline="middle">{esc(g[1])}</text>')
        for mi, m in enumerate(models):
            v = values.get(g[0], {}).get(m)
            if v is None:
                continue
            y = y0 + mi * row_h
            w = plot_w * v
            out.append(f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{row_h - 6}" rx="3" '
                       f'fill="{colors[m]}"><title>{esc(m)}: {pct(v)}</title></rect>')
            if cis and cis.get(g[0], {}).get(m):
                lo, hi = cis[g[0]][m]
                xl, xh = label_w + plot_w * lo, label_w + plot_w * hi
                cy = y + (row_h - 6) / 2
                out.append(f'<line x1="{xl:.1f}" y1="{cy}" x2="{xh:.1f}" y2="{cy}" class="ci"/>'
                           f'<line x1="{xl:.1f}" y1="{cy - 4}" x2="{xl:.1f}" y2="{cy + 4}" class="ci"/>'
                           f'<line x1="{xh:.1f}" y1="{cy - 4}" x2="{xh:.1f}" y2="{cy + 4}" class="ci"/>')
            out.append(f'<text x="{label_w + max(w, plot_w * (cis[g[0]][m][1] if cis and cis.get(g[0], {}).get(m) else v)) + 6:.1f}" '
                       f'y="{y + (row_h - 6) / 2}" class="val" dominant-baseline="middle">{pct(v)}</text>')
    out.append("</svg>")
    return "".join(out)


def render(out_dir: Path, title="Reasoning benchmark"):
    s = json.loads((out_dir / "summary.json").read_text())
    models, tasks = s["models"], s["tasks"]
    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
    mlabel = {m: s["model_info"][m]["label"] for m in models}
    tlabel = {t: s["task_info"][t]["label"] for t in tasks}

    legend = "".join(f'<span class="lg"><i style="background:{colors[m]}"></i>{esc(mlabel[m])}</span>'
                     for m in models)

    # overall
    overall_vals = {"overall": {m: s["overall"][m]["macro_acc"] for m in models if m in s["overall"]}}
    overall_cis = {"overall": {m: (s["overall"][m]["ci_lo"], s["overall"][m]["ci_hi"])
                               for m in models if m in s["overall"]}}
    overall_svg = bar_chart([("overall", "Macro average")], models, colors, overall_vals, overall_cis)

    task_vals = {t: {m: v["acc"] for m, v in s["per_task"][t].items()} for t in tasks}
    task_cis = {t: {m: (v["ci_lo"], v["ci_hi"]) for m, v in s["per_task"][t].items()} for t in tasks}
    task_svg = bar_chart([(t, tlabel[t]) for t in tasks], models, colors, task_vals, task_cis)

    # per-task table
    rows = []
    for t in tasks:
        first = True
        for m in models:
            e = s["per_task"][t].get(m)
            if not e:
                continue
            extra = f' <span class="muted">(cells {pct(e["cell_acc"])})</span>' if "cell_acc" in e else ""
            rows.append(
                f"<tr>{'<td rowspan=%d class=tname>%s<div class=muted>%d items × %s</div></td>' % (len(s['per_task'][t]), esc(tlabel[t]), e['n_items'], e['k']) if first else ''}"
                f'<td><i class="dot" style="background:{colors[m]}"></i>{esc(mlabel[m])}</td>'
                f'<td class="r strong">{pct(e["acc"])}{extra}</td>'
                f'<td class="r muted">{pct(e["ci_lo"])} – {pct(e["ci_hi"])}</td>'
                f'<td class="r">{pct(e["maj_at_k"]) if e.get("maj_at_k") is not None else "—"}</td>'
                f'<td class="r">{pct(e["pass_at_k"])}</td>'
                f'<td class="r">{num(e["mean_out_tokens"])}</td>'
                f'<td class="r">{num(e["out_tokens_per_correct"])}</td>'
                f'<td class="r">{pct(e["truncation_rate"])}</td>'
                f'<td class="r">{pct(e["error_rate"])}</td></tr>')
            first = False
    task_table = "".join(rows)

    # pairwise
    prow = []
    for key, per in s["pairwise"].items():
        a, b = key.split("__vs__")
        for t, r in per.items():
            sig = r["p_value"] < 0.05
            winner = (mlabel[a] if r["diff"] > 0 else mlabel[b]) if sig else "no clear winner"
            ag = s["agreement"][key][t]
            prow.append(
                f"<tr><td>{esc(mlabel[a])} vs {esc(mlabel[b])}</td><td>{esc(tlabel[t])}</td>"
                f'<td class="r">{100 * r["diff"]:+.1f} pts</td>'
                f'<td class="r muted">{100 * r["ci_lo"]:+.1f} to {100 * r["ci_hi"]:+.1f}</td>'
                f'<td class="r">{r["p_value"]:.3f}</td>'
                f'<td class="{"strong" if sig else "muted"}">{esc(winner)}</td>'
                f'<td class="r muted">{ag["only_a"]} / {ag["only_b"]} / {ag["both"]} / {ag["neither"]}</td></tr>')
    pair_table = "".join(prow)

    # outcomes stacked
    outcome_order = ["correct", "wrong_answer", "wrong_output", "no_final_answer", "truncated",
                     "compile_error", "runtime_error", "timeout", "memory_error", "crashed", "api_error"]
    ocolors = {"correct": "#1f9e74", "wrong_answer": "#c94f4f", "wrong_output": "#c94f4f",
               "no_final_answer": "#e0a32e", "truncated": "#d9822b", "compile_error": "#8d6bd1",
               "runtime_error": "#a35bc4", "timeout": "#6b7a90", "memory_error": "#6b7a90",
               "crashed": "#6b7a90", "api_error": "#999"}
    orow = []
    for t in tasks:
        for m in models:
            e = s["per_task"][t].get(m)
            if not e:
                continue
            tot = sum(e["outcomes"].values())
            segs = []
            keys = outcome_order + [k for k in e["outcomes"] if k not in outcome_order]
            for k in keys:
                n = e["outcomes"].get(k, 0)
                if n:
                    segs.append(f'<span style="width:{100 * n / tot:.2f}%;background:{ocolors.get(k, "#777")}" '
                                f'title="{esc(k)}: {n} ({100 * n / tot:.1f}%)"></span>')
            orow.append(f'<div class="orow"><div class="olabel">{esc(tlabel[t])} · {esc(mlabel[m])}</div>'
                        f'<div class="ostack">{"".join(segs)}</div></div>')
    olegend = "".join(f'<span class="lg"><i style="background:{ocolors[k]}"></i>{k.replace("_", " ")}</span>'
                      for k in ["correct", "wrong_answer", "no_final_answer", "truncated", "compile_error",
                                "runtime_error", "timeout", "api_error"])

    # cost
    crow = []
    for m in models:
        c = s["cost"].get(m, {})
        crow.append(f'<tr><td><i class="dot" style="background:{colors[m]}"></i>{esc(mlabel[m])}</td>'
                    f'<td class="r">{num(c.get("output_tokens"))}</td>'
                    f'<td class="r strong">{"$" + num(c["usd_estimated"], 2) if c.get("usd_estimated") is not None else "—"}</td>'
                    f'<td class="r">{"$" + num(c["usd_per_100_correct"], 2) if c.get("usd_per_100_correct") else "—"}</td>'
                    f'<td class="muted">{esc(c.get("basis", "not configured"))}</td></tr>')

    # breakdowns
    brk = []
    for t, keys in s["breakdowns"].items():
        for key, per_model in keys.items():
            groups = sorted({g for v in per_model.values() for g in v})
            vals = {g: {m: per_model[m][g]["acc"] for m in per_model if g in per_model[m]} for g in groups}
            ns = {g: max(per_model[m][g]["n"] for m in per_model if g in per_model[m]) for g in groups}
            brk.append(f'<h3>{esc(tlabel[t])} by {esc(key)}</h3>' +
                       bar_chart([(g, f"{g} (n={ns[g]})") for g in groups], models, colors, vals))

    # examples
    ex = []
    for t, per in s["examples"].items():
        for m, lst in per.items():
            for e in lst:
                ex.append(f'<details><summary>{esc(tlabel[t])} · {esc(mlabel[m])} · {esc(e["item_id"])} · '
                          f'<b>{esc(e["outcome"])}</b> (pred {esc(e.get("pred"))}, gold {esc(e.get("gold"))})</summary>'
                          f'<pre>{esc(e["tail"])}</pre></details>')

    notes = "".join(f'<li><b>{esc(mlabel[m])}</b>: {esc(s["model_info"][m]["notes"])}</li>'
                    for m in models if s["model_info"][m]["notes"])

    page = TEMPLATE.format(
        title=esc(title), date=dt.date.today().isoformat(), legend=legend, overall_svg=overall_svg,
        task_svg=task_svg, task_table=task_table, pair_table=pair_table or
        '<tr><td colspan=7 class="muted">Needs at least two models.</td></tr>',
        outcomes="".join(orow), olegend=olegend, cost_table="".join(crow),
        breakdowns="".join(brk) or '<p class="muted">No breakdowns configured.</p>',
        examples="".join(ex) or '<p class="muted">None.</p>', notes=notes,
        task_desc="".join(f"<li><b>{esc(tlabel[t])}</b>: {esc(s['task_info'][t]['description'])}</li>"
                          for t in tasks))
    path = out_dir / "report.html"
    path.write_text(page)
    return path


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6f6d68;--line:#e4e1da;--card:#fff;--grid:#ece9e2}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ecebe7;--muted:#9c9a94;--line:#2c2b29;--card:#1d1c1b;--grid:#2a2927}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:32px 16px 80px}}
h1{{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}} h2{{font-size:19px;margin:40px 0 12px}} h3{{font-size:15px;margin:22px 0 6px}}
.sub{{color:var(--muted);margin:0 0 18px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;overflow-x:auto}}
.lg{{display:inline-flex;align-items:center;gap:6px;margin-right:16px;font-size:13px}} .lg i,.dot{{display:inline-block;width:10px;height:10px;border-radius:3px}}
.dot{{margin-right:7px}} table{{border-collapse:collapse;width:100%;font-size:13.5px}} th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.03em}} .r{{text-align:right;font-variant-numeric:tabular-nums}}
.muted{{color:var(--muted)}} .strong{{font-weight:650}} .tname{{font-weight:600}}
svg.chart{{width:100%;height:auto;display:block}} .grid{{stroke:var(--grid)}} .tick,.val{{fill:var(--muted);font-size:11px}} .glabel{{fill:var(--fg);font-size:12.5px}}
.ci{{stroke:var(--fg);stroke-width:1.2;opacity:.55}}
.orow{{display:flex;align-items:center;gap:10px;margin:5px 0}} .olabel{{width:260px;font-size:12.5px;flex:none}}
.ostack{{flex:1;display:flex;height:14px;border-radius:4px;overflow:hidden;background:var(--grid)}}
details{{border-bottom:1px solid var(--line);padding:6px 0}} summary{{cursor:pointer;font-size:13px}}
pre{{white-space:pre-wrap;font-size:12px;background:var(--bg);padding:10px;border-radius:6px;max-height:260px;overflow:auto}}
ul{{padding-left:20px}} li{{margin:3px 0}}
@media (max-width:640px){{.olabel{{width:120px}}}}
</style></head><body><main>
<h1>{title}</h1><p class="sub">Generated {date}. Accuracy is pass@1 averaged over k samples per item; brackets are 95% bootstrap CIs over items.</p>
<div>{legend}</div>
<h2>Overall (macro average across tasks)</h2><div class="card">{overall_svg}</div>
<h2>Accuracy by task</h2><div class="card">{task_svg}</div>
<div class="card" style="margin-top:14px"><table><thead><tr><th>Task</th><th>Model</th><th class="r">pass@1</th><th class="r">95% CI</th>
<th class="r">maj@k</th><th class="r">pass@k</th><th class="r">mean out tok</th><th class="r">tok / correct</th><th class="r">truncated</th><th class="r">API err</th></tr></thead><tbody>{task_table}</tbody></table></div>
<h2>Head-to-head (paired over shared items)</h2>
<div class="card"><table><thead><tr><th>Pair</th><th>Task</th><th class="r">Δ (A−B)</th><th class="r">95% CI</th><th class="r">p</th><th>Verdict (p&lt;0.05)</th><th class="r">only A / only B / both / neither</th></tr></thead><tbody>{pair_table}</tbody></table></div>
<h2>Where the answers go wrong</h2><div>{olegend}</div><div class="card" style="margin-top:8px">{outcomes}</div>
<h2>Cost</h2><div class="card"><table><thead><tr><th>Model</th><th class="r">output tokens</th><th class="r">est. cost</th><th class="r">per 100 correct</th><th>basis</th></tr></thead><tbody>{cost_table}</tbody></table></div>
<h2>Breakdowns</h2><div class="card">{breakdowns}</div>
<h2>Sample failures</h2><p class="sub">Final-answer tails only. GPQA items are never shown, per its terms.</p><div class="card">{examples}</div>
<h2>Setup</h2><div class="card"><ul>{task_desc}</ul><ul>{notes}</ul></div>
</main></body></html>"""
