#!/usr/bin/env bash
# Smoke test, then the full run with both models in parallel. Resumable: just re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

echo ">> Smoke test: 2 items per task per model into results_smoke/"
python -m bench run --limit 2 --out results_smoke
python -m bench grade --out results_smoke --workers 4
python -m bench report --out results_smoke --title "Smoke test"
echo ">> Check results_smoke/report.html: no API errors, answers being extracted, nothing at 0% by accident."
read -r -p "Continue with the full run? [y/N] " ok
[[ "$ok" == "y" ]] || exit 0

python -m bench run --models falcon-h1r-7b > logs/run_falcon.log 2>&1 &
python -m bench run --models k2-think-v2   > logs/run_k2.log 2>&1 &
echo ">> Running (tail -f logs/run_*.log). Waiting for both..."
wait

echo ">> Grading (code is executed in a network-less container if Docker is available)"
bash scripts/grade_sandboxed.sh
python -m bench report
echo ">> Done: results/report.html and results/summary.json"
