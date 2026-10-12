#!/usr/bin/env bash
# Grading executes model-written code (LiveCodeBench). Run it in a throwaway container with
# no network and only this repo mounted. Falls back to local grading if Docker is missing.
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v docker >/dev/null 2>&1; then
  docker build -q -t bench-grader -f scripts/Dockerfile.grader . >/dev/null
  docker run --rm --network none --memory 16g --cpus 8 --pids-limit 512 --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD":/work -w /work bench-grader python -m bench grade --workers 8
else
  echo "WARNING: Docker not found - grading locally. Model code runs with rlimits but no network isolation."
  source .venv/bin/activate 2>/dev/null || true
  python -m bench grade --workers 8
fi
