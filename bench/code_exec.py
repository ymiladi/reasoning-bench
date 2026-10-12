"""Run model-written Python against LiveCodeBench-style test cases.

All tests for one problem run inside ONE child process (fast), with a per-test
alarm, memory/CPU rlimits, and a hard wall-clock backstop on the whole process.

SECURITY: this executes untrusted model output. Run the grading step inside a
throwaway container/VM (see README), never on a machine with secrets mounted.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

DRIVER = r'''
import sys, json, io, signal, resource, math, os, builtins

def _limits(mem_mb):
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
    except Exception:
        pass

class _TO(Exception):
    pass

def _alarm(signum, frame):
    raise _TO()

def _norm_lines(s):
    return [ln.strip() for ln in s.strip().splitlines() if ln.strip() != ""]

def _tok_equal(a, b):
    if a == b:
        return True
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
            if not math.isclose(fx, fy, rel_tol=1e-6, abs_tol=1e-6):
                return False
        except ValueError:
            return False
    return True

def _out_equal(got, exp):
    g, e = _norm_lines(got), _norm_lines(exp)
    return len(g) == len(e) and all(_tok_equal(x, y) for x, y in zip(g, e))

def _val_equal(a, b):
    if isinstance(a, tuple):
        a = list(a)
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
        except Exception:
            return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_val_equal(x, y) for x, y in zip(a, b))
    return a == b

def main():
    spec = json.load(open(sys.argv[1]))
    code, tests, fn = spec["code"], spec["tests"], spec.get("func_name")
    per_test = spec.get("per_test_timeout", 6)
    _limits(spec.get("mem_mb", 2048))
    signal.signal(signal.SIGALRM, _alarm)
    real_stdout, real_stdin = sys.stdout, sys.stdin
    global _real_open
    _real_open = builtins.open
    result = {"status": "passed", "passed": 0, "total": len(tests), "detail": ""}

    for i, t in enumerate(tests):
        ok, detail = False, ""
        try:
            signal.alarm(per_test)
            if t.get("testtype") == "functional":
                g = {"__name__": "__sol__"}
                exec(compile("from typing import *\nfrom collections import *\nimport math, heapq, bisect, itertools, functools\n" + code, "<solution>", "exec"), g)
                args = [json.loads(line) for line in t["input"].split("\n") if line.strip() != ""]
                if "Solution" in g:
                    f = getattr(g["Solution"](), fn)
                else:
                    f = g[fn]
                got = f(*args)
                exp = json.loads(t["output"])
                ok = _val_equal(got, exp)
                if not ok:
                    detail = "wrong_output"
            else:
                inp = io.TextIOWrapper(io.BytesIO(t["input"].encode()), encoding="utf-8")
                out_buf = io.BytesIO()
                out = io.TextIOWrapper(out_buf, encoding="utf-8")
                sys.stdin, sys.stdout = inp, out
                builtins.input = lambda prompt="": (inp.readline() or (_ for _ in ()).throw(EOFError())).rstrip("\n")
                _raw = t["input"].encode()
                def _open(file, mode="r", *a, **k):
                    if file == 0:
                        b = io.BytesIO(_raw)
                        return b if "b" in mode else io.TextIOWrapper(b, encoding="utf-8")
                    return _real_open(file, mode, *a, **k)
                builtins.open = _open
                g = {"__name__": "__main__"}
                try:
                    exec(compile(code, "<solution>", "exec"), g)
                except SystemExit:
                    pass
                out.flush()
                sys.stdin, sys.stdout = real_stdin, real_stdout
                ok = _out_equal(out_buf.getvalue().decode("utf-8", "replace"), t["output"])
                if not ok:
                    detail = "wrong_output"
        except _TO:
            detail = "timeout"
        except SyntaxError:
            detail = "compile_error"
        except MemoryError:
            detail = "memory_error"
        except BaseException as e:
            detail = "runtime_error:" + type(e).__name__
        finally:
            signal.alarm(0)
            sys.stdin, sys.stdout = real_stdin, real_stdout
            builtins.open = _real_open
        if not ok:
            result.update(status="failed", detail=detail, first_failed=i)
            break
        result["passed"] += 1
    real_stdout.write("\n@@RESULT@@" + json.dumps(result) + "\n")
    real_stdout.flush()
    os._exit(0)

main()
'''


def run_tests(code: str, tests: list[dict], func_name: str | None = None,
              per_test_timeout: int = 6, mem_mb: int = 2048) -> dict:
    """Return {"status": passed|failed, "passed": n, "total": m, "detail": ...}."""
    with tempfile.TemporaryDirectory(prefix="bench_exec_") as d:
        spec = os.path.join(d, "spec.json")
        driver = os.path.join(d, "driver.py")
        with open(spec, "w") as f:
            json.dump({"code": code, "tests": tests, "func_name": func_name,
                       "per_test_timeout": per_test_timeout, "mem_mb": mem_mb}, f)
        with open(driver, "w") as f:
            f.write(DRIVER)
        backstop = per_test_timeout * max(1, len(tests)) + 10
        env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "HOME": d}
        try:
            p = subprocess.run([sys.executable, "-I", driver, spec], cwd=d, env=env,
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=backstop, text=True,
                               errors="replace")
        except subprocess.TimeoutExpired:
            return {"status": "failed", "passed": 0, "total": len(tests), "detail": "timeout"}
        marker = p.stdout.rfind("@@RESULT@@")
        if marker < 0:
            tail = (p.stderr or "")[-300:]
            return {"status": "failed", "passed": 0, "total": len(tests),
                    "detail": "crashed", "stderr": tail}
        return json.loads(p.stdout[marker + len("@@RESULT@@"):].strip().splitlines()[0])
