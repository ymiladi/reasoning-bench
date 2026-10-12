"""Grader + extractor + sandbox tests.  Run:  python -m pytest -q tests  (or python tests/test_graders.py)"""
from bench.code_exec import run_tests
from bench.extract import last_boxed, mcq_letter, strip_think
from bench.graders import grade


def S(content, finish="stop", error=None):
    return {"content": content, "finish_reason": finish, "error": error}


def M(gold):
    return {"grader": "math", "gold": gold}


def test_strip_think():
    assert strip_think("<think>x \\boxed{1}</think>\nAnswer \\boxed{2}") == "Answer \\boxed{2}"
    assert strip_think("<think>never closed \\boxed{1}") == ""
    assert strip_think("reasoning</think>final") == "final"


def test_boxed_nested():
    assert last_boxed("a \\boxed{1} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_math():
    assert grade(M("70"), S("so \\boxed{70}"))["correct"]
    assert grade(M("70"), S("so \\boxed{070}"))["correct"]
    assert not grade(M("70"), S("so \\boxed{71}"))["correct"]
    assert grade(M("\\frac{1}{2}"), S("\\boxed{0.5}"))["correct"]
    assert grade(M("\\left( 3, \\frac{\\pi}{2} \\right)"), S("\\boxed{(3, \\pi/2)}"))["correct"]
    # answer only inside the reasoning trace must NOT count
    assert grade(M("70"), S("<think>\\boxed{70}"))["outcome"] == "no_final_answer"
    assert grade(M("70"), S("<think>\\boxed{70}", finish="length"))["outcome"] == "truncated"
    assert grade(M("70"), S(None, error="RateLimit"))["outcome"] == "api_error"


def test_mcq():
    assert mcq_letter("blah\nAnswer: C") == "C"
    assert mcq_letter("The answer is (B). Wait, actually Answer: D") == "D"
    assert mcq_letter("\\boxed{A}") == "A"
    assert mcq_letter("**B**") == "B"
    assert grade({"grader": "mcq", "gold": "B"}, S("Answer: B"))["correct"]


def test_zebra():
    item = {"grader": "zebra", "gold": {"header": ["House", "Name"], "rows": [["1", "A"], ["2", "B"]]}}
    g = grade(item, S('```json\n{"solution": {"House 1": {"Name": "a"}, "House 2": {"Name": "B"}}}\n```'))
    assert g["correct"] and g["cell_acc"] == 1.0
    g = grade(item, S('{"solution": {"House 1": {"Name": "B"}, "House 2": {"Name": "B"}}}'))
    assert not g["correct"] and g["cell_acc"] == 0.5


def test_code_stdin_and_functional():
    t = [{"input": "3\n", "output": "6\n", "testtype": "stdin"}]
    assert run_tests("n=int(input())\nprint(n*2)", t)["status"] == "passed"
    assert run_tests("import sys\nprint(int(sys.stdin.read())*2)", t)["status"] == "passed"
    assert run_tests("print(int(open(0).read())*2)", t)["status"] == "passed"
    assert run_tests("import sys\nn=int(sys.stdin.buffer.read())\nprint(n*2)\nsys.exit(0)", t)["status"] == "passed"
    assert run_tests("print(5)", t)["detail"] == "wrong_output"
    assert run_tests("while True: pass", t, per_test_timeout=1)["detail"] == "timeout"
    assert run_tests("def f(:\n", t)["detail"] == "compile_error"
    assert run_tests("x=[0]*(10**10)", t, mem_mb=512)["detail"] in ("memory_error", "runtime_error:MemoryError", "crashed")
    ft = [{"input": "[1, 2]\n3", "output": "[0, 1]", "testtype": "functional"},
          {"input": "1.5", "output": "3.0", "testtype": "functional"}]
    code = ("class Solution:\n    def f(self, *a):\n        if len(a)==2: return (0, 1)\n"
            "        return a[0]*2\n")
    assert run_tests(code, ft, "f")["status"] == "passed"
    floats = [{"input": "1\n", "output": "0.3333333\n", "testtype": "stdin"}]
    assert run_tests("input()\nprint(1/3)", floats)["status"] == "passed"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
