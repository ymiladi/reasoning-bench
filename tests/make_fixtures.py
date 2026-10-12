"""Write small offline fixture datasets (same task names as configs/suite.yaml) for dry runs.

    python tests/make_fixtures.py tests/fixtures/data
"""
import json
import sys
from pathlib import Path

from bench.tasks import GPQA_TMPL, LCB_FUNC, LCB_STDIN, MATH_SUFFIX, ZEBRA_TMPL

out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/data")
out.mkdir(parents=True, exist_ok=True)


def write(task, items):
    with open(out / f"{task}.jsonl", "w") as f:
        for it in items:
            it["task"] = task
            f.write(json.dumps(it) + "\n")


math_items = [(f"q{i}", f"What is {i} + {i * 3}?", str(i + i * 3)) for i in range(1, 21)]
write("competition_math_2026", [
    {"id": f"aime26-{i}" if n < 10 else f"hmmt26-{i}", "prompt": p + MATH_SUFFIX, "gold": g,
     "grader": "math", "meta": {"source": "aime26" if n < 10 else "hmmt26"}}
    for n, (i, p, g) in enumerate(math_items)])
write("aime_2025", [{"id": f"aime25-{i}", "prompt": p + MATH_SUFFIX, "gold": g, "grader": "math",
                     "meta": {"source": "aime25"}} for i, p, g in math_items[:10]])
write("math500", [{"id": f"m{i}", "prompt": "Simplify $\\frac{2}{4}$." + MATH_SUFFIX,
                   "gold": "\\frac{1}{2}", "grader": "math", "meta": {"level": 1 + i % 5}}
                  for i in range(15)])
write("gpqa_diamond", [{"id": f"g{i}", "prompt": GPQA_TMPL.format(question="(fixture)", A="a", B="b", C="c", D="d"),
                        "gold": "ABCD"[i % 4], "grader": "mcq",
                        "meta": {"domain": ["Physics", "Chemistry", "Biology"][i % 3], "sensitive": True}}
                       for i in range(18)])

stdin_sol = "n=int(input())\nprint(sum(map(int,input().split()))*n)"
func_starter = "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n        "
func_sol = ("class Solution:\n    def twoSum(self, nums, target):\n        seen={}\n"
            "        for i,x in enumerate(nums):\n            if target-x in seen: return [seen[target-x],i]\n"
            "            seen[x]=i\n")
lcb = []
for i in range(8):
    if i % 2 == 0:
        lcb.append({"id": f"lcb-s{i}", "prompt": LCB_STDIN.format(q="Read n then a list; print n*sum."),
                    "gold": {"tests": [{"input": "2\n1 2 3\n", "output": "12\n", "testtype": "stdin"},
                                       {"input": "1\n5\n", "output": "5", "testtype": "stdin"}],
                             "func_name": None},
                    "grader": "code", "meta": {"difficulty": ["easy", "medium", "hard"][i % 3],
                                               "mock_solution": stdin_sol}})
    else:
        lcb.append({"id": f"lcb-f{i}", "prompt": LCB_FUNC.format(q="Two sum.", starter=func_starter),
                    "gold": {"tests": [{"input": "[2, 7, 11, 15]\n9", "output": "[0, 1]", "testtype": "functional"},
                                       {"input": "[3, 2, 4]\n6", "output": "[1, 2]", "testtype": "functional"}],
                             "func_name": "twoSum"},
                    "grader": "code", "meta": {"difficulty": ["easy", "medium", "hard"][i % 3],
                                               "mock_solution": func_sol}})
write("livecodebench", lcb)

hdr = ["House", "Name", "Drink"]
rows = [["1", "Arnold", "tea"], ["2", "Eric", "milk"]]
tmpl = {f"House {i + 1}": {c: "___" for c in hdr[1:]} for i in range(2)}
write("zebralogic", [{"id": f"z{i}", "prompt": ZEBRA_TMPL.format(puzzle="(fixture puzzle)", template=json.dumps(tmpl)),
                      "gold": {"header": hdr, "rows": rows}, "grader": "zebra",
                      "meta": {"size": "2*2", "bucket": ["small", "medium"][i % 2]}} for i in range(12)])
print(f"fixtures written to {out}")
