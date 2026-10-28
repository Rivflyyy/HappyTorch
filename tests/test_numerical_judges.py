"""Regression tests for the numerical checks in the practice judge.

Run from the repository root: python -m unittest discover -s tests -v
Only imports and definitions are loaded from solutions: notebook examples and
check() calls are never executed, so this does not change practice progress.
"""

import ast
import json
from pathlib import Path
import unittest

from torch_judge.tasks import get_task


ROOT = Path(__file__).resolve().parents[1]
SOLUTIONS = {
    "gqa": "10_gqa_solution.ipynb",
    "gpt2_block": "13_gpt2_block_solution.ipynb",
    "dora": "18_dora_solution.ipynb",
}


def load_solution(task_id, replacements=()):
    path = ROOT / "solutions" / SOLUTIONS[task_id]
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    # Mutate a known solution to reproduce a particular learner mistake. Fail
    # loudly if the solution changes rather than accidentally testing no mutation.
    for before, after in replacements:
        if source.count(before) != 1:
            raise AssertionError(f"Expected one mutation target: {before!r}")
        source = source.replace(before, after)
    tree = ast.parse(source, filename=str(path))
    tree.body = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))
    ]
    namespace = {}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[get_task(task_id)["function_name"]]


def judge_failures(task_id, implementation):
    task = get_task(task_id)
    failures = []
    for test in task["tests"]:
        namespace = {task["function_name"]: implementation}
        code = test["code"].replace("{fn}", task["function_name"])
        try:
            exec(compile(code, test["name"], "exec"), namespace)
        except AssertionError:
            failures.append(test["name"])
        # Unexpected errors must fail this regression suite, not count as a
        # successful rejection of a numerically incorrect implementation.
    return failures


class NumericalJudgeTests(unittest.TestCase):
    def test_published_solutions_are_accepted(self):
        for task_id in SOLUTIONS:
            with self.subTest(task=task_id):
                self.assertEqual(judge_failures(task_id, load_solution(task_id)), [])

    def test_gqa_rejects_interleaved_groups(self):
        broken = load_solution("gqa", [
            ("k.repeat_interleave(repeats, dim=1)", "k.repeat(1, repeats, 1, 1)"),
            ("v.repeat_interleave(repeats, dim=1)", "v.repeat(1, repeats, 1, 1)"),
        ])
        self.assertIn("KV heads are shared correctly", judge_failures("gqa", broken))

    def test_gqa_rejects_missing_output_projection(self):
        broken = load_solution("gqa", [("return self.W_o(out)", "return out")])
        self.assertIn("KV heads are shared correctly", judge_failures("gqa", broken))

    def test_gpt2_rejects_relu_instead_of_gelu(self):
        broken = load_solution("gpt2_block", [("nn.GELU()", "nn.ReLU()")])
        self.assertIn("Numerical match to pre-norm causal block", judge_failures("gpt2_block", broken))

    def test_gpt2_rejects_normalized_residual(self):
        broken = load_solution("gpt2_block", [
            ("x = x + self._attn(self.ln1(x))", "x = self.ln1(x) + self._attn(self.ln1(x))"),
        ])
        self.assertIn("Numerical match to pre-norm causal block", judge_failures("gpt2_block", broken))

    def test_dora_rejects_missing_update_scaling(self):
        broken = load_solution("dora", [
            ("self.B @ self.A * (self.alpha / self.rank)", "self.B @ self.A"),
        ])
        self.assertIn("Numerical match with non-zero low-rank update", judge_failures("dora", broken))

    def test_dora_rejects_normalizing_before_update(self):
        broken = load_solution("dora", [
            ("W_eff.norm(dim=1, keepdim=True)", "V.norm(dim=1, keepdim=True)"),
        ])
        self.assertIn("Numerical match with non-zero low-rank update", judge_failures("dora", broken))


if __name__ == "__main__":
    unittest.main()
