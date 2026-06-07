"""Mutation-testing agent — verify the tests actually catch bugs.

The adversary here mutates the *code* (flips an operator, bumps a constant) and
re-runs the tests. If the tests still pass, the mutant "survived" — meaning that
line is not really being verified. A high kill rate is evidence the green test
suite is meaningful rather than vacuous.

Self-contained (stdlib only) and safe: it copies src/ and tests/ into a temp
directory and mutates *there*, so the working tree is never touched. Each target
module is paired with the fast test files that exercise it.

    python scripts/mutation_check.py                 # default targets
    python scripts/mutation_check.py --max-mutants 20 --min-score 0.85
"""

from __future__ import annotations

import argparse
import ast
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# module (relative to src/pseudomarble) -> test files that exercise it
TARGETS = {
    "splits.py": ["test_splits.py", "test_region_splits.py", "test_mutation_killers.py"],
    "probes.py": ["test_probes.py", "test_dataset.py", "test_mutation_killers.py"],
    "materials.py": ["test_materials.py", "test_material_sampler.py"],
    "models/losses.py": ["test_losses.py"],
    "models/coherence.py": ["test_coherence.py", "test_coherence_bench.py",
                            "test_mutation_killers.py"],
}

BINOP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div,
         ast.Div: ast.Mult, ast.Mod: ast.Mult, ast.Pow: ast.Mult,
         ast.FloorDiv: ast.Mult}
CMP = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE,
       ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
BOOL = {ast.And: ast.Or, ast.Or: ast.And}


class Mutator(ast.NodeTransformer):
    """Applies the ``target``-th mutation; with target=None, just counts sites."""

    def __init__(self, target):
        self.target = target
        self.count = 0
        self.applied = None  # (lineno, label)

    def _hit(self, lineno, label):
        i = self.count
        self.count += 1
        if i == self.target:
            self.applied = (lineno, label)
            return True
        return False

    def visit_BinOp(self, node):
        self.generic_visit(node)
        t = type(node.op)
        if t in BINOP and self._hit(node.lineno, f"{t.__name__}->{BINOP[t].__name__}"):
            node.op = BINOP[t]()
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        for idx, op in enumerate(node.ops):
            t = type(op)
            if t in CMP and self._hit(node.lineno, f"{t.__name__}->{CMP[t].__name__}"):
                node.ops[idx] = CMP[t]()
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        t = type(node.op)
        if t in BOOL and self._hit(node.lineno, f"{t.__name__}->{BOOL[t].__name__}"):
            node.op = BOOL[t]()
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            if self._hit(node.lineno, f"bool {node.value}->{not node.value}"):
                node.value = not node.value
        elif isinstance(node.value, (int, float)):
            if self._hit(node.lineno, f"num {node.value}->{node.value + 1}"):
                node.value = node.value + 1
        return node


def count_sites(source: str) -> int:
    m = Mutator(None)
    m.visit(ast.parse(source))
    return m.count


def make_mutant(source: str, i: int):
    m = Mutator(i)
    tree = m.visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), m.applied


def run_tests(test_dir: Path, test_files) -> bool:
    """True if ALL given test files pass (so a failure => mutant killed)."""
    for tf in test_files:
        r = subprocess.run([sys.executable, str(test_dir / tf)],
                           capture_output=True, timeout=180)
        if r.returncode != 0:
            return False
    return True


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mutants", type=int, default=12, help="per target")
    # 0.70 is a deterministic ratchet (seeded sampling). Residual survivors are
    # mostly equivalent mutants — symmetric scaling/sign in the coherence metric
    # that pearson is invariant to, plus a few config-default constants. Raise the
    # floor over time as tests strengthen; never lower it without justification.
    ap.add_argument("--min-score", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Exclude __pycache__/*.pyc so a stale bytecode cache can never shadow the
        # mutated source in the subprocess imports.
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        shutil.copytree(ROOT / "src", tmp / "src", ignore=ignore)
        shutil.copytree(ROOT / "tests", tmp / "tests", ignore=ignore)
        src_pkg = tmp / "src" / "pseudomarble"
        test_dir = tmp / "tests"
        rng = random.Random(args.seed)

        total_killed = total = 0
        print(f"mutation check (max {args.max_mutants} mutants/target, "
              f"min score {args.min_score})\n" + "-" * 70, flush=True)

        for mod, tests in TARGETS.items():
            path = src_pkg / mod
            original = path.read_text()
            # Sanity: the unmutated module must pass its tests.
            if not run_tests(test_dir, tests):
                print(f"{mod}: BASELINE FAILED — skipping", flush=True)
                continue

            n = count_sites(original)
            idxs = list(range(n))
            rng.shuffle(idxs)
            idxs = sorted(idxs[:args.max_mutants])

            killed, survivors = 0, []
            for i in idxs:
                mutated, applied = make_mutant(original, i)
                path.write_text(mutated)
                try:
                    passed = run_tests(test_dir, tests)
                finally:
                    path.write_text(original)  # always restore
                if passed:
                    survivors.append(applied)
                else:
                    killed += 1

            run = len(idxs)
            total_killed += killed
            total += run
            score = killed / run if run else 1.0
            print(f"{mod:24s} {killed:2d}/{run:2d} killed  (score {score:.2f})",
                  flush=True)
            for lineno, label in survivors:
                print(f"      survived: line {lineno}  [{label}]", flush=True)

        overall = total_killed / total if total else 1.0
        print("-" * 70)
        print(f"OVERALL: {total_killed}/{total} mutants killed — "
              f"mutation score {overall:.2f}")
        if overall < args.min_score:
            print(f"FAIL: below threshold {args.min_score}")
            return 1
        print("PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
