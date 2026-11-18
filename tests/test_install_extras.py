"""The install extras keep their promises.

The core installs with no dependencies (CLAUDE.md: pure-Python core, lazy heavy
deps). Grading a preregistered run needs scipy (crossview_verdicts' Welch t and
NULL bound, F36), so an extra must provide it, and `dev` must too, or the rule
tests that need it skip silently.
"""

from __future__ import annotations

import os

import pytest

tomllib = pytest.importorskip("tomllib")  # stdlib from Python 3.11

PYPROJECT = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")


def _project():
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)["project"]


def _names(reqs):
    out = set()
    for r in reqs:
        name = r
        for sep in "<>=!~[; ":
            name = name.split(sep)[0]
        out.add(name.strip().lower())
    return out


def test_core_has_no_required_dependencies():
    assert _project()["dependencies"] == []


def test_analysis_and_dev_extras_provide_scipy():
    extras = _project()["optional-dependencies"]
    assert "scipy" in _names(extras["analysis"])
    assert "scipy" in _names(extras["dev"])
