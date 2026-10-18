from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"example_{name}", ROOT / "examples" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_guardrail_demo_runs_offline_and_shows_the_fallback(capsys):
    load("guardrail_demo").main()
    out = capsys.readouterr().out
    assert "honest stub: narrative_source = llm" in out
    assert "inventing stub: narrative_source = template_after_guardrail" in out
    assert "none: narrative_source = template" in out
    assert "2025-06-30" in out  # quoted only as an objection, never in the attached narrative
    narrative_lines = [ln for ln in out.splitlines() if ln.startswith("WO") or "likely has" in ln]
    assert narrative_lines


def test_guardrail_demo_order_is_valid():
    wo = load("guardrail_demo").example_order()
    assert wo.synthetic and wo.rul.p10_days <= wo.rul.p50_days <= wo.rul.p90_days


def test_quickstart_example_imports_cleanly():
    mod = load("quickstart")
    assert callable(mod.main)


@pytest.mark.parametrize("name", ["quickstart", "guardrail_demo"])
def test_examples_do_not_reach_for_the_network(name):
    src = (ROOT / "examples" / f"{name}.py").read_text(encoding="utf-8")
    for needle in ("requests", "urllib", "httpx", "anthropic", "socket"):
        assert needle not in src
