"""The README is part of the contract: structure, honest labelling, and numbers that trace to the
benchmark output."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from asset_health.config import PolicyConfig
from asset_health.evaluate.markdown import results_to_markdown

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
RESULTS = ROOT / "benchmarks" / "results.json"


def test_exactly_one_h1_and_four_badges_ci_first():
    h1 = [ln for ln in README.splitlines() if re.match(r"^# [^#]", ln)]
    assert len(h1) == 1
    badges = re.findall(r"^\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", README, flags=re.M)
    assert len(badges) == 4
    assert "actions/workflows/tests.yml" in badges[0]


@pytest.mark.parametrize(
    "heading",
    [
        "Quick start",
        "How it works",
        "Results",
        "Design decisions",
        "Limitations",
        "Project layout",
        "Tests",
    ],
)
def test_required_sections_exist(heading):
    assert re.search(rf"^## {re.escape(heading)}\s*$", README, flags=re.M)


def test_why_this_exists_and_industrial_section_exist():
    assert "**Why this exists.**" in README
    assert re.search(r"^## For industrial teams", README, flags=re.M)


def test_readme_labels_the_data_synthetic_and_does_not_claim_real_data():
    assert "synthetic" in README.lower()
    assert "none of it is measured data" in README
    for banned in ("production-grade", "blazingly", "state-of-the-art"):
        assert banned not in README.lower()


def test_opc_ua_and_cmms_are_documented_but_not_claimed():
    low = README.lower()
    assert "no opc ua or historian adapter ships here" in low
    assert "no cmms connector ships" in low


def test_has_a_mermaid_diagram_and_the_author_footer():
    assert "```mermaid" in README
    assert README.rstrip().endswith(
        "Built by [Darrshan Govender](https://github.com/darrshangovender) · "
        "[Agulhas Code](https://agulhascode.co.za)"
    )


def test_local_links_resolve():
    for target in re.findall(r"\]\(((?!https?://|#)[^)\s]+)\)", README):
        assert (ROOT / target.split("#")[0]).exists(), f"broken link: {target}"


@pytest.mark.skipif(not RESULTS.exists(), reason="benchmark not run yet")
def test_results_block_is_exactly_what_the_benchmark_script_produces():
    block = README.split("<!-- RESULTS:BEGIN -->")[1].split("<!-- RESULTS:END -->")[0].strip()
    expected = results_to_markdown(json.loads(RESULTS.read_text(encoding="utf-8"))).strip()
    assert block == expected, "README results drifted from benchmarks/results.json; re-render them"


@pytest.mark.skipif(not RESULTS.exists(), reason="benchmark not run yet")
def test_results_md_matches_results_json():
    md = (ROOT / "benchmarks" / "RESULTS.md").read_text(encoding="utf-8")
    expected = results_to_markdown(json.loads(RESULTS.read_text(encoding="utf-8")))
    assert expected.strip() in md


@pytest.mark.skipif(not RESULTS.exists(), reason="benchmark not run yet")
def test_shipped_policy_defaults_are_the_tuned_policy_that_was_benchmarked():
    tuned = json.loads(RESULTS.read_text(encoding="utf-8"))["tuned_policy"]
    assert PolicyConfig().model_dump() == tuned


@pytest.mark.skipif(not RESULTS.exists(), reason="benchmark not run yet")
def test_benchmark_seeds_in_the_results_are_disjoint():
    seeds = json.loads(RESULTS.read_text(encoding="utf-8"))["seeds"]
    a, b, c = (set(seeds[k]) for k in ("dev", "cal", "test"))
    assert not (a & b or a & c or b & c)
    assert json.loads(RESULTS.read_text(encoding="utf-8"))["synthetic"] is True
