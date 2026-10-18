.PHONY: install test lint bench bench-quick demo clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -q

lint:
	ruff check .

# Full benchmark: regenerates benchmarks/results.json and benchmarks/RESULTS.md
# (several minutes; calibrates on development seeds, evaluates on held-out seeds).
bench:
	python benchmarks/run.py

# Reduced benchmark (fewer seeds) for a smoke check; writes nothing under benchmarks/.
bench-quick:
	python benchmarks/run.py --quick

# End to end on the default seed: simulate -> check -> detect -> rul -> HTML report.
demo:
	asset-health simulate --seed 42 --out out/demo
	asset-health check out/demo
	asset-health detect out/demo --as-of 2025-07-28
	asset-health rul out/demo --as-of 2025-07-28
	asset-health report out/demo --as-of 2025-07-28 --out out/report.html

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info .ruff_cache out
