.DEFAULT_GOAL := test

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -m src.training.train --config configs/tiny.yaml --dry-run

format-check:
	$(PYTHON) -m compileall -q src tests
