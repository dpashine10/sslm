#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train --config configs/tiny.yaml --dry-run
python -m pytest -q
