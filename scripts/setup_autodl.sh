#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Inherit the supplied AutoDL torch build. New Python packages live in this venv.
python - <<'PY'
import sys, torch
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.split('+')[0] == '2.1.2', torch.__version__
assert torch.version.cuda == '11.8', torch.version.cuda
assert torch.cuda.is_available(), 'CUDA unavailable'
print('Using existing', torch.__version__, torch.cuda.get_device_name())
PY
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m jev_evolve.cli preflight
python -m unittest discover -s tests -v
echo 'Ready. Run: source .venv/bin/activate'
