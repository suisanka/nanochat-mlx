#!/bin/bash
# Configuration inspection only. No downloads, model loading or training.
set -euo pipefail
python -m scripts.train --depth=12 --dry-run
