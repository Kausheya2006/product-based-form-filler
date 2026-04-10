#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_e2e_headless.sh
# Optional env:
#   APP_BASE_URL=http://127.0.0.1:8000

APP_BASE_URL="${APP_BASE_URL:-http://127.0.0.1:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_SRC_DIR}"
export PYTHONPATH="${PROJECT_SRC_DIR}:${PYTHONPATH:-}"

python -m playwright install chromium
RUN_E2E=1 APP_BASE_URL="$APP_BASE_URL" \
python -m pytest tests/e2e -m e2e \
  --browser chromium \
  --html=reports/e2e_headless_report.html --self-contained-html \
  --tracing=retain-on-failure --video=retain-on-failure --screenshot=only-on-failure
