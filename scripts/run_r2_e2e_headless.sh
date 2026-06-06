#!/usr/bin/env bash
set -euo pipefail

# Run CSV R2 browser flows in headless mode.
APP_BASE_URL="${APP_BASE_URL:-http://127.0.0.1:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_SRC_DIR}"
export PYTHONPATH="${PROJECT_SRC_DIR}:${PYTHONPATH:-}"

python -m playwright install chromium
R2_E2E_TC_EXPR="tc09 or tc10 or tc13 or tc21 or tc25 or tc26"
RUN_E2E=1 APP_BASE_URL="$APP_BASE_URL" \
python -m pytest tests/e2e/test_ui_flows.py -m e2e -k "$R2_E2E_TC_EXPR" \
  --browser chromium \
  --html=reports/e2e_r2_headless_report.html --self-contained-html \
  --tracing=retain-on-failure --video=retain-on-failure --screenshot=only-on-failure
