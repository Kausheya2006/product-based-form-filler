#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_e2e_headed.sh
# Optional env:
#   APP_BASE_URL=http://127.0.0.1:8000
#   E2E_TC_EXPR="tc09 or tc10"

APP_BASE_URL="${APP_BASE_URL:-http://127.0.0.1:8000}"
E2E_TC_EXPR="${E2E_TC_EXPR:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_SRC_DIR}"
export PYTHONPATH="${PROJECT_SRC_DIR}:${PYTHONPATH:-}"

EXTRA_PYTEST_ARGS=()
if [[ -n "${E2E_TC_EXPR}" ]]; then
  EXTRA_PYTEST_ARGS=(-k "${E2E_TC_EXPR}")
fi

python -m playwright install chromium
RUN_E2E=1 APP_BASE_URL="$APP_BASE_URL" \
python -m pytest tests/e2e -m e2e "${EXTRA_PYTEST_ARGS[@]-}" \
  --headed --browser chromium \
  --html=reports/e2e_headed_report.html --self-contained-html \
  --tracing=retain-on-failure --video=retain-on-failure --screenshot=only-on-failure
