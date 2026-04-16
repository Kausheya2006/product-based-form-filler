#!/usr/bin/env bash
set -euo pipefail

# Run only CSV R2 pytest interface/API cases (Sl no: 9,10,14-26).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_SRC_DIR}"
export PYTHONPATH="${PROJECT_SRC_DIR}:${PYTHONPATH:-}"

R2_TC_EXPR="tc09 or tc10 or tc14 or tc15 or tc16 or tc17 or tc18 or tc19 or tc20 or tc21 or tc22 or tc23 or tc24 or tc25 or tc26"

python -m pytest tests/test_interface_flows.py -k "$R2_TC_EXPR" -q
