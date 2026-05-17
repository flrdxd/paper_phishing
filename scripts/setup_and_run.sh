#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

source venv/bin/activate
pip install --quiet -r requirements.txt
pip install --quiet -e .

phishing-check-setup
phishing-test-data-quality

echo "Choose pipeline:"
echo "1) sklearn only (recommended)"
echo "2) full transformers pipeline"
read -r -p "Enter choice (1 or 2): " choice

case "$choice" in
  1)
    python -m phishing_detection.main_sklearn_only
    ;;
  2)
    pip install --quiet -r requirements-transformers.txt
    python -m phishing_detection.main
    ;;
  *)
    echo "Invalid choice" >&2
    exit 1
    ;;
esac

echo "Artifacts saved under .artifacts/"
