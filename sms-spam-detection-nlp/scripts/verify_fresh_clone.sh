#!/usr/bin/env bash
# Usage: bash scripts/verify_fresh_clone.sh <repo-url> [tag]   (run from inside your working copy)
# Clones the repo fresh, installs it, runs the pipeline and compares the output hash with your local file.
set -euo pipefail
REPO="${1:?give the repository URL}"; TAG="${2:-assignment-1}"
LOCAL_HASH=$(sha256sum data/processed/sms_spam_clean.csv | cut -d' ' -f1)
TMP=$(mktemp -d); echo "Cloning into $TMP ..."
git clone --branch "$TAG" "$REPO" "$TMP/repo" && cd "$TMP/repo"
python -m venv .venv && source .venv/bin/activate
pip install -q -r requirements.txt && python -m src.setup_nltk >/dev/null
python -m src.run_pipeline > run.log
FRESH_HASH=$(sha256sum data/processed/sms_spam_clean.csv | cut -d' ' -f1)
echo "local : $LOCAL_HASH"; echo "fresh : $FRESH_HASH"
[ "$LOCAL_HASH" = "$FRESH_HASH" ] && echo "REPRODUCIBLE: identical output" || { echo "MISMATCH - investigate"; exit 1; }
