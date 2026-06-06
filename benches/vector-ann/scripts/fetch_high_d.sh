#!/usr/bin/env bash
#
# One-shot dataset preparation for the high-dimension bench tiers.
# Reads $DATASET_ROOT (same variable the workflow uses) and writes to
# $DATASET_ROOT/high-d/<d>/. Idempotent: each tier checks for an
# existing triplet and skips if present.
#
# Tier sources:
#
#   1536D: ann-benchmarks dbpedia-openai-1000k-angular (1M x 1536,
#          OpenAI text-embedding-ada-002 over DBpedia entities, the
#          canonical public 1536D evaluation corpus).
#
#   3072D: no widely-published evaluation corpus exists today, so we
#          synthesize a 100k x 3072 angular dataset deterministically
#          (seeded N(0,1), L2-normalised, brute-force top-100 truth).
#          Switch to a real corpus by setting a `--hdf5-url` override
#          if Stella / VDBBench publishes one.
#
#   4096D / 8192D: same synthetic path. Useful as a stress probe on
#          the distance kernel + cache-footprint side; switch out for
#          real corpora once available.
#
# Usage:
#
#   DATASET_ROOT=/srv/coordinode-bench/datasets ./fetch_high_d.sh 1536
#   DATASET_ROOT=/srv/coordinode-bench/datasets ./fetch_high_d.sh all
#
# Disk usage at the four tiers combined: ~30 GB. Each synthetic tier
# downloads nothing; the 1536D tier pulls ~6 GB once.

set -euo pipefail

if [ -z "${DATASET_ROOT:-}" ]; then
  echo "DATASET_ROOT not set; aborting." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HIGH_D_ROOT="$DATASET_ROOT/high-d"
mkdir -p "$HIGH_D_ROOT"

prepare_1536() {
  python3 "$SCRIPT_DIR/prepare_high_d.py" \
    --d        1536 \
    --output   "$HIGH_D_ROOT/1536" \
    --prefix   dbpedia-openai-1M-1536 \
    --hdf5-url https://ann-benchmarks.com/dbpedia-openai-1000k-angular.hdf5
}

prepare_3072() {
  python3 "$SCRIPT_DIR/prepare_high_d.py" \
    --d       3072 \
    --output  "$HIGH_D_ROOT/3072" \
    --prefix  synthetic-3072 \
    --n-train 100000 \
    --n-test  1000
}

prepare_4096() {
  python3 "$SCRIPT_DIR/prepare_high_d.py" \
    --d       4096 \
    --output  "$HIGH_D_ROOT/4096" \
    --prefix  synthetic-4096 \
    --n-train 100000 \
    --n-test  1000
}

prepare_8192() {
  python3 "$SCRIPT_DIR/prepare_high_d.py" \
    --d       8192 \
    --output  "$HIGH_D_ROOT/8192" \
    --prefix  synthetic-8192 \
    --n-train 50000 \
    --n-test  1000
}

case "${1:-all}" in
  1536) prepare_1536 ;;
  3072) prepare_3072 ;;
  4096) prepare_4096 ;;
  8192) prepare_8192 ;;
  all)
    prepare_1536
    prepare_3072
    prepare_4096
    prepare_8192
    ;;
  *)
    echo "usage: $0 [1536|3072|4096|8192|all]" >&2
    exit 1
    ;;
esac

echo "high-d datasets ready under $HIGH_D_ROOT"
