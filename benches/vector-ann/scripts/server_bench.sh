#!/usr/bin/env bash
# Server-mode vector bench: measure the full product path
# (gRPC client -> Cypher -> planner -> HnswScan -> engine) against a
# real coordinode process, as opposed to the in-process HnswIndex
# numbers from bench-vector-ann. The delta between the two prices the
# server overlay (client serialisation, HTTP/2, parse/plan, result
# encode).
#
# Usage (from the repo root):
#   DATASET_ROOT=/path/to/datasets ./benches/vector-ann/scripts/server_bench.sh
#
# Env knobs:
#   DATASET_ROOT  required; expects $DATASET_ROOT/sift/sift_{base,query}.fvecs
#   SUBSET        train subset size (default 50000, matches the
#                 in-process per-commit cells for apples-to-apples)
#   PORT          gRPC port for the throwaway server (default 7190)
#   OUTPUT        bench-results output dir (default bench-results)
#
# The script owns the server process it starts and kills ONLY that pid.

set -euo pipefail

SUBSET="${SUBSET:-50000}"
PORT="${PORT:-7190}"
OPS_PORT=$((PORT + 4))
OUTPUT="${OUTPUT:-bench-results}"

if [ -z "${DATASET_ROOT:-}" ]; then
  echo "DATASET_ROOT is required" >&2
  exit 1
fi
SIFT_DIR="$DATASET_ROOT/sift"
for f in sift_base.fvecs sift_query.fvecs; do
  test -f "$SIFT_DIR/$f" || { echo "missing $SIFT_DIR/$f" >&2; exit 1; }
done

echo "::group::build server + bench bins"
cargo build --release -p coordinode-server
( cd benches/vector-ann && cargo build --release )
echo "::endgroup::"

WORK_DIR="$(mktemp -d -t cn-server-bench-XXXXXX)"
DATA_DIR="$WORK_DIR/data"
GT_FILE="$WORK_DIR/subset_gt.ivecs"
SERVER_LOG="$WORK_DIR/server.log"
mkdir -p "$DATA_DIR"

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "::group::start server (port $PORT, data $DATA_DIR)"
./target/release/coordinode serve \
  --addr "127.0.0.1:$PORT" \
  --ops-addr "127.0.0.1:$OPS_PORT" \
  --data "$DATA_DIR" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 30); do
  if grep -q 'gRPC server listening' "$SERVER_LOG" 2>/dev/null; then
    break
  fi
  sleep 1
done
grep -q 'gRPC server listening' "$SERVER_LOG" \
  || { echo "server failed to start"; tail -20 "$SERVER_LOG"; exit 1; }
echo "::endgroup::"

echo "::group::load $SUBSET sift vectors + recompute subset groundtruth"
./benches/vector-ann/target/release/bench-vector-load \
  --endpoint "http://127.0.0.1:$PORT" \
  --train "$SIFT_DIR/sift_base.fvecs" \
  --subset-size "$SUBSET" \
  --metric euclidean \
  --query "$SIFT_DIR/sift_query.fvecs" \
  --write-groundtruth "$GT_FILE"
echo "::endgroup::"

K_THOUSAND=$((SUBSET / 1000))
DATASET_NAME="sift-128-${K_THOUSAND}k-euclidean"

for C in 1 4; do
  echo "::group::server-mode search concurrency=$C"
  ./benches/vector-ann/target/release/bench-vector-grpc \
    --endpoint "http://127.0.0.1:$PORT" \
    --query "$SIFT_DIR/sift_query.fvecs" \
    --groundtruth "$GT_FILE" \
    --dataset-name "$DATASET_NAME" \
    --concurrency "$C" \
    --output "$OUTPUT"
  echo "::endgroup::"
done

echo "server-mode bench complete; results in $OUTPUT"
