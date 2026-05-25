#!/usr/bin/env bash
# Run VDBBench against a freshly-built coordinode-server (CoordiNode at
# a specific SHA) on the bench host.  Designed to be invoked from CI
# over ssh, but also stand-alone-runnable from any host with the
# dependencies installed.
#
# Usage:
#   scripts/run-coordinode-vdbbench.sh [--sha <SHA>] [--cases <case1,case2>]
#
# Defaults:
#   --sha   = current HEAD of $COORDINODE_REPO
#   --cases = Performance1536D50K
#             (smoke case; expand to Performance768D1M / Performance1024D1M /
#              Performance1536D500K once the engine perf budget allows)
#
# Output:
#   $OUT_DIR/vector/<dataset>/<sha>-coordinode-vdbbench-<ts>.json
#
# Env-var overrides (defaults match the bench-host layout):
#   COORDINODE_REPO        = /opt/coordinode
#   VDBBENCH_REPO          = /opt/vdbbench/VectorDBBench
#   COORDINODE_GRPC_PORT   = 7090  (separate port to avoid colliding with
#                                   the systemd-managed production server)
#   COORDINODE_OPS_PORT    = 7094
#   COORDINODE_DATA        = /opt/coordinode-data-fresh
#   OUT_DIR                = $COORDINODE_REPO/bench-results

set -euo pipefail

: "${COORDINODE_REPO:=/opt/coordinode}"
: "${VDBBENCH_REPO:=/opt/vdbbench/VectorDBBench}"
: "${COORDINODE_GRPC_PORT:=7090}"
: "${COORDINODE_OPS_PORT:=7094}"
: "${COORDINODE_DATA:=/opt/coordinode-data-fresh}"
: "${OUT_DIR:=${COORDINODE_REPO}/bench-results}"

SHA=""
CASES="Performance1536D50K"
M=16
EF_CONSTRUCTION=200
EF_SEARCH=100
NUM_CONCURRENCY="1,4,16"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sha) SHA="$2"; shift 2 ;;
    --cases) CASES="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --ef-construction) EF_CONSTRUCTION="$2"; shift 2 ;;
    --ef-search) EF_SEARCH="$2"; shift 2 ;;
    --num-concurrency) NUM_CONCURRENCY="$2"; shift 2 ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$SHA" ]]; then
  SHA=$(git -C "$COORDINODE_REPO" rev-parse HEAD)
fi
SHORT_SHA=${SHA:0:7}

echo "===> SHA       : $SHA"
echo "===> cases     : $CASES"
echo "===> repo      : $COORDINODE_REPO"
echo "===> grpc port : $COORDINODE_GRPC_PORT"
echo "===> out dir   : $OUT_DIR"

# ── 1. Sync coordinode to the SHA we want to bench ──────────────────
git -C "$COORDINODE_REPO" fetch origin "$SHA" 2>/dev/null || true
git -C "$COORDINODE_REPO" checkout "$SHA"
git -C "$COORDINODE_REPO" submodule update --init --recursive

# ── 2. Build release binary (idempotent — cargo caches) ─────────────
echo "===> building coordinode-server --release"
( cd "$COORDINODE_REPO" && cargo build -p coordinode-server --release )

# ── 3. Stop any prior fresh-bench server, start a new one with empty data ──
echo "===> stopping any prior bench server"
pkill -f "${COORDINODE_REPO}/target/release/coordinode" || true
sleep 2

if [[ -d "$COORDINODE_DATA" ]]; then
  # Archive instead of rm — operator can inspect last run's state.
  mv "$COORDINODE_DATA" "${COORDINODE_DATA}.$(date +%s).old"
fi
mkdir -p "$COORDINODE_DATA"

echo "===> starting fresh coordinode-server on :$COORDINODE_GRPC_PORT"
LOG="${COORDINODE_DATA}.startup.log"
nohup "$COORDINODE_REPO/target/release/coordinode" serve \
    --addr "0.0.0.0:${COORDINODE_GRPC_PORT}" \
    --ops-addr "127.0.0.1:${COORDINODE_OPS_PORT}" \
    --data "$COORDINODE_DATA" \
    --mode full \
  > "$LOG" 2>&1 &
SERVER_PID=$!
echo "===> server PID $SERVER_PID"
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# Wait for it to bind the gRPC port (max 30s).
for _ in $(seq 1 30); do
  if ss -tln 2>/dev/null | grep -q ":${COORDINODE_GRPC_PORT}\b"; then
    break
  fi
  sleep 1
done

# ── 4. Refresh the VDBBench adapter from this checkout ──────────────
ADAPTER_DST="$VDBBENCH_REPO/vectordb_bench/backend/clients/coordinode"
mkdir -p "$ADAPTER_DST"
for f in __init__.py config.py coordinode.py cli.py; do
  cp "$COORDINODE_REPO/benches/vdbbench-adapter/$f" "$ADAPTER_DST/$f"
done
echo "===> staged adapter into $ADAPTER_DST"

# ── 5. Run VDBBench per case ────────────────────────────────────────
IFS=',' read -ra CASE_ARR <<< "$CASES"
for case in "${CASE_ARR[@]}"; do
  echo "===> vectordbbench coordinodehnsw --case-type $case"
  (
    cd "$VDBBENCH_REPO"
    .venv/bin/vectordbbench coordinodehnsw \
      --host 127.0.0.1 \
      --port "$COORDINODE_GRPC_PORT" \
      --case-type "$case" \
      --m "$M" \
      --ef-construction "$EF_CONSTRUCTION" \
      --ef-search "$EF_SEARCH" \
      --db-label "coordinode-${SHORT_SHA}" \
      --num-concurrency "$NUM_CONCURRENCY"
  )
done

# ── 6. Convert VDBBench JSONs → bench-results schema ────────────────
echo "===> converting VDBBench results"
mkdir -p "$OUT_DIR"
# Filter to only THIS run's JSONs by mtime > start_of_script
SRC_DIR="$VDBBENCH_REPO/vectordb_bench/results/CoordiNode"
"$VDBBENCH_REPO/.venv/bin/python3" "$COORDINODE_REPO/scripts/vdbbench-to-json.py" \
  --src "$SRC_DIR" \
  --sha "$SHA" \
  --coordinode-repo "$COORDINODE_REPO" \
  --version "$(git -C "$COORDINODE_REPO" describe --tags --always 2>/dev/null || echo HEAD)" \
  --out-dir "$OUT_DIR"

echo "===> Done. JSONs under: $OUT_DIR/vector/"
