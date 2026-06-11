#!/usr/bin/env bash
# Cluster-mode vector bench: 3-node localhost RF=3 cluster, measuring
# the raft overlay on top of the single-process gRPC numbers from
# server_bench.sh. Emits two cell families:
#
#   L2  search against the LEADER only      -> raft read-path overlay
#   L3  search round-robin across all nodes -> replica read scaling
#
# (L0 = in-process HnswIndex, L1 = single-process gRPC; see
# server_bench.sh. L2/L1 prices the cluster read fence, L3/L2 shows
# whether replica reads scale.)
#
# Usage (from the repo root):
#   DATASET_ROOT=/path/to/datasets ./benches/vector-ann/scripts/cluster_bench.sh
#
# Env knobs:
#   DATASET_ROOT  required; expects $DATASET_ROOT/sift/sift_{base,query}.fvecs
#   SUBSET        train subset size (default 50000)
#   BASE_PORT     first node's gRPC port (default 7200; nodes use
#                 BASE_PORT, +10, +20; ops ports are gRPC+4)
#   OUTPUT        bench-results output dir (default bench-results)
#
# The script owns the three server processes it starts and kills ONLY
# those pids.

set -euo pipefail

SUBSET="${SUBSET:-50000}"
BASE_PORT="${BASE_PORT:-7200}"
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

WORK_DIR="$(mktemp -d -t cn-cluster-bench-XXXXXX)"
GT_FILE="$WORK_DIR/subset_gt.ivecs"

PORTS=("$BASE_PORT" "$((BASE_PORT + 10))" "$((BASE_PORT + 20))")
PIDS=()

cleanup() {
  status=$?
  # On failure, surface the node logs before the temp dir is lost —
  # without them a CI failure is undiagnosable (leader elections,
  # backfill errors, replication stalls all live here).
  if [ "$status" -ne 0 ]; then
    for i in 1 2 3; do
      echo "::group::node$i.log tail (exit $status)"
      tail -40 "$WORK_DIR/node$i.log" 2>/dev/null || true
      echo "::endgroup::"
    done
  fi
  for pid in "${PIDS[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  # The work dir holds 3 node data dirs plus raft snapshots (hundreds
  # of MB per run); leaking it fills the runner's disk within a few
  # runs.
  rm -rf -- "$WORK_DIR"
}
trap cleanup EXIT

# Peer list for node N excludes its own address. Raft peer/advertise
# addresses carry the http:// scheme: the gRPC network layer feeds
# them to tonic Endpoint::from_shared, which rejects scheme-less URIs.
peers_for() {
  local self="$1" out=""
  for p in "${PORTS[@]}"; do
    if [ "$p" != "$self" ]; then
      out="${out:+$out,}http://127.0.0.1:$p"
    fi
  done
  echo "$out"
}

echo "::group::start 3 nodes (ports ${PORTS[*]})"
for i in 1 2 3; do
  port="${PORTS[$((i - 1))]}"
  data="$WORK_DIR/node$i"
  mkdir -p "$data"
  ./target/release/coordinode serve \
    --node-id "$i" \
    --addr "127.0.0.1:$port" \
    --advertise-addr "http://127.0.0.1:$port" \
    --ops-addr "127.0.0.1:$((port + 4))" \
    --data "$data" \
    --peers "$(peers_for "$port")" > "$WORK_DIR/node$i.log" 2>&1 &
  PIDS+=($!)
done
for i in 1 2 3; do
  log="$WORK_DIR/node$i.log"
  for _ in $(seq 1 30); do
    grep -q 'gRPC server listening' "$log" 2>/dev/null && break
    sleep 1
  done
  grep -q 'gRPC server listening' "$log" \
    || { echo "node$i failed to start"; tail -20 "$log"; exit 1; }
done
echo "::endgroup::"

LEADER="http://127.0.0.1:${PORTS[0]}"
ALL_ENDPOINTS="http://127.0.0.1:${PORTS[0]},http://127.0.0.1:${PORTS[1]},http://127.0.0.1:${PORTS[2]}"

echo "::group::join nodes 2 and 3"
for i in 2 3; do
  port="${PORTS[$((i - 1))]}"
  ./target/release/coordinode admin node join \
    --node "$LEADER" \
    --id "$i" \
    --addr "http://127.0.0.1:$port" \
    --follow
done
echo "::endgroup::"

echo "::group::load $SUBSET sift vectors through the leader + subset groundtruth"
./benches/vector-ann/target/release/bench-vector-load \
  --endpoint "$LEADER" \
  --train "$SIFT_DIR/sift_base.fvecs" \
  --subset-size "$SUBSET" \
  --metric euclidean \
  --query "$SIFT_DIR/sift_query.fvecs" \
  --write-groundtruth "$GT_FILE"
echo "::endgroup::"

# The CREATE VECTOR INDEX backfill is asynchronous and runs
# independently on every replica; searching before it settles measures
# a partially built index (visible as low recall on followers). No
# readiness RPC exists yet, so wait a fixed settle period scaled for
# the subset size.
INDEX_WAIT="${INDEX_WAIT:-30}"
echo "waiting ${INDEX_WAIT}s for index backfill to settle on all replicas"
sleep "$INDEX_WAIT"

K_THOUSAND=$((SUBSET / 1000))
DATASET_NAME="sift-128-${K_THOUSAND}k-euclidean"

run_search() {
  local endpoints="$1" conc="$2" topo="$3" pref="$4"
  ./benches/vector-ann/target/release/bench-vector-grpc \
    --endpoint "$endpoints" \
    --query "$SIFT_DIR/sift_query.fvecs" \
    --groundtruth "$GT_FILE" \
    --dataset-name "$DATASET_NAME" \
    --concurrency "$conc" \
    --topology "$topo" \
    --read-preference "$pref" \
    --output "$OUTPUT"
}

# L2: leader-only reads. Same concurrency points as server_bench.sh
# so L2/L1 divides cleanly. All endpoints are passed so the bench's
# leader-locating retry survives elections (leadership has been
# observed to move off node1 mid-run under load); `primary` still
# guarantees every recorded query was served by the leader.
for C in 1 4; do
  echo "::group::L2 leader-only search concurrency=$C"
  run_search "$ALL_ENDPOINTS" "$C" "L2" "primary"
  echo "::endgroup::"
done

# L3: round-robin across all three nodes; nearest lets followers
# serve their workers' reads locally. Concurrency multiples of 3
# keep the worker -> endpoint spread even.
for C in 3 6; do
  echo "::group::L3 round-robin search concurrency=$C"
  run_search "$ALL_ENDPOINTS" "$C" "L3" "nearest"
  echo "::endgroup::"
done

echo "cluster-mode bench complete; results in $OUTPUT"
