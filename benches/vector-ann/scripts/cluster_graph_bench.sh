#!/usr/bin/env bash
# Cluster-mode graph-traversal bench: an N-node localhost raft cluster
# measuring k-hop traversal read scaling on top of the single-process
# gRPC numbers from bench-graph. Two phases:
#
#   G1  load the synthetic social graph through the leader and measure
#       k-hop traversal against the LEADER only  -> single-node baseline
#   G2  re-measure the same queries round-robin across ALL nodes
#       -> replica read scaling (E_node = QPS(N nodes) / QPS(1 node))
#
# bench-graph generates and loads its own Barabasi-Albert graph, so no
# external dataset is required (unlike the vector cluster bench).
#
# Usage (from the repo root):
#   ./benches/vector-ann/scripts/cluster_graph_bench.sh
#
# Env knobs:
#   NODES         cluster size (default 3; 2 also works but has no raft
#                 failure tolerance)
#   GRAPH_NODES   Person nodes in the synthetic graph (default 20000)
#   HOPS          comma-separated hop sweep (default 1,2,3)
#   CONCURRENCY   comma-separated concurrency for the multi-node phase
#                 (default 3,6; keep multiples of NODES for an even spread)
#   BASE_PORT     first node's gRPC port (default 7300; nodes step by 10;
#                 ops port is gRPC+4)
#   SETTLE        seconds to wait for raft replication before G2 (default 15)
#   OUTPUT        bench-results output dir (default bench-results)
#
# The script owns the server processes it starts and kills ONLY those pids.

set -euo pipefail

NODES="${NODES:-3}"
GRAPH_NODES="${GRAPH_NODES:-20000}"
HOPS="${HOPS:-1,2,3}"
CONCURRENCY="${CONCURRENCY:-3,6}"
BASE_PORT="${BASE_PORT:-7300}"
SETTLE="${SETTLE:-15}"
OUTPUT="${OUTPUT:-bench-results}"

echo "::group::build server + bench-graph bins"
cargo build --release -p coordinode-server
( cd benches/vector-ann && cargo build --release --bin bench-graph )
echo "::endgroup::"

WORK_DIR="$(mktemp -d -t cn-graph-cluster-XXXXXX)"

PORTS=()
for i in $(seq 0 $((NODES - 1))); do
  PORTS+=("$((BASE_PORT + i * 10))")
done
PIDS=()

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    for i in $(seq 1 "$NODES"); do
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
  rm -rf -- "$WORK_DIR"
}
trap cleanup EXIT

# Peer list for a node excludes its own address. Raft peer/advertise
# addresses carry the http:// scheme (tonic Endpoint rejects scheme-less).
peers_for() {
  local self="$1" out=""
  for p in "${PORTS[@]}"; do
    if [ "$p" != "$self" ]; then
      out="${out:+$out,}http://127.0.0.1:$p"
    fi
  done
  echo "$out"
}

echo "::group::start $NODES nodes (ports ${PORTS[*]})"
for i in $(seq 1 "$NODES"); do
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
for i in $(seq 1 "$NODES"); do
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
ALL_ENDPOINTS="$LEADER"
for i in $(seq 1 $((NODES - 1))); do
  ALL_ENDPOINTS="$ALL_ENDPOINTS,http://127.0.0.1:${PORTS[$i]}"
done

if [ "$NODES" -gt 1 ]; then
  echo "::group::join nodes 2..$NODES"
  for i in $(seq 2 "$NODES"); do
    port="${PORTS[$((i - 1))]}"
    ./target/release/coordinode admin node join \
      --node "$LEADER" \
      --id "$i" \
      --addr "http://127.0.0.1:$port" \
      --follow
  done
  echo "::endgroup::"
fi

BENCH=./benches/vector-ann/target/release/bench-graph

echo "::group::G1 load graph + leader-only traversal baseline"
"$BENCH" \
  --endpoint "$LEADER" \
  --nodes "$GRAPH_NODES" \
  --hops "$HOPS" \
  --concurrency "$CONCURRENCY" \
  --dataset-name "social-ba-${GRAPH_NODES}" \
  --output "$OUTPUT"
echo "::endgroup::"

echo "waiting ${SETTLE}s for raft replication to settle on all replicas"
sleep "$SETTLE"

echo "::group::G2 round-robin traversal across all $NODES nodes"
"$BENCH" \
  --endpoint "$ALL_ENDPOINTS" \
  --nodes "$GRAPH_NODES" \
  --hops "$HOPS" \
  --concurrency "$CONCURRENCY" \
  --no-load \
  --dataset-name "social-ba-${GRAPH_NODES}" \
  --output "$OUTPUT"
echo "::endgroup::"

echo "cluster graph bench complete; results in $OUTPUT (compare G2 QPS vs G1 for E_node)"
