#!/usr/bin/env bash
# Build the coordinode-embedded wheel against a specific coordinode SHA and
# run ann-benchmarks Docker harness with the CoordiNode adapter against the
# documented dataset ladder. Designed for the bench-CI self-hosted runner
# (ro / 192.168.1.200) but works on any Linux host with docker + uv installed.
#
# Usage:
#   scripts/run-coordinode-ann-benchmarks.sh [--sha <coordinode SHA>] [--datasets ds1,ds2,...]
#
# Defaults:
#   --sha     = current HEAD of the COORDINODE_REPO checkout
#   --datasets= glove-25-angular,sift-128-euclidean,nytimes-256-angular,fashion-mnist-784-euclidean
#               (gist-960 + glove-{50,100,200} added opt-in via --datasets)
#
# Output:
#   ${ANNB_ROOT}/results/<dataset>/10/coordinode/*.hdf5   (ann-benchmarks native)
#   ${OUT_JSON}                                            (flat JSON for the docs site)
#
# Env-var overrides (with defaults that match the bench host layout):
#   COORDINODE_REPO         = /opt/coordinode
#   COORDINODE_PYTHON_REPO  = /opt/coordinode-python
#   ANNB_ROOT               = /opt/annb/ann-benchmarks
#   OUT_JSON                = ${COORDINODE_REPO}/bench-data-staging/coordinode-<sha>.json

set -euo pipefail

: "${COORDINODE_REPO:=/opt/coordinode}"
: "${COORDINODE_PYTHON_REPO:=/opt/coordinode-python}"
: "${ANNB_ROOT:=/opt/annb/ann-benchmarks}"

DATASETS="glove-25-angular,sift-128-euclidean,nytimes-256-angular,fashion-mnist-784-euclidean"
SHA=""
ALGORITHM="coordinode"
SUBJECT=""
VERSION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sha) SHA="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --algorithm) ALGORITHM="$2"; shift 2 ;;
    --subject) SUBJECT="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown flag $1"; exit 2 ;;
  esac
done

if [[ -z "$SHA" ]]; then
  SHA=$(git -C "$COORDINODE_REPO" rev-parse HEAD)
fi
SHORT_SHA=${SHA:0:7}
SUBJECT=${SUBJECT:-$ALGORITHM}
: "${OUT_DIR:=${COORDINODE_REPO}/bench-results}"

echo "===> coordinode SHA      : $SHA"
echo "===> datasets            : $DATASETS"
echo "===> ann-benchmarks root : $ANNB_ROOT"
echo "===> ann-benchmarks algo : $ALGORITHM"
echo "===> bench subject       : $SUBJECT"
echo "===> out dir             : $OUT_DIR"

# 1. coordinode checkout at the right SHA
git -C "$COORDINODE_REPO" fetch origin "$SHA" 2>/dev/null || true
git -C "$COORDINODE_REPO" checkout "$SHA"

# 2. coordinode-python: latest main + submodule swap to coordinode SHA
if [[ ! -d "$COORDINODE_PYTHON_REPO/.git" ]]; then
  git clone https://github.com/structured-world/coordinode-python.git "$COORDINODE_PYTHON_REPO"
fi
git -C "$COORDINODE_PYTHON_REPO" fetch origin main
git -C "$COORDINODE_PYTHON_REPO" checkout main
git -C "$COORDINODE_PYTHON_REPO" reset --hard origin/main
git -C "$COORDINODE_PYTHON_REPO" submodule update --init --recursive --depth 1

# Point the coordinode-rs submodule at the SHA we're benching (engine HEAD on this run).
git -C "$COORDINODE_PYTHON_REPO/coordinode-rs" fetch --depth 1 origin "$SHA"
git -C "$COORDINODE_PYTHON_REPO/coordinode-rs" checkout "$SHA"
echo "===> coordinode-rs submodule pinned at $SHA"

# 3. Build wheel
WHL_DIR="$COORDINODE_PYTHON_REPO/target/wheels"
rm -rf "$WHL_DIR"
(
  cd "$COORDINODE_PYTHON_REPO/coordinode-embedded"
  uv run --with maturin maturin build --release --out "$WHL_DIR"
)
WHEEL=$(ls "$WHL_DIR"/coordinode_embedded-*.whl | head -1)
if [[ -z "$WHEEL" ]]; then
  echo "ERROR: maturin produced no wheel"
  exit 1
fi
echo "===> built wheel: $WHEEL"

# 4. Stage adapter + wheel into ann-benchmarks workspace
ADAPTER_DST="$ANNB_ROOT/ann_benchmarks/algorithms/coordinode"
rm -rf "$ADAPTER_DST"
mkdir -p "$ADAPTER_DST"
cp "$COORDINODE_REPO/benches/ann-benchmarks-adapter/"{module.py,config.yml,Dockerfile,__init__.py} "$ADAPTER_DST/"
cp "$WHEEL" "$ADAPTER_DST/"
echo "===> staged adapter to $ADAPTER_DST"

# 5. Build adapter Docker image
(
  cd "$ANNB_ROOT"
  docker build --rm -t ann-benchmarks-coordinode \
    -f ann_benchmarks/algorithms/coordinode/Dockerfile \
    ann_benchmarks/algorithms/coordinode/
)

# 6. Run benches for each dataset
mkdir -p "$OUT_DIR"
IFS=',' read -ra DS_ARR <<< "$DATASETS"
for ds in "${DS_ARR[@]}"; do
  echo "===> Running $ALGORITHM on $ds"
  (
    cd "$ANNB_ROOT"
    .venv/bin/python run.py \
      --algorithm "$ALGORITHM" \
      --dataset "$ds" \
      --runs 1 \
      --count 10
  )
done

# 7. Export HDF5 → bench-results/vector/<dataset>/<sha>-<subject>-M<m>-<ts>.json
VERSION_ARG=()
if [[ -n "$VERSION" ]]; then
  VERSION_ARG=(--version "$VERSION")
fi
python3 "$COORDINODE_REPO/scripts/ann-benchmarks-to-json.py" \
  --annb-root "$ANNB_ROOT" \
  --algorithm "$ALGORITHM" \
  --subject "$SUBJECT" \
  --coordinode-repo "$COORDINODE_REPO" \
  --sha "$SHA" \
  --datasets "$DATASETS" \
  --out-dir "$OUT_DIR" \
  "${VERSION_ARG[@]}"

echo "===> Done.  JSONs under: $OUT_DIR/vector/"
