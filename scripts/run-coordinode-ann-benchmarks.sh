#!/usr/bin/env bash
# Build the coordinode-embedded wheel against a specific coordinode SHA and
# run the ann-benchmarks Docker harness with the CoordiNode adapter against
# the documented dataset ladder.
#
# Ephemeral by design: everything writable lives under $WORK
# (defaults to $RUNNER_TEMP on a GH runner, or `mktemp -d` elsewhere).
# The only /opt paths it touches are read-only: ann-benchmarks library +
# dataset cache + .venv (with python3 reachable). No writes back to /opt.
#
# Usage:
#   scripts/run-coordinode-ann-benchmarks.sh [--sha <SHA>] [--datasets ds1,ds2,...]
#
# Output:
#   $OUT_DIR/vector/<dataset>/<sha>-<subject>-M<m>-<ts>.json
#
# Env-var overrides:
#   COORDINODE_REPO      = $GITHUB_WORKSPACE | $PWD  (must contain
#                          benches/ann-benchmarks-adapter/)
#   ANNB_ROOT            = /opt/annb/ann-benchmarks  (read-only base)
#   WORK                 = $RUNNER_TEMP | mktemp     (everything writable)
#   OUT_DIR              = $COORDINODE_REPO/bench-results

set -euo pipefail

: "${COORDINODE_REPO:=${GITHUB_WORKSPACE:-$PWD}}"
: "${ANNB_ROOT:=/opt/annb/ann-benchmarks}"
: "${WORK:=${RUNNER_TEMP:-$(mktemp -d -t coordinode-bench.XXXXXX)}}"
: "${OUT_DIR:=${COORDINODE_REPO}/bench-results}"

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
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$SHA" ]]; then
  SHA=$(git -C "$COORDINODE_REPO" rev-parse HEAD)
fi
SHORT_SHA=${SHA:0:7}
SUBJECT=${SUBJECT:-$ALGORITHM}

# Workspace layout (everything below is ephemeral):
#   $WORK/cp/                  – fresh coordinode-python clone for this run
#   $WORK/wheels/              – maturin build output
#   $WORK/annb-defs/coordinode – ann-benchmarks adapter staging (--definitions)
#   $WORK/annb-cwd/            – ann-benchmarks runtime cwd (symlinks into ANNB_ROOT
#                                + a writable `results/` subdir for Docker bind)
CP_REPO="$WORK/cp"
WHL_DIR="$WORK/wheels"
DEFS_DIR="$WORK/annb-defs"
ADAPTER_DST="$DEFS_DIR/coordinode"
ANNB_CWD="$WORK/annb-cwd"

mkdir -p "$WHL_DIR" "$ADAPTER_DST" "$ANNB_CWD/results"

echo "===> coordinode SHA      : $SHA"
echo "===> datasets            : $DATASETS"
echo "===> ANNB_ROOT (ro)      : $ANNB_ROOT"
echo "===> WORK (ephemeral)    : $WORK"
echo "===> OUT_DIR             : $OUT_DIR"

# 1. Clone coordinode-python into the ephemeral workspace + pin
#    coordinode-rs submodule to the SHA we're benching.
if [[ ! -d "$CP_REPO/.git" ]]; then
  git clone --depth 1 https://github.com/structured-world/coordinode-python.git "$CP_REPO"
fi
git -C "$CP_REPO" fetch --depth 1 origin main
git -C "$CP_REPO" reset --hard origin/main
git -C "$CP_REPO" submodule update --init --recursive --depth 1
git -C "$CP_REPO/coordinode-rs" fetch --depth 1 origin "$SHA"
git -C "$CP_REPO/coordinode-rs" checkout "$SHA"
echo "===> coordinode-rs submodule pinned at $SHA"

# 2. Build the PyO3 wheel into the ephemeral $WHL_DIR.
(
  cd "$CP_REPO/coordinode-embedded"
  uv run --with maturin maturin build --release --out "$WHL_DIR"
)
WHEEL=$(ls "$WHL_DIR"/coordinode_embedded-*.whl | head -1)
if [[ -z "$WHEEL" ]]; then
  echo "ERROR: maturin produced no wheel" >&2
  exit 1
fi
echo "===> built wheel: $WHEEL"

# 3. Stage adapter into the ephemeral --definitions tree.
cp "$COORDINODE_REPO/benches/ann-benchmarks-adapter/"{module.py,config.yml,Dockerfile,__init__.py} "$ADAPTER_DST/"
cp "$WHEEL" "$ADAPTER_DST/"
echo "===> staged adapter to $ADAPTER_DST"

# 4. Build adapter Docker image (cwd is the Docker build context).
(
  cd "$ADAPTER_DST"
  docker build --rm -t ann-benchmarks-coordinode -f Dockerfile .
)

# 5. ann-benchmarks runtime cwd: symlink the read-only ANNB_ROOT bits
#    we need (ann_benchmarks/, data/, run.py, .venv) into $ANNB_CWD,
#    then `cd $ANNB_CWD` so `results/` is the one writable mount.
for name in ann_benchmarks data run.py .venv; do
  ln -sfn "$ANNB_ROOT/$name" "$ANNB_CWD/$name"
done

# 6. Run benches per dataset.
mkdir -p "$OUT_DIR"
IFS=',' read -ra DS_ARR <<< "$DATASETS"
for ds in "${DS_ARR[@]}"; do
  echo "===> Running $ALGORITHM on $ds"
  (
    cd "$ANNB_CWD"
    .venv/bin/python run.py \
      --definitions "$DEFS_DIR" \
      --algorithm "$ALGORITHM" \
      --dataset "$ds" \
      --runs 1 \
      --count 10
  )
done

# 7. Flatten HDF5 results → bench-data schema.
VERSION_ARG=()
if [[ -n "$VERSION" ]]; then
  VERSION_ARG=(--version "$VERSION")
fi
"$ANNB_CWD/.venv/bin/python" "$COORDINODE_REPO/scripts/ann-benchmarks-to-json.py" \
  --annb-root "$ANNB_CWD" \
  --algorithm "$ALGORITHM" \
  --subject "$SUBJECT" \
  --coordinode-repo "$COORDINODE_REPO" \
  --sha "$SHA" \
  --datasets "$DATASETS" \
  --out-dir "$OUT_DIR" \
  "${VERSION_ARG[@]}"

echo "===> Done. JSONs under: $OUT_DIR/vector/"
