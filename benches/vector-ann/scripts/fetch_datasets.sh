#!/usr/bin/env bash
#
# One-shot preparation of the datasets the per-commit vector bench reads:
# SIFT1M (128-d, L2) and GloVe-100 (angular). Reads $DATASET_ROOT (the
# variable the workflow uses) and writes the .fvecs/.ivecs triplets to
# $DATASET_ROOT/sift/ and $DATASET_ROOT/glove-100-angular/. Idempotent: a
# dataset whose triplet is present is skipped.
#
# Both come from ann-benchmarks.com as HDF5 over HTTPS (the original
# Texmex SIFT archive is served over FTP, which a CI network may not pass)
# and are converted with hdf5_to_fvecs.py; the numbers are the same.
#
# Needs curl and python3 with h5py and numpy (Fedora: dnf install
# python3-h5py python3-numpy). Downloads about 1 GB once.
#
# Usage:
#
#   DATASET_ROOT=/srv/coordinode-bench/datasets ./fetch_datasets.sh

set -euo pipefail

if [ -z "${DATASET_ROOT:-}" ]; then
  echo "DATASET_ROOT not set; aborting." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# fetch <name on ann-benchmarks> <directory> <file prefix>
fetch() {
  local name="$1" dir="$DATASET_ROOT/$2" prefix="$3"
  if [ -f "$dir/${prefix}_base.fvecs" ] && [ -f "$dir/${prefix}_query.fvecs" ] \
    && [ -f "$dir/${prefix}_groundtruth.ivecs" ]; then
    echo "$dir: present, skipped"
    return
  fi
  mkdir -p "$dir"
  local hdf5="$dir/$name.hdf5"
  curl -fL --retry 3 -o "$hdf5.part" "https://ann-benchmarks.com/$name.hdf5"
  mv "$hdf5.part" "$hdf5"
  python3 "$SCRIPT_DIR/hdf5_to_fvecs.py" --hdf5 "$hdf5" --out-dir "$dir" --prefix "$prefix"
  rm -f "$hdf5"
}

fetch sift-128-euclidean sift sift
fetch glove-100-angular glove-100-angular glove

echo "datasets ready under $DATASET_ROOT"
