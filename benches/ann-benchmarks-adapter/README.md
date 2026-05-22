# ann-benchmarks adapter for CoordiNode

Thin Python adapter that lets the canonical [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
harness drive CoordiNode's native HNSW. Used by the bench-CI on every push
to `main` to produce the per-commit vector benchmark numbers on
docs.coordinode.com. Also used for the eventual upstream PR that puts
CoordiNode on the ann-benchmarks.com leaderboard.

## Layout

```
benches/ann-benchmarks-adapter/
├── Dockerfile        # FROM ann-benchmarks + install coordinode-embedded
├── module.py         # CoordiNode(BaseANN) — thin wrapper over coordinode_embedded.Hnsw
├── config.yml        # M / ef sweep, mirrors hnswlib's canonical ladder
├── __init__.py       # empty — makes the directory a Python package
└── README.md         # this file
```

## How the bench-CI uses it

1. Check out coordinode at the SHA being benched
2. Check out `structured-world/coordinode-python` main
3. Swap `coordinode-python/coordinode-rs/` submodule to point at the
   coordinode SHA from step 1
4. `maturin build --release -m coordinode-python/coordinode-embedded/Cargo.toml`
   produces `target/wheels/coordinode_embedded-*-cp311-abi3-manylinux*.whl`
5. Copy this directory to
   `/opt/annb/ann-benchmarks/ann_benchmarks/algorithms/coordinode/`
6. Copy the wheel from step 4 into the same directory
7. `cd /opt/annb/ann-benchmarks && python install.py --algorithm coordinode` —
   builds `ann-benchmarks-coordinode` Docker image (the wheel + module.py
   gets baked in)
8. `python run.py --algorithm coordinode --dataset <ds> --runs 1 --count 10`
   for every dataset in the matrix
9. Parse HDF5 results → JSON → push to `bench-data` branch keyed by
   coordinode SHA

The orchestrator lives at `scripts/run-coordinode-ann-benchmarks.sh`.

## Local run (for debugging the adapter)

```bash
# 1. Build the wheel
cd ~/projects/sw/coordinode/coordinode-python/coordinode-embedded
maturin build --release

# 2. Set up ann-benchmarks workspace (one-time)
git clone https://github.com/erikbern/ann-benchmarks /opt/annb/ann-benchmarks
cd /opt/annb/ann-benchmarks
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
docker build --rm -t ann-benchmarks -f ann_benchmarks/algorithms/base/Dockerfile .

# 3. Copy adapter + wheel
mkdir -p ann_benchmarks/algorithms/coordinode
cp ~/projects/sw/coordinode/coordinode/benches/ann-benchmarks-adapter/*.{py,yml,Dockerfile} \
   ann_benchmarks/algorithms/coordinode/
cp ~/projects/sw/coordinode/coordinode-python/target/wheels/coordinode_embedded-*.whl \
   ann_benchmarks/algorithms/coordinode/

# 4. Build adapter image (uses the wheel)
.venv/bin/python install.py --algorithm coordinode

# 5. Run on a small dataset
.venv/bin/python run.py --algorithm coordinode --dataset glove-25-angular --runs 1 --count 10
```

Results land under `results/<dataset>/<k>/coordinode/` as HDF5 files,
one per (M, ef) configuration.

## Two install modes for the Dockerfile

* **Local wheel (default)** — `docker build` finds `coordinode_embedded-*.whl`
  in the build context and `pip install`s it. This is what bench-CI uses
  for pre-release builds.
* **PyPI release** — pass `--build-arg PYPI_VERSION=<ver>` (e.g. `1.0.6`)
  and the Dockerfile pulls from PyPI instead. This is what the upstream
  ann-benchmarks PR will use — fully reproducible from public artifacts,
  no submodule magic required.

## Differences vs hnswlib's adapter

| | hnswlib | CoordiNode |
|---|---|---|
| Library install | `pip install hnswlib==0.8.0` | `pip install coordinode-embedded` |
| Index class | `hnswlib.Index(space=..., dim=...)` | `coordinode_embedded.Hnsw(dim=..., metric=...)` |
| Add | `p.add_items(X)` (parallel) | `idx.fit(X)` (serial — see PR #70 notes for the recall reason) |
| Set ef | `p.set_ef(ef)` | `idx.set_ef(ef)` |
| Query | `p.knn_query(v, k=n)[0][0]` | `idx.knn_query(v, k=n)` |

ef sweep + M ladder match hnswlib's canonical config so the resulting
Pareto curves overlay directly on the docs page.
