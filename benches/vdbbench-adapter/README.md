# VDBBench adapter for CoordiNode

In-process HNSW adapter for [VDBBench](https://github.com/zilliztech/VectorDBBench),
the Level B (server-class) benchmark suite from Zilliz. Uses
`coordinode_embedded.Hnsw` (the same PyO3 binding as the
[ann-benchmarks adapter](../ann-benchmarks-adapter/README.md)) so the
build / search hot path is identical and only the harness changes.

## Status

Local-only. Not packaged inside the upstream `vectordb_bench` distribution
yet — point VDBBench at this directory via `PYTHONPATH` until we publish
a plugin or upstream it.

## What you get

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports |
| `config.py`   | `CoordinodeConfig` (DBConfig, empty — in-process) + `CoordinodeHnswConfig` (M / efConstruction / efSearch / maxElements) |
| `coordinode.py` | `CoordinodeDB` — `VectorDB` impl |

Filter pushdown is **not** wired yet (`supported_filter_types = [FilterOp.NonFilter]`).
Single-process by construction: `thread_safe = False` so each runner worker
gets a deep-copied instance and its own Rust `HnswIndex`.

## Prerequisites

```bash
pip install vectordb_bench coordinode-embedded numpy
```

The `coordinode-embedded` wheel ships via
[structured-world/coordinode-python](https://github.com/structured-world/coordinode-python)
(PyO3 + maturin, manylinux\_2\_28 on Linux x86\_64).

## Run

From this directory:

```bash
PYTHONPATH="$(pwd)/.." \
  vectordbbench coordinodehnsw \
    --case-type Performance768D1M \
    --num-concurrency 1 \
    --k 10
```

Or from a Python driver script — see
[`vectordb_bench/cli/cli.py`](https://github.com/zilliztech/VectorDBBench/blob/main/vectordb_bench/cli/cli.py)
for the canonical entry points. Register the DB module by adding to
`vectordb_bench/backend/clients/__init__.py` locally:

```python
from vdbbench_adapter import CoordinodeDB, CoordinodeConfig, CoordinodeHnswConfig
# wire into DB / DBConfig / DBCaseConfig enums per the VDBBench plugin protocol
```

(Upstream PR will add this registration; until then it's a manual edit on
the runner host. The same constraint applies to chroma / lancedb when run
outside their bundled paths.)

## Why this is separate from ann-benchmarks

ann-benchmarks measures **library-class** HNSW quality (recall vs QPS on a
single thread, no concurrency, no network). VDBBench measures
**server-class** behaviour: build time on real datasets (Cohere-1024,
OpenAI-1536, LAION-100M), concurrent search, tail latency under load,
streaming insert during query.

The shared core is the same `Hnsw` Rust handle — so any improvement to
`coordinode_vector::hnsw` shows up in both leaderboards without harness
drift.

## License

Apache-2.0 — same as the rest of CoordiNode CE.
