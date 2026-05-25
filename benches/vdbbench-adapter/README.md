# VDBBench adapter for CoordiNode (gRPC)

[VDBBench](https://github.com/zilliztech/VectorDBBench) (Zilliz) is the
server-class benchmark suite for vector databases — real LLM-embedding
datasets (Cohere 768d, Bioasq 1024d, OpenAI 1536d), concurrent search,
tail-latency under load.

This adapter talks to **`coordinode-server`** over gRPC, the same binary
that drives production deployments. Every VDBBench runner subprocess
(load + search) opens its own client to the shared server — the data
itself lives on the server, so it survives the `multiprocessing.spawn`
boundary that breaks an in-process adapter.

> **Why not in-process?** An earlier revision of this directory wrapped
> the `coordinode_embedded.Hnsw` PyO3 handle directly. That works for
> ann-benchmarks (single-process harness) but produces `recall ≈ 0` under
> VDBBench: the index built in the insert subprocess is dropped when
> that subprocess exits, and the freshly-spawned search subprocess sees
> an empty index. The same architectural reason is why Chroma / Milvus /
> Qdrant adapters all talk to a server. The in-process path remains
> available via `benches/ann-benchmarks-adapter/` for library-tier
> benchmarks.

## What's in here

| File | Purpose |
|------|---------|
| `__init__.py`   | Re-exports |
| `config.py`     | `CoordinodeConfig` (host + port) + `CoordinodeHnswConfig` (M / ef_construction / ef_search / max_elements) |
| `coordinode.py` | `CoordinodeDB` — `VectorDB` implementation: idempotent schema bootstrap, batched UNWIND-driven inserts, gRPC-thread-safe search |
| `cli.py`        | `vectordbbench coordinodehnsw` click command |

Filter pushdown is **not** wired yet (`supported_filter_types =
[FilterOp.NonFilter]`). When that lands, the adapter can advertise
`FilterOp.NumGE` etc. and dispatch via Cypher `WHERE` clauses.

## Prerequisites

- `coordinode-server` reachable from the runner host (default
  `localhost:7080`). On the bench host (`ro` / 192.168.1.200) it's
  typically managed via `systemd` or a tmux session — see
  `scripts/run-coordinode-ann-benchmarks.sh` for the canonical
  startup invocation, adapted for the server target.
- Python packages: `vectordb_bench`, `coordinode` (the gRPC client
  from
  [coordinode-python](https://github.com/structured-world/coordinode-python)),
  `numpy`.

## Run

VDBBench's `bench-data` registration pattern requires the adapter to
live under `vectordb_bench/backend/clients/coordinode/` on the runner
host. The simplest install is to symlink (or `rsync`) this directory
into the local VDBBench checkout:

```bash
ln -s "$(pwd)/benches/vdbbench-adapter" \
      /opt/vdbbench/VectorDBBench/vectordb_bench/backend/clients/coordinode
```

Then add the four registration lines to
`vectordb_bench/backend/clients/__init__.py` (one entry per `init_cls`
/ `config_cls` / `case_config_cls` table — same shape as `chroma`),
and the `from ..backend.clients.coordinode.cli import CoordinodeHNSW`
+ `cli.add_command(CoordinodeHNSW)` lines in
`vectordb_bench/cli/vectordbbench.py`.

Smoke run:

```bash
vectordbbench coordinodehnsw \
  --host localhost --port 7080 \
  --case-type Performance1536D50K \
  --m 16 --ef-construction 200 --ef-search 100 \
  --db-label coordinode-grpc-smoke \
  --num-concurrency 1,4,16
```

Real LLM-embedding cases (P0):

| Case | Dataset | Dim | Train | Notes |
|------|---------|-----|-------|-------|
| `Performance768D1M`   | Cohere   | 768  | 1 M   | Real Cohere embed-v3 |
| `Performance1024D1M`  | Bioasq   | 1024 | 1 M   | Real biomedical text |
| `Performance1536D500K` | OpenAI  | 1536 | 500K  | Real OpenAI ada-002 |

## License

Apache-2.0 — same as the rest of CoordiNode CE.
