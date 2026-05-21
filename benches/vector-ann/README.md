# `bench-vector-ann` — ANN-Benchmarks adapter for CoordiNode HNSW

Recall@k + QPS sweep against the [Texmex INRIA](http://corpus-texmex.irisa.fr/) SIFT1M / GIST / Deep-Image / GloVe datasets, matching the methodology used by [ann-benchmarks.com](https://ann-benchmarks.com).

**Outputs canonical JSON** in [`bench-results/vector/<dataset>/<sha>-<subject>-<timestamp>.json`](../../bench-results/) — the gh-pages dashboard renders this format.

---

## What this is, and isn't

- **Is:** the CoordiNode side of an ANN benchmark. Runs CN's own HNSW (from `coordinode-vector`), records per-`ef_search` recall@10 + QPS + p50/p95/p99 latency, stamps with git + hardware fingerprint.
- **Isn't:** a competitor runner. **Competitors run separately** in Docker on the same host (see [Competitor runs](#competitor-runs) below) — their JSON results are placed into `bench-results/` **manually**, once per release / quarterly / when a competitor ships a notable HNSW change.
- **CI** runs ONLY the CoordiNode side, on `push` to `main`, on the self-hosted bench runner (`192.168.1.200`). Competitor numbers do NOT re-run on every commit.

---

## Server layout (192.168.1.200 / `gpu.private.systems`)

Dataset lives **permanently** on the runner — NEVER committed to the repo:

```
/srv/coordinode-bench/
├── datasets/
│   └── sift/                                  # SIFT1M — 128-dim, L2
│       ├── sift_base.fvecs                    # 1 000 000 × 128 × f32 (~512 MB)
│       ├── sift_query.fvecs                   # 10 000 × 128 × f32 (~5 MB)
│       └── sift_groundtruth.ivecs             # 10 000 × 100 × i32 (~4 MB)
├── github-runner/                              # self-hosted GH Actions runner
└── competitors/                                # Docker volume mounts for competitor benches
```

### One-time dataset placement

```bash
ssh gpu.private.systems
sudo mkdir -p /srv/coordinode-bench/datasets/sift
cd /srv/coordinode-bench/datasets/sift
curl -L -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar xzf sift.tar.gz --strip-components=1
rm sift.tar.gz
# Now you should have sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs
ls -lh
sudo chown -R gh-runner:gh-runner /srv/coordinode-bench
```

**Why on-disk:** SIFT1M is ~520 MB total. Committing it to the repo would inflate clones by ~half a gig for everyone. The dataset is also identical across the world's ANN literature — placing it on the host once is the standard pattern.

### Adding GloVe / Deep-Image / GIST later

Same `mkdir + curl + tar` pattern — point `--train` / `--query` / `--groundtruth` at the new files in the workflow YAML. The bench is dataset-agnostic; the harness reads any `.fvecs` / `.ivecs` pair.

---

## Running locally (during dev)

```bash
# A tiny synthetic dataset for smoke-testing changes to the harness
# itself (the runner doesn't need to be reachable):
cd benches/vector-ann
cargo run --release -- \
  --train /srv/coordinode-bench/datasets/sift/sift_base.fvecs \
  --query /srv/coordinode-bench/datasets/sift/sift_query.fvecs \
  --groundtruth /srv/coordinode-bench/datasets/sift/sift_groundtruth.ivecs \
  --dataset-name sift-128-euclidean \
  --m 32 \
  --ef-construction 200 \
  --codec none \
  --output ../../bench-results
```

Expect **15–30 minutes** for SIFT1M build + 6-point sweep on a modern desktop. The runner logs progress every 100 000 vectors.

---

## Competitor runs

Competitor benchmarks **are not re-run on every commit**. Standard procedure:

1. **Once per release** (or whenever a competitor publishes a notable HNSW change), SSH to the runner.
2. **Pick a competitor** from the list below. Each ships as a Docker container with a tiny Python wrapper that loads the same `.fvecs` files and emits CoordiNode JSON schema.
3. **Run it once.** Move the resulting JSON into the repo under `bench-results/vector/<dataset>/`. Commit it as `chore(bench): refresh <competitor> baseline`.

Docker compose for the competitor stack lives at [`docker/compose.competitors.yml`](docker/compose.competitors.yml):

| Competitor | Image | Output JSON `subject` field |
|---|---|---|
| hnswlib (specialist anchor) | `coordinode-bench-hnswlib:latest` | `hnswlib` |
| Faiss CPU (specialist anchor) | `coordinode-bench-faiss:latest` | `faiss-cpu` |
| MongoDB 8.x Atlas Vector (multi-model competitor) | `coordinode-bench-mongo-vector:latest` | `mongodb-8` |
| OpenSearch 2.x kNN (multi-model competitor) | `coordinode-bench-opensearch:latest` | `opensearch-2` |
| SurrealDB 3.x HNSW (multi-model competitor) | `coordinode-bench-surrealdb:latest` | `surrealdb-3` |

### One-shot competitor run

```bash
ssh gpu.private.systems
cd /srv/coordinode-bench/
docker compose -f /opt/coordinode/benches/vector-ann/docker/compose.competitors.yml \
  run --rm hnswlib \
  --dataset sift-128-euclidean
# JSON appears in ./out/vector/sift-128-euclidean/<sha-of-cn-repo>-hnswlib-<stamp>.json
# Move into the repo:
scp out/vector/sift-128-euclidean/*.json $LOCAL_REPO/bench-results/vector/sift-128-euclidean/
```

Then on local machine:

```bash
git add bench-results/vector/sift-128-euclidean/*.json
git commit -m "chore(bench): refresh hnswlib + faiss SIFT1M baseline"
git push origin main
```

The gh-pages workflow picks up the new JSON and re-renders the comparison chart.

---

## Output schema

See [`crates/coordinode-bench/src/lib.rs`](../../crates/coordinode-bench/src/lib.rs) for the canonical `BenchReport` struct. Vector-specific fields recorded by this binary:

| Field | Type | Meaning |
|---|---|---|
| `hnsw_m` | u32 | HNSW M parameter |
| `hnsw_ef_construction` | u32 | HNSW ef_construction parameter |
| `build_secs` | f64 | Wall-clock index build time |
| `dataset_n_train` | u32 | Training vector count |
| `dataset_n_test` | u32 | Query vector count |
| `dataset_dim` | u32 | Vector dimension |
| `k` | u32 | recall@k k parameter |
| `sweep` | array of SweepPoint | One row per ef_search value |
| `recall_at_k_peak` | f64 | Best recall@k across the sweep |
| `qps_at_recall_peak` | f64 | QPS at peak recall |
| `qps_at_recall_0_95` | f64 | QPS at first sweep point reaching recall ≥ 0.95 (dashboard headline cell) |

Each `SweepPoint`:

```json
{
  "ef_search": 64,
  "recall_at_k": 0.952,
  "qps": 4321.0,
  "latency_us_mean": 231.4,
  "latency_us_p50": 220.0,
  "latency_us_p95": 280.0,
  "latency_us_p99": 350.0
}
```

---

## Why `.fvecs`, not HDF5?

Texmex `.fvecs` / `.ivecs` is the **original** distribution format for SIFT1M, GIST, etc. ann-benchmarks.com later converts to HDF5 for tooling convenience, but every Rust HDF5 binding either pulls in `libhdf5` C deps or has a maintenance lag. `.fvecs` is a 5-line binary parser ([`src/fvecs.rs`](src/fvecs.rs)) — zero external dependencies, same numeric content as the HDF5 equivalent.

If a benchmark suite ONLY ships HDF5 (e.g. Deep-Image-96), the runner can convert once with a tiny Python script. We don't intend to add HDF5 to our Cargo deps for one dataset.
