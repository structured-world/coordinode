---
title: Benchmarks
description: CoordiNode benchmark results, updated automatically on every push to main. Compares against multi-model competitors (SurrealDB, ArangoDB, MongoDB, OpenSearch, PostGIS) and modality specialists (hnswlib, Faiss, Neo4j, QuestDB).
---

<script setup>
import { defineAsyncComponent } from "vue";
// Loaded only on the client — the underlying ECharts / vue-echarts
// bundle accesses `document` at import time, which breaks
// VitePress's SSR render pass. Async + ClientOnly defers everything
// to mount-time.
const BenchVectorAnn = defineAsyncComponent(() => import("./BenchVectorAnn.vue"));
</script>

# Benchmarks

CoordiNode benchmarks follow [public, reproducible methodology](https://github.com/structured-world/coordinode/blob/main/arch/benchmarks/methodology.md) — every dataset is an industry-standard suite (ann-benchmarks, LDBC SNB, YCSB, TSBS, Search Benchmark Game), every result is JSON-recorded with hardware fingerprint + Git SHA so the timeline is reproducible end-to-end.

::: tip Live data
The charts below are generated from JSON files at [`bench-results/`](https://github.com/structured-world/coordinode/tree/main/bench-results) on every commit. CoordiNode results are produced automatically by CI on a dedicated bench host (Intel i9-9900K, 8C/16T) on every push to `main`. Competitor baselines (hnswlib, Faiss, MongoDB, etc.) are rerun manually on the same host per release.
:::

## Vector — ANN benchmarks (SIFT1M, 128-dim, L2)

The headline ANN benchmark: 1 000 000 SIFT1M base vectors, 10 000 query vectors, ground-truth top-100. CoordiNode HNSW (`coordinode-vector`) with `M=32`, `ef_construction=200`, six-point `ef_search` sweep ∈ {16, 32, 64, 128, 256, 512}, 10 query-replay rounds per point, single-thread search.

<ClientOnly>
  <BenchVectorAnn />
</ClientOnly>

### How to read these charts

- **Pareto frontier** — higher and to the right is better. Recall@10 on the x-axis, QPS (single-thread) on the y-axis (log scale). Each curve is one engine.
- **Recall@10 timeline** — peak recall across CN commits. The CoordiNode line moves on every push to `main`; competitor lines stay flat between manual re-baselines.
- **QPS @ recall ≥ 0.95** — the canonical ann-benchmarks dashboard cell. The taller the bar, the better the engine handles the canonical operating point (~95% true top-10 retrieved at the highest possible throughput).

### Hardware fingerprint

Every bench JSON records the host:

```
Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz  (8 cores / 16 threads, 64 GB RAM)
```

This is intentionally **modest desktop-class hardware** — when CoordiNode reports a number on this CPU, the equivalent server-class run is going to be faster, not slower. The point is comparability: every engine runs on the same box.

## Other modalities

Graph (LDBC SNB), spatial (PostGIS-shape), time-series (TSBS), full-text (Search Benchmark Game), document (YCSB A/C) — coming as the per-modality bench binaries land. The bench harness ([`crates/coordinode-bench`](https://github.com/structured-world/coordinode/tree/main/crates/coordinode-bench)) is modality-agnostic; adding a new dataset is a JSON + chart-spec addition.

See the [methodology document](https://github.com/structured-world/coordinode/blob/main/arch/benchmarks/methodology.md) for the full multi-model competitor matrix and codec-fairness rules.
