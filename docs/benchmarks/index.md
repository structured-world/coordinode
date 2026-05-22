---
title: Benchmarks
description: CoordiNode benchmark results across every modality — vector, graph, document, time-series, spatial, full-text. Compared against multi-model competitors and modality specialists on identical hardware.
---

<script setup>
import { defineAsyncComponent } from "vue";
// Loaded only on the client — the underlying ECharts / vue-echarts
// bundle accesses `document` at import time, which breaks
// VitePress's SSR render pass. Async + ClientOnly defers everything
// to mount-time.
const ModalityTabs = defineAsyncComponent(() => import("./ModalityTabs.vue"));
</script>

# Benchmarks

CoordiNode benchmarks use industry-standard suites (ann-benchmarks, LDBC SNB, YCSB, TSBS, Search Benchmark Game); every result is JSON-recorded with hardware fingerprint + Git SHA so the timeline is reproducible end-to-end.

::: tip Live data
The charts below are generated from JSON files at [`bench-results/`](https://github.com/structured-world/coordinode/tree/main/bench-results) on every commit. CoordiNode results are produced automatically by CI on a dedicated bench host (Intel i9-9900K, 8C/16T) on every push to `main`. Competitor baselines (hnswlib, Faiss, MongoDB, etc.) are rerun manually on the same host per release.
:::

<ClientOnly>
  <ModalityTabs />
</ClientOnly>

## How to read these charts

- **Pareto frontier** — higher and to the right is better. Recall (or correctness proxy) on the x-axis, throughput on the y-axis (log scale). Each curve is one engine.
- **Recall / accuracy timeline** — peak score across CoordiNode commits. The CoordiNode line moves on every push to `main`; competitor lines stay flat between manual re-baselines.
- **Throughput @ target quality** — the canonical "useful operating point" cell (e.g. QPS @ recall ≥ 0.95 for ANN). The taller the bar, the better the engine handles that point.

## Hardware fingerprint

Every bench JSON records the host:

```
Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz  (8 cores / 16 threads, 64 GB RAM)
```

This is intentionally **modest desktop-class hardware** — when CoordiNode reports a number on this CPU, the equivalent server-class run is going to be faster, not slower. The point is comparability: every engine runs on the same box.

## Why every modality is on one page

CoordiNode is a single engine, single transaction, single query language across all six modalities. A real workload mixes them — graph traversal feeding a vector search filtered by a time-series predicate over geo-indexed documents. Benchmarking each modality in isolation is the first step; the cross-modality workloads land as the per-modality binaries mature. The bench harness ([`crates/coordinode-bench`](https://github.com/structured-world/coordinode/tree/main/crates/coordinode-bench)) is modality-agnostic — adding a new dataset is a JSON + chart-spec addition.
