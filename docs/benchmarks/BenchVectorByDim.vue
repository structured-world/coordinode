<script setup lang="ts">
// Product-facing vector benchmark view: "QPS at recall ≥ X across dimensions".
//
// A developer arrives with an embedding model (e.g., text-embedding-3-large = 3072d,
// Cohere v3 = 1024d, BERT = 768d). They want to look at one chart and answer
// "at MY dimension, is CoordiNode faster than $alternative?" — without scrolling
// through commit timelines or interpreting Pareto frontiers.
//
// Primary chart:
//   X = dimension (log)
//   Y = QPS at recall ≥ target (log)
//   one line per engine
//
// Secondary chart:
//   X = dimension (log)
//   Y = RAM cost per 1M vectors (log)
//
// Data source: a flat JSON registry of measured datapoints — one row per
// (engine, dataset, dim, scale, recall_target, qps, ram_mb). Each row carries
// provenance (source: 'measured-on-bench-host' | 'ann-benchmarks.com' | 'vendor').
// Cells without measurements render as "pending — see ROADMAP".
import { computed, ref } from "vue";
import VChart from "vue-echarts";
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { LineChart } from "echarts/charts";
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  ToolboxComponent,
  MarkLineComponent,
} from "echarts/components";

use([
  CanvasRenderer,
  LineChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  ToolboxComponent,
  MarkLineComponent,
]);

interface Datapoint {
  engine: string;
  dataset: string;
  dim: number;
  scale: number; // dataset size (vectors)
  recall: number; // target recall@10 the QPS is measured at
  qps: number;
  ram_mb_per_1m: number | null; // RAM cost normalised to 1M vectors
  source: "bench-host" | "ann-benchmarks.com" | "VDBBench";
  notes?: string;
}

// Data registry — ONLY cited public numbers + our own bench measurements.
// Mirrors arch/benchmarks/methodology.md § Modality 3.
//
// Source rules:
//  * CoordiNode entries: measured on our runner (Intel i9-9900K, 1 thread).
//  * Competitor entries: cited from ann-benchmarks.com (Level A) and
//    VectorDBBench (Level B) per documented methodology baselines.
//  * NO "projected" / forward-looking estimates. The chart shows what
//    actually exists; gaps are gaps. Numbers populate as bench runs land.
const DATA: Datapoint[] = [
  // === Level A: ann-benchmarks SIFT1M, d=128, recall ≥ 0.95, single CPU ===
  // Cited from arch/benchmarks/methodology.md § Modality 3 Level A baselines.
  // RAM not in ann-benchmarks output schema → null (chart RAM row stays empty).
  { engine: "hnswlib", dataset: "sift-128-euclidean", dim: 128, scale: 1_000_000, recall: 0.95, qps: 18000, ram_mb_per_1m: null, source: "ann-benchmarks.com" },
  { engine: "FAISS-HNSW", dataset: "sift-128-euclidean", dim: 128, scale: 1_000_000, recall: 0.95, qps: 15000, ram_mb_per_1m: null, source: "ann-benchmarks.com" },
  { engine: "ScaNN", dataset: "sift-128-euclidean", dim: 128, scale: 1_000_000, recall: 0.95, qps: 20000, ram_mb_per_1m: null, source: "ann-benchmarks.com" },
  { engine: "Annoy", dataset: "sift-128-euclidean", dim: 128, scale: 1_000_000, recall: 0.95, qps: 5000, ram_mb_per_1m: null, source: "ann-benchmarks.com" },
  { engine: "pgvector (HNSW)", dataset: "sift-128-euclidean", dim: 128, scale: 1_000_000, recall: 0.95, qps: 3000, ram_mb_per_1m: null, source: "ann-benchmarks.com" },

  // CoordiNode SIFT1M — measured on our runner (Intel i9-9900K).
  // Bench harness today records timing + recall, NOT RSS — the RSS-sampling
  // addition is the next part of R868. ram_mb_per_1m stays null until that lands.
  { engine: "CoordiNode (current main)", dataset: "sift-128-euclidean", dim: 128, scale: 1_000_000, recall: 0.95, qps: 1317, ram_mb_per_1m: null, source: "bench-host", notes: "f763f86 single-thread, codec=none, M=32, ef_construction=200" },

  // === Level B: VectorDBBench Cohere-768 (d=768, 1M vectors) ===
  // Cited from arch/benchmarks/methodology.md § Modality 3 Level B baselines.
  // Modality specialists as secondary anchors per documented methodology
  // (primary multi-model competitors land here as we run the matrix on our host).
  { engine: "Qdrant", dataset: "cohere-768-vdbbench", dim: 768, scale: 1_000_000, recall: 0.98, qps: 5000, ram_mb_per_1m: null, source: "VDBBench", notes: "p99 ~8 ms" },
  { engine: "Milvus", dataset: "cohere-768-vdbbench", dim: 768, scale: 1_000_000, recall: 0.97, qps: 4000, ram_mb_per_1m: null, source: "VDBBench", notes: "p99 ~12 ms" },
  { engine: "Weaviate", dataset: "cohere-768-vdbbench", dim: 768, scale: 1_000_000, recall: 0.96, qps: 3000, ram_mb_per_1m: null, source: "VDBBench", notes: "p99 ~15 ms" },
  { engine: "Elasticsearch 8.x", dataset: "cohere-768-vdbbench", dim: 768, scale: 1_000_000, recall: 0.94, qps: 2000, ram_mb_per_1m: null, source: "VDBBench", notes: "p99 ~20 ms" },
  { engine: "pgvector (HNSW)", dataset: "cohere-768-vdbbench", dim: 768, scale: 1_000_000, recall: 0.95, qps: 1500, ram_mb_per_1m: null, source: "VDBBench", notes: "p99 ~25 ms" },
];

// Controls
const recallTarget = ref<0.9 | 0.95 | 0.99>(0.95);
const scale = ref<number>(1_000_000);

// Compute engines list (stable order, CoordiNode first, then alphabetical)
const allEngines = computed(() => {
  const set = new Set<string>(DATA.map((d) => d.engine));
  const cn = [...set].filter((e) => e.startsWith("CoordiNode")).sort();
  const others = [...set].filter((e) => !e.startsWith("CoordiNode")).sort();
  return [...cn, ...others];
});
const enabledEngines = ref<Set<string>>(new Set(allEngines.value));

function toggleEngine(e: string): void {
  const next = new Set(enabledEngines.value);
  if (next.has(e)) next.delete(e);
  else next.add(e);
  enabledEngines.value = next;
}

function isCoordinode(e: string): boolean {
  return e.startsWith("CoordiNode");
}

// Group datapoints by engine for the active filter
const seriesByEngine = computed(() => {
  const grouped = new Map<string, { dim: number; qps: number; ram: number | null; source: string; notes?: string }[]>();
  for (const d of DATA) {
    if (d.scale !== scale.value) continue;
    if (d.recall !== recallTarget.value) continue;
    if (!enabledEngines.value.has(d.engine)) continue;
    if (d.qps === 0) continue; // skip 0-QPS pending rows in the chart
    const arr = grouped.get(d.engine) ?? [];
    arr.push({ dim: d.dim, qps: d.qps, ram: d.ram_mb_per_1m, source: d.source, notes: d.notes });
    grouped.set(d.engine, arr);
  }
  // Sort each series by dim ascending
  for (const arr of grouped.values()) {
    arr.sort((a, b) => a.dim - b.dim);
  }
  return grouped;
});

const lineColor = (engine: string): string => {
  if (engine.startsWith("CoordiNode")) return "#c94d20";
  if (engine.startsWith("hnswlib")) return "#7f8c8d";
  if (engine.startsWith("FAISS")) return "#3b5998";
  if (engine.startsWith("ScaNN")) return "#34a853";
  if (engine.startsWith("Annoy")) return "#e8a33d";
  if (engine.startsWith("pgvector")) return "#336791";
  if (engine.startsWith("Qdrant")) return "#dc382d";
  if (engine.startsWith("Milvus")) return "#00a1ea";
  if (engine.startsWith("Weaviate")) return "#1a9e6e";
  if (engine.startsWith("Elastic")) return "#f6b352";
  return "#34495e";
};

const qpsChartOption = computed(() => {
  const series = [...seriesByEngine.value.entries()].map(([engine, points]) => ({
    name: engine,
    type: "line" as const,
    data: points.map((p) => [p.dim, p.qps]),
    showSymbol: true,
    symbolSize: 8,
    lineStyle: { color: lineColor(engine), width: engine.startsWith("CoordiNode") ? 3 : 2 },
    itemStyle: { color: lineColor(engine) },
    emphasis: { focus: "series" as const },
  }));
  return {
    grid: { left: 60, right: 30, top: 30, bottom: 50 },
    legend: { type: "scroll" as const, top: 0, textStyle: { color: "var(--vp-c-text-1)" } },
    tooltip: {
      trigger: "axis" as const,
      axisPointer: { type: "cross" as const },
      formatter: (params: { seriesName: string; data: [number, number]; color: string }[]) => {
        const dim = params[0]?.data[0];
        const head = `<strong>Dimension ${dim}</strong><br/>`;
        const rows = params.map((p) => {
          const pts = seriesByEngine.value.get(p.seriesName) ?? [];
          const pt = pts.find((x) => x.dim === dim);
          const note = pt?.notes ? ` <span style="opacity:.6">— ${pt.notes}</span>` : "";
          const src = pt?.source === "bench-host" ? "✓ measured on our runner" : `cite: ${pt?.source}`;
          return `<span style="color:${p.color}">●</span> ${p.seriesName}: <strong>${p.data[1].toLocaleString()}</strong> QPS <span style="opacity:.6">(${src})</span>${note}`;
        }).join("<br/>");
        return head + rows;
      },
    },
    xAxis: {
      type: "log" as const,
      name: "Embedding dimension",
      nameLocation: "middle" as const,
      nameGap: 30,
      data: [128, 200, 768, 1024, 1536, 3072],
      axisLabel: { formatter: (v: number) => `${v}d` },
    },
    yAxis: {
      type: "log" as const,
      name: `QPS at recall ≥ ${recallTarget.value}`,
      nameLocation: "middle" as const,
      nameGap: 50,
      axisLabel: { formatter: (v: number) => v.toLocaleString() },
    },
    series,
  };
});

const ramChartOption = computed(() => {
  const series = [...seriesByEngine.value.entries()].map(([engine, points]) => {
    const withRam = points.filter((p) => p.ram !== null);
    return {
      name: engine,
      type: "line" as const,
      data: withRam.map((p) => [p.dim, p.ram as number]),
      showSymbol: true,
      symbolSize: 8,
      lineStyle: { color: lineColor(engine), width: engine.startsWith("CoordiNode") ? 3 : 2 },
      itemStyle: { color: lineColor(engine) },
      emphasis: { focus: "series" as const },
    };
  });
  return {
    grid: { left: 60, right: 30, top: 30, bottom: 50 },
    legend: { type: "scroll" as const, top: 0, textStyle: { color: "var(--vp-c-text-1)" } },
    tooltip: {
      trigger: "axis" as const,
      axisPointer: { type: "cross" as const },
    },
    xAxis: {
      type: "log" as const,
      name: "Embedding dimension",
      nameLocation: "middle" as const,
      nameGap: 30,
      axisLabel: { formatter: (v: number) => `${v}d` },
    },
    yAxis: {
      type: "log" as const,
      name: "RAM (MB) per 1M vectors",
      nameLocation: "middle" as const,
      nameGap: 50,
      axisLabel: { formatter: (v: number) => v.toLocaleString() },
    },
    series,
  };
});
</script>

<template>
  <div class="bench-bydim">
    <!-- Pick-your-dimension prompt - the moment the chart should answer "is CoordiNode fast at MY dim" -->
    <p class="prompt">
      Pick your embedding model's dimension, then read off the chart. Higher is better.
    </p>

    <!-- Controls -->
    <div class="controls">
      <div class="ctrl-group">
        <span class="ctrl-label">Recall target:</span>
        <button
          v-for="r in [0.9, 0.95, 0.99]"
          :key="r"
          :class="['chip', recallTarget === r ? 'on' : 'off']"
          @click="recallTarget = r as 0.9 | 0.95 | 0.99"
        >
          ≥ {{ r }}
        </button>
      </div>
      <div class="ctrl-group">
        <span class="ctrl-label">Scale:</span>
        <button
          v-for="s in [1_000_000]"
          :key="s"
          :class="['chip', scale === s ? 'on' : 'off']"
          @click="scale = s"
        >
          {{ s.toLocaleString() }} vectors
        </button>
        <span class="muted">(10K / 100K / 10M sweeps land with R868 harness)</span>
      </div>
    </div>

    <!-- Primary: QPS chart -->
    <h3>QPS at recall ≥ {{ recallTarget }} across dimensions</h3>
    <v-chart class="echart" :option="qpsChartOption" autoresize />

    <!-- Secondary: RAM chart -->
    <h3>RAM cost across dimensions (per 1M vectors)</h3>
    <v-chart class="echart" :option="ramChartOption" autoresize />

    <!-- Engine legend - click row to toggle on/off across both charts -->
    <h4 class="legend-title">Engines <span class="legend-hint">— click row to toggle</span></h4>
    <table class="engine-legend">
      <thead>
        <tr>
          <th class="col-toggle">On</th>
          <th>Engine</th>
          <th>Status</th>
          <th>Data source</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="e in allEngines"
          :key="e"
          :class="['legend-row', enabledEngines.has(e) ? 'is-on' : 'is-off', isCoordinode(e) ? 'is-coordinode' : '']"
          @click="toggleEngine(e)"
        >
          <td class="col-toggle">
            <span :class="['toggle-dot', enabledEngines.has(e) ? 'on' : 'off']">
              {{ enabledEngines.has(e) ? "●" : "○" }}
            </span>
          </td>
          <td class="engine-name">{{ e }}</td>
          <td>
            <span v-if="isCoordinode(e)" class="status-pill status-current">measured on our runner</span>
            <span v-else class="status-pill status-competitor">cited baseline</span>
          </td>
          <td class="source">
            <span v-if="isCoordinode(e)">bench-host (i9-9900K)</span>
            <span v-else>ann-benchmarks.com / VectorDBBench (per arch/benchmarks/methodology.md)</span>
          </td>
        </tr>
      </tbody>
    </table>

    <p class="footnote">
      <strong>Methodology.</strong> The full benchmark spec lives in
      <code>arch/benchmarks/methodology.md</code>. CoordiNode bars come from measured runs
      on our bench host (Intel i9-9900K, 8C/16T, 64 GB RAM, Fedora). Competitor bars cite
      ann-benchmarks.com (Level A: SIFT1M / GloVe / GIST / NYTimes) and VectorDBBench
      (Level B: Cohere-768 / OpenAI-1536). We do not invent or extrapolate; gaps in the
      chart are real gaps — they fill in as the corresponding bench run lands.
    </p>
    <p class="footnote">
      <strong>RAM column status.</strong> Neither ann-benchmarks nor VDBBench publish
      per-engine RSS in their leaderboard output, and our own harness records timing +
      recall only. The RAM chart will populate as part of R868 once (a) our harness
      samples RSS during the build phase, and (b) we re-run competitors in our own
      Docker harness with RSS sampling enabled. Until then, the "MB per 1M vectors"
      panel is empty — by design, not omission.
    </p>
    <p class="footnote">
      <strong>Codec disclosure.</strong> CoordiNode SIFT1M (current main, f763f86) runs
      <code>codec=none</code> (raw f32) at <code>M=32, ef_construction=200</code>, single
      thread. hnswlib reference uses its default <code>M=16</code>, also single-thread,
      per ann-benchmarks.com configuration. Higher M means a larger graph but better
      recall headroom — comparability lives in the published recall target (≥ 0.95).
    </p>
  </div>
</template>

<style scoped>
.bench-bydim {
  margin: 1.5rem 0 2rem;
}
.prompt {
  font-size: 1rem;
  color: var(--vp-c-text-2);
  margin-bottom: 1rem;
}
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 1.5rem;
  padding: 0.75rem 0;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid var(--vp-c-divider);
  font-size: 0.9rem;
}
.ctrl-group {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
}
.ctrl-label {
  color: var(--vp-c-text-2);
  font-weight: 500;
}
.chip {
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  padding: 0.25rem 0.7rem;
  border-radius: 999px;
  cursor: pointer;
  font: inherit;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
}
.chip:hover {
  border-color: var(--vp-c-brand-1);
}
.chip.on {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
}
.chip.off {
  opacity: 0.7;
}
.muted {
  color: var(--vp-c-text-3);
  font-size: 0.85rem;
}
.echart {
  height: 380px;
  width: 100%;
  margin-bottom: 1rem;
}
h3 {
  margin-top: 1.5rem;
  font-size: 1.05rem;
}
.legend-title {
  margin-top: 2rem;
  font-size: 1rem;
  color: var(--vp-c-text-1);
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
}
.legend-hint {
  font-size: 0.85rem;
  font-weight: 400;
  color: var(--vp-c-text-2);
}
.engine-legend {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
  margin-top: 0.5rem;
}
.engine-legend th,
.engine-legend td {
  text-align: left;
  padding: 0.45rem 0.7rem;
  border-bottom: 1px solid var(--vp-c-divider);
}
.engine-legend th {
  font-weight: 600;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-soft);
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.col-toggle {
  width: 3rem;
  text-align: center;
  font-size: 1.05rem;
}
.legend-row {
  cursor: pointer;
  transition: background-color 0.12s;
}
.legend-row:hover {
  background: var(--vp-c-bg-soft);
}
.legend-row.is-off {
  opacity: 0.55;
}
.legend-row.is-coordinode .engine-name {
  font-weight: 600;
  color: var(--vp-c-brand-1);
}
.toggle-dot {
  display: inline-block;
  width: 1.1rem;
  text-align: center;
  font-weight: 700;
}
.toggle-dot.on {
  color: var(--vp-c-brand-1);
}
.toggle-dot.off {
  color: var(--vp-c-text-3);
}
.status-pill {
  display: inline-block;
  font-size: 0.78rem;
  font-weight: 500;
  padding: 0.12rem 0.5rem;
  border-radius: 10px;
  white-space: nowrap;
}
.status-pill.status-current {
  background: rgba(46, 160, 67, 0.16);
  color: rgb(63, 185, 80);
}
.status-pill.status-pending {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-3);
  border: 1px dashed var(--vp-c-divider);
}
.status-pill.status-competitor {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
}
.source {
  color: var(--vp-c-text-2);
  font-size: 0.85rem;
}
.footnote {
  margin-top: 1.5rem;
  padding: 0.75rem 1rem;
  border-left: 3px solid var(--vp-c-brand-1);
  background: var(--vp-c-bg-soft);
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
  line-height: 1.55;
}
.engine-name {
  font-weight: 500;
}
</style>
