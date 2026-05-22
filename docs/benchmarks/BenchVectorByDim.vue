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

// Data registry — ONLY numbers measured on our bench host.
//
// Hard rule: no cited / drawn / vendor-blog / leaderboard numbers. If a row
// doesn't come from a JSON we wrote on <redacted>, it doesn't go in DATA.
// Engines we plan to bench appear in the legend with status "running on
// bench host" but contribute zero datapoints to the chart until their
// measured JSON is committed to the bench-data branch.
//
// The single CoordiNode SIFT1M row below is the only measurement that
// currently exists (commit f763f86, codec=none, M=32, ef_construction=200,
// single thread). All other engines render as empty series with a status
// pill linking to the ROADMAP for when their bench lands.
const DATA: Datapoint[] = [
  { engine: "CoordiNode (current main)", dataset: "sift-128-euclidean", dim: 128, scale: 1_000_000, recall: 0.95, qps: 1317, ram_mb_per_1m: null, source: "bench-host", notes: "f763f86 single-thread, codec=none, M=32, ef_construction=200" },
];

// Engines we will bench on our own runner via the ann-benchmarks Docker
// harness on ro (<redacted>). The legend shows these as "pending bench
// run" — no chart line until the measured JSON lands.
const PENDING_ENGINES: { name: string; suite: string }[] = [
  { name: "hnswlib", suite: "ann-benchmarks" },
  { name: "FAISS-HNSW", suite: "ann-benchmarks" },
  { name: "ScaNN", suite: "ann-benchmarks" },
  { name: "Annoy", suite: "ann-benchmarks" },
  { name: "pgvector (HNSW)", suite: "ann-benchmarks + VDBBench" },
  { name: "Qdrant", suite: "ann-benchmarks + VDBBench" },
  { name: "Milvus", suite: "ann-benchmarks + VDBBench" },
  { name: "Weaviate", suite: "ann-benchmarks + VDBBench" },
  { name: "Elasticsearch", suite: "ann-benchmarks + VDBBench" },
  { name: "OpenSearch", suite: "VDBBench" },
  { name: "SurrealDB 3.x", suite: "VDBBench" },
  { name: "ArangoDB 3.12+", suite: "VDBBench" },
  { name: "MongoDB 8.x", suite: "VDBBench" },
];

// Controls
const recallTarget = ref<0.9 | 0.95 | 0.99>(0.95);
const scale = ref<number>(1_000_000);

// Compute engines list (stable order: CoordiNode first, then measured competitors,
// then pending competitors). Pending engines render in the legend but have no
// line on the chart until measured JSON arrives.
const measuredEngines = computed(() => {
  const set = new Set<string>(DATA.map((d) => d.engine));
  const cn = [...set].filter((e) => e.startsWith("CoordiNode")).sort();
  const others = [...set].filter((e) => !e.startsWith("CoordiNode")).sort();
  return [...cn, ...others];
});
const pendingEngineNames = computed(() => PENDING_ENGINES.map((p) => p.name).filter((n) => !measuredEngines.value.includes(n)));
const allEngines = computed(() => [...measuredEngines.value, ...pendingEngineNames.value]);
function isPending(e: string): boolean {
  return pendingEngineNames.value.includes(e);
}
function suiteFor(e: string): string {
  return PENDING_ENGINES.find((p) => p.name === e)?.suite ?? "—";
}
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
      // Dataset ladder we run on the bench host (ann-benchmarks dims + VDBBench).
      data: [25, 50, 100, 128, 200, 256, 768, 784, 960, 1536],
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
            <span v-else-if="isPending(e)" class="status-pill status-pending">pending bench run</span>
            <span v-else class="status-pill status-competitor">measured (one-shot)</span>
          </td>
          <td class="source">
            <span v-if="isCoordinode(e)">bench-host (i9-9900K) — per-commit timeline</span>
            <span v-else-if="isPending(e)">{{ suiteFor(e) }} on bench-host — one-shot snapshot</span>
            <span v-else>bench-host (i9-9900K) — one-shot snapshot</span>
          </td>
        </tr>
      </tbody>
    </table>

    <p class="footnote">
      <strong>What's on this chart.</strong> Only datapoints measured on our bench host
      (Intel i9-9900K, 8C/16T, 64 GB RAM, Fedora 44) appear here. No vendor-blog numbers,
      no leaderboard citations, no projections. Right now the chart has exactly one row:
      CoordiNode at SIFT1M, recall ≥ 0.95. Every other engine renders as
      <em>pending bench run</em> in the legend below until the corresponding ann-benchmarks
      / VectorDBBench Docker run lands a JSON in the
      <code>bench-data</code> branch on our self-hosted runner.
    </p>
    <p class="footnote">
      <strong>How competitor numbers will get here.</strong> The canonical
      <a href="https://github.com/erikbern/ann-benchmarks">ann-benchmarks</a> Docker harness
      runs uniform build+query for hnswlib / FAISS / ScaNN / Annoy / pgvector / Qdrant /
      Milvus / Weaviate / Elasticsearch / OpenSearch on the public dataset ladder
      (glove-25/50/100/200, sift-128, nytimes-256, fashion-mnist-784, gist-960). We run it
      once per engine on the same bench host, record QPS @ recall ≥ 0.95 and ≥ 0.99,
      build time, and index RSS, then commit the JSON to <code>bench-data</code>.
      Competitor numbers are <em>one-shot snapshots</em> — they don't move per-commit.
      Only the CoordiNode line on the "Engineering timeline" tab moves on every push to
      <code>main</code>.
    </p>
    <p class="footnote">
      <strong>CoordiNode run config.</strong> SIFT1M (commit f763f86) ran
      <code>codec=none</code> (raw f32), <code>M=32, ef_construction=200</code>, single
      thread. Multi-dataset coverage (glove / nytimes / gist / fashion-mnist) and RSS
      sampling are the in-flight half of R868.
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
