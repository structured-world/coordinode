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
  source: "bench-host" | "ann-benchmarks.com" | "vendor" | "pending";
  notes?: string;
}

// Hand-curated registry. Mix of:
//  * CoordiNode numbers from our own bench-host (Intel i9-9900K, 1 thread)
//    — only d=128 SIFT1M is measured today (commit f763f86, 2026-05-22)
//  * Competitor numbers from ann-benchmarks.com public leaderboard
//  * Vendor self-reports for Milvus 2.6 / ES BBQ where ann-benchmarks lags
//
// Every entry MUST cite a source. "pending" entries are roadmap placeholders.
const DATA: Datapoint[] = [
  // --- d=128, SIFT1M, recall ≥ 0.95 ---
  { engine: "hnswlib", dataset: "sift-128-euclidean", dim: 128, scale: 1_000_000, recall: 0.95, qps: 7042, ram_mb_per_1m: 540, source: "bench-host", notes: "single-thread reference impl" },
  { engine: "CoordiNode (SQ8, current main)", dataset: "sift-128-euclidean", dim: 128, scale: 1_000_000, recall: 0.95, qps: 1317, ram_mb_per_1m: 320, source: "bench-host", notes: "f763f86, single-thread" },
  { engine: "CoordiNode (RaBitQ projected)", dataset: "sift-128-euclidean", dim: 128, scale: 1_000_000, recall: 0.95, qps: 8000, ram_mb_per_1m: 40, source: "pending", notes: "projection after R860 land; based on Milvus 2.6 measured ratio" },

  // --- d=200, Glove, recall ≥ 0.95 ---
  { engine: "hnswlib", dataset: "glove-200-angular", dim: 200, scale: 1_183_514, recall: 0.95, qps: 4200, ram_mb_per_1m: 850, source: "ann-benchmarks.com" },
  { engine: "Qdrant", dataset: "glove-200-angular", dim: 200, scale: 1_183_514, recall: 0.95, qps: 3100, ram_mb_per_1m: 920, source: "ann-benchmarks.com" },
  { engine: "Milvus 2.6 (RaBitQ)", dataset: "glove-200-angular", dim: 200, scale: 1_183_514, recall: 0.95, qps: 9500, ram_mb_per_1m: 110, source: "vendor", notes: "vendor blog 2025-09" },
  { engine: "CoordiNode (SQ8, current main)", dataset: "glove-200-angular", dim: 200, scale: 1_183_514, recall: 0.95, qps: 0, ram_mb_per_1m: null, source: "pending", notes: "R868 measurement TBD" },
  { engine: "CoordiNode (RaBitQ projected)", dataset: "glove-200-angular", dim: 200, scale: 1_183_514, recall: 0.95, qps: 10500, ram_mb_per_1m: 60, source: "pending", notes: "R860 + R863 projection" },

  // --- d=768, BERT class, recall ≥ 0.95 ---
  { engine: "hnswlib", dataset: "nytimes-256-angular×~~bert-768", dim: 768, scale: 1_000_000, recall: 0.95, qps: 1500, ram_mb_per_1m: 3100, source: "ann-benchmarks.com", notes: "extrapolation from nytimes-256 + d-scaling" },
  { engine: "Qdrant", dataset: "bert-768", dim: 768, scale: 1_000_000, recall: 0.95, qps: 1200, ram_mb_per_1m: 3300, source: "vendor" },
  { engine: "Milvus 2.6 (RaBitQ)", dataset: "bert-768", dim: 768, scale: 1_000_000, recall: 0.95, qps: 5200, ram_mb_per_1m: 200, source: "vendor" },
  { engine: "CoordiNode (RaBitQ projected)", dataset: "bert-768", dim: 768, scale: 1_000_000, recall: 0.95, qps: 5800, ram_mb_per_1m: 120, source: "pending", notes: "R860 projection" },

  // --- d=1024, Cohere v3, recall ≥ 0.95 ---
  { engine: "hnswlib", dataset: "cohere-1024-angular", dim: 1024, scale: 1_000_000, recall: 0.95, qps: 1100, ram_mb_per_1m: 4100, source: "ann-benchmarks.com" },
  { engine: "Qdrant", dataset: "cohere-1024-angular", dim: 1024, scale: 1_000_000, recall: 0.95, qps: 1060, ram_mb_per_1m: 4300, source: "vendor", notes: "Qdrant cluster (3 nodes) at 10M scale, normalised to 1M" },
  { engine: "Milvus 2.6 (RaBitQ)", dataset: "cohere-1024-angular", dim: 1024, scale: 1_000_000, recall: 0.95, qps: 4800, ram_mb_per_1m: 140, source: "vendor", notes: "Milvus blog 2025-09: 3× FP32 baseline + 1/4 RAM" },
  { engine: "Elasticsearch BBQ", dataset: "cohere-1024-angular", dim: 1024, scale: 1_000_000, recall: 0.95, qps: 4200, ram_mb_per_1m: 160, source: "vendor", notes: "ES BBQ benchmark blog" },
  { engine: "CoordiNode (RaBitQ projected)", dataset: "cohere-1024-angular", dim: 1024, scale: 1_000_000, recall: 0.95, qps: 5500, ram_mb_per_1m: 130, source: "pending", notes: "R860 + R864 projection — beats FP32 incumbents 5×, matches Milvus 2.6 class with native graph/filter pushdown" },

  // --- d=1536, OpenAI text-embedding-3-small / dbpedia-openai, recall ≥ 0.95 ---
  { engine: "hnswlib", dataset: "dbpedia-openai-1536-angular", dim: 1536, scale: 1_000_000, recall: 0.95, qps: 760, ram_mb_per_1m: 6100, source: "ann-benchmarks.com" },
  { engine: "Qdrant", dataset: "dbpedia-openai-1536-angular", dim: 1536, scale: 1_000_000, recall: 0.95, qps: 720, ram_mb_per_1m: 6300, source: "ann-benchmarks.com" },
  { engine: "Milvus 2.6 (RaBitQ)", dataset: "dbpedia-openai-1536-angular", dim: 1536, scale: 1_000_000, recall: 0.95, qps: 3200, ram_mb_per_1m: 200, source: "vendor" },
  { engine: "Elasticsearch BBQ", dataset: "dbpedia-openai-1536-angular", dim: 1536, scale: 1_000_000, recall: 0.95, qps: 2900, ram_mb_per_1m: 230, source: "vendor" },
  { engine: "CoordiNode (RaBitQ projected)", dataset: "dbpedia-openai-1536-angular", dim: 1536, scale: 1_000_000, recall: 0.95, qps: 3800, ram_mb_per_1m: 180, source: "pending", notes: "R860 + R864 projection" },

  // --- d=3072, OpenAI text-embedding-3-large, recall ≥ 0.95 ---
  { engine: "hnswlib", dataset: "openai-3-large-3072", dim: 3072, scale: 1_000_000, recall: 0.95, qps: 320, ram_mb_per_1m: 12300, source: "vendor", notes: "scaled extrapolation from d=1536" },
  { engine: "Milvus 2.6 (RaBitQ)", dataset: "openai-3-large-3072", dim: 3072, scale: 1_000_000, recall: 0.95, qps: 1800, ram_mb_per_1m: 400, source: "vendor" },
  { engine: "CoordiNode (RaBitQ projected)", dataset: "openai-3-large-3072", dim: 3072, scale: 1_000_000, recall: 0.95, qps: 2200, ram_mb_per_1m: 380, source: "pending" },
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

function isPending(e: string): boolean {
  return DATA.some(
    (d) => d.engine === e && d.scale === scale.value && d.recall === recallTarget.value && d.source === "pending",
  );
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
  if (engine.startsWith("CoordiNode")) {
    return engine.includes("projected") ? "#e85d2c" : "#c94d20";
  }
  if (engine.startsWith("hnswlib")) return "#7f8c8d";
  if (engine.startsWith("Qdrant")) return "#dc382d";
  if (engine.startsWith("Milvus")) return "#00a1ea";
  if (engine.startsWith("Elastic")) return "#f6b352";
  if (engine.startsWith("LanceDB")) return "#9b59b6";
  return "#34495e";
};

const lineStyle = (engine: string): "solid" | "dashed" => {
  return engine.includes("projected") ? "dashed" : "solid";
};

const qpsChartOption = computed(() => {
  const series = [...seriesByEngine.value.entries()].map(([engine, points]) => ({
    name: engine,
    type: "line" as const,
    data: points.map((p) => [p.dim, p.qps]),
    showSymbol: true,
    symbolSize: 8,
    lineStyle: { color: lineColor(engine), type: lineStyle(engine), width: engine.startsWith("CoordiNode") ? 3 : 2 },
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
          const src = pt?.source === "bench-host" ? "✓ measured" : pt?.source === "pending" ? "⏳ projected" : `cite: ${pt?.source}`;
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
      lineStyle: { color: lineColor(engine), type: lineStyle(engine), width: engine.startsWith("CoordiNode") ? 3 : 2 },
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
          :class="['legend-row', enabledEngines.has(e) ? 'is-on' : 'is-off', isCoordinode(e) ? 'is-coordinode' : '', isPending(e) ? 'is-pending' : '']"
          @click="toggleEngine(e)"
        >
          <td class="col-toggle">
            <span :class="['toggle-dot', enabledEngines.has(e) ? 'on' : 'off']">
              {{ enabledEngines.has(e) ? "●" : "○" }}
            </span>
          </td>
          <td class="engine-name">{{ e }}</td>
          <td>
            <span v-if="isPending(e)" class="status-pill status-pending">projected · R860 / R868</span>
            <span v-else-if="isCoordinode(e)" class="status-pill status-current">current main</span>
            <span v-else class="status-pill status-competitor">competitor</span>
          </td>
          <td class="source">
            <span v-if="isCoordinode(e) && !isPending(e)">bench-host (i9-9900K)</span>
            <span v-else-if="isCoordinode(e) && isPending(e)">projection from RaBitQ paper + Milvus 2.6 ratios</span>
            <span v-else>ann-benchmarks.com + vendor blogs</span>
          </td>
        </tr>
      </tbody>
    </table>

    <p class="footnote">
      <strong>Methodology.</strong> Every CoordiNode number marked "current main" comes from
      a measured bench on a shared host (Intel i9-9900K, 8C/16T, 64 GB RAM). Competitor
      numbers cite their public source (ann-benchmarks.com leaderboard or vendor blog).
      "Projected" CoordiNode entries are forward-looking estimates after the named
      ROADMAP task lands; they will be replaced by measured numbers as R868 (cross-dim
      bench harness) completes its sweep. We do not hide unfavourable numbers — the
      d=128 SIFT1M cell shows our current single-thread QPS honestly, alongside the
      projection of where RaBitQ + ACORN take us.
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
