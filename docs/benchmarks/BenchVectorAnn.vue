<script setup lang="ts">
// SIFT1M ANN benchmark dashboard. Apache ECharts via vue-echarts
// — picked over Vega-Lite because:
//   • Legend toggle per engine works out of the box (click engine
//     name to hide/show its line — every chart on the page).
//   • Axis scale switch (linear ↔ log) is a one-line config change
//     we expose via a UI toggle.
//   • dataZoom slider lets the user brush a commit-range window on
//     the timeline chart to inspect drift between releases.
//   • Native dark-mode theme — auto-syncs with VitePress.
//   • Animated transitions between data updates.
//
// All three charts read the same in-memory `reports[]` array;
// recomputed reactively when the user clicks a control.
import { computed, onMounted, ref, watch } from "vue";
import VChart from "vue-echarts";
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import {
  LineChart,
  ScatterChart,
  BarChart,
} from "echarts/charts";
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  DataZoomComponent,
  MarkLineComponent,
  ToolboxComponent,
} from "echarts/components";

use([
  CanvasRenderer,
  LineChart,
  ScatterChart,
  BarChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  DataZoomComponent,
  MarkLineComponent,
  ToolboxComponent,
]);

const DATASET = "sift-128-euclidean";

interface ManifestEntry {
  path: string;
  modality: string;
  dataset: string;
  subject: string;
  sha?: string;
  sha_short?: string;
  timestamp?: string;
}

interface Manifest {
  schema_version: number;
  entries: ManifestEntry[];
}

interface SweepPoint {
  ef_search: number;
  recall_at_k: number;
  qps: number;
  latency_us_p50: number;
  latency_us_p95: number;
  latency_us_p99: number;
  latency_us_mean: number;
}

interface BenchReport {
  timestamp: string;
  git: { sha: string; sha_short: string; commit_date: string };
  subject: string;
  modality: string;
  dataset: string;
  hardware: { cpu_brand: string; cpu_cores: number; ram_gb: number };
  metrics: {
    sweep?: SweepPoint[];
    recall_at_k_peak?: number;
    qps_at_recall_peak?: number;
    qps_at_recall_0_95?: number;
    build_secs?: number;
    dataset_n_train?: number;
  };
}

const status = ref<"loading" | "empty" | "ok" | "error">("loading");
const errorMsg = ref<string>("");
const reports = ref<BenchReport[]>([]);

// Engine roster for the clickable legend table under the charts.
// Click any row to toggle that engine's lines on/off across every chart.
// Kept in sync with `ModalityTabs.vue` Vector competitors — same source-of-
// truth manifest, but the table lives in this component because clicking
// must toggle visibility on charts that BenchVectorAnn owns.
interface EngineMeta {
  name: string;
  kind: "coordinode" | "specialist" | "multi-model";
  // QPS is only comparable WITHIN a transport tier: `embedded` engines run
  // in-process (no network), `server` engines answer each query over the wire
  // (gRPC / HTTP round-trip). recall@ef is comparable across tiers; QPS is not.
  transport: "embedded" | "server";
  license: string;
  status: "live" | "planned";
  note?: string;
}
const ENGINES: EngineMeta[] = [
  { name: "coordinode", kind: "coordinode", transport: "embedded", license: "AGPL-3.0", status: "live", note: "in-process engine" },
  { name: "coordinode-grpc", kind: "coordinode", transport: "server", license: "AGPL-3.0", status: "live", note: "same engine, served over gRPC" },
  { name: "hnswlib", kind: "specialist", transport: "embedded", license: "Apache-2.0", status: "live", note: "reference implementation" },
  { name: "chromadb", kind: "specialist", transport: "embedded", license: "Apache-2.0", status: "live", note: "hnswlib backend" },
  { name: "qdrant", kind: "specialist", transport: "server", license: "Apache-2.0", status: "live", note: "multi-segment search" },
  { name: "Faiss (HNSW)", kind: "specialist", transport: "embedded", license: "MIT", status: "planned" },
  { name: "Milvus", kind: "specialist", transport: "server", license: "Apache-2.0", status: "planned" },
  { name: "pgvector", kind: "multi-model", transport: "server", license: "PostgreSQL", status: "planned" },
  { name: "MongoDB Atlas Vector", kind: "multi-model", transport: "server", license: "SSPL", status: "planned" },
];

// UI state
const qpsLogScale = ref<boolean>(true);
const yMetric = ref<"qps" | "latency_us_p99">("qps");
const enabledSubjects = ref<Set<string>>(new Set());

// Ratio chart UI state. Recall on its own is not a speedup metric —
// it's the accuracy floor at which we measure speed. The canonical
// ann-benchmarks operating point pins recall ≥ 0.95 and reports
// throughput / latency there. Both metrics below are speed-at-quality.
type RatioMetric = "qps_at_0_95" | "p99_at_0_95";
const ratioMetric = ref<RatioMetric>("qps_at_0_95");
// Which specialists are visible as ratio lines. CoordiNode is always the
// numerator and never appears in this set; only competitor subjects.
const ratioBaselines = ref<Set<string>>(new Set());

const COORDINODE = "coordinode";
const competitors = computed<string[]>(() =>
  subjects.value.filter((s) => s.toLowerCase() !== COORDINODE),
);

watch(competitors, (list) => {
  if (ratioBaselines.value.size === 0 && list.length > 0) {
    ratioBaselines.value = new Set(list);
  }
});

function toggleBaseline(subject: string) {
  if (ratioBaselines.value.has(subject)) {
    ratioBaselines.value.delete(subject);
  } else {
    ratioBaselines.value.add(subject);
  }
  ratioBaselines.value = new Set(ratioBaselines.value);
}

// Look up a single scalar metric on a report; for sweep-style metrics
// (p99 at the 0.95 recall point) walk the sweep array.
function metricValue(rep: BenchReport, m: RatioMetric): number | null {
  if (m === "qps_at_0_95") {
    return typeof rep.metrics.qps_at_recall_0_95 === "number"
      ? rep.metrics.qps_at_recall_0_95
      : null;
  }
  // p99_at_0_95: pick the sweep point closest to recall = 0.95.
  const sweep = rep.metrics.sweep ?? [];
  if (sweep.length === 0) return null;
  let best = sweep[0]!;
  let bestDelta = Math.abs(best.recall_at_k - 0.95);
  for (const p of sweep) {
    const d = Math.abs(p.recall_at_k - 0.95);
    if (d < bestDelta) {
      best = p;
      bestDelta = d;
    }
  }
  return best.latency_us_p99;
}

// Companion recall for a commit — shown in the speedup-chart tooltip
// next to the ratio so the reader can confirm the speed number was
// measured at the canonical recall ≥ 0.95 operating point (or see how
// far it drifted if a sweep point landed below).
function commitRecall(rep: BenchReport, m: RatioMetric): number | null {
  if (m === "qps_at_0_95") {
    // qps_at_recall_0_95 is by definition the QPS at the sweep point
    // whose recall ≥ 0.95; report 0.95 as the floor or the exact recall
    // of the matched sweep point when known.
    const sweep = rep.metrics.sweep ?? [];
    if (sweep.length === 0) return null;
    const ge = sweep.filter((p) => p.recall_at_k >= 0.95);
    if (ge.length === 0) return Math.max(...sweep.map((p) => p.recall_at_k));
    return Math.min(...ge.map((p) => p.recall_at_k));
  }
  // p99_at_0_95 — same sweep-point-nearest-0.95 logic.
  const sweep = rep.metrics.sweep ?? [];
  if (sweep.length === 0) return null;
  let best = sweep[0]!;
  let bestDelta = Math.abs(best.recall_at_k - 0.95);
  for (const p of sweep) {
    const d = Math.abs(p.recall_at_k - 0.95);
    if (d < bestDelta) {
      best = p;
      bestDelta = d;
    }
  }
  return best.recall_at_k;
}

// For QPS / recall, larger = better → ratio = CN / baseline.
// For p99 latency, smaller = better → ratio = baseline / CN.
// Both orientations make "> 1.0 means CoordiNode wins" hold true.
function ratioDirection(m: RatioMetric): "higher-better" | "lower-better" {
  return m === "p99_at_0_95" ? "lower-better" : "higher-better";
}

const subjects = computed<string[]>(() => {
  const s = new Set<string>();
  for (const r of reports.value) s.add(r.subject);
  return [...s].sort();
});

watch(subjects, (list) => {
  if (enabledSubjects.value.size === 0 && list.length > 0) {
    enabledSubjects.value = new Set(list);
  }
});

function toggleSubject(subject: string) {
  if (enabledSubjects.value.has(subject)) {
    enabledSubjects.value.delete(subject);
  } else {
    enabledSubjects.value.add(subject);
  }
  enabledSubjects.value = new Set(enabledSubjects.value);
}

const visibleReports = computed<BenchReport[]>(() =>
  reports.value.filter((r) => enabledSubjects.value.has(r.subject)),
);

async function loadManifest(): Promise<Manifest> {
  const base = (import.meta.env.BASE_URL ?? "/").replace(/\/+$/, "/");
  const url = `${base}bench-data/index.json`;
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) {
    throw new Error(
      `Could not load ${url} (HTTP ${r.status}). The docs build must run with bench-results in scope.`,
    );
  }
  return r.json();
}

async function loadReport(path: string): Promise<BenchReport> {
  const base = (import.meta.env.BASE_URL ?? "/").replace(/\/+$/, "/");
  const url = `${base}${path}`;
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

// Pareto frontier — Recall@10 vs QPS (or P99 latency). Each engine
// has one line through its sweep points, sorted by recall ascending.
const paretoOption = computed(() => {
  const subjectMap = new Map<string, { recall: number; y: number; ef: number; sha: string }[]>();
  for (const rep of visibleReports.value) {
    for (const p of rep.metrics.sweep ?? []) {
      const arr = subjectMap.get(rep.subject) ?? [];
      arr.push({
        recall: p.recall_at_k,
        y: yMetric.value === "qps" ? p.qps : p.latency_us_p99,
        ef: p.ef_search,
        sha: rep.git.sha_short,
      });
      subjectMap.set(rep.subject, arr);
    }
  }
  const series = [...subjectMap.entries()].map(([subject, points]) => ({
    name: subject,
    type: "line" as const,
    smooth: true,
    symbol: "circle",
    symbolSize: 9,
    data: points
      .sort((a, b) => a.recall - b.recall)
      .map((p) => [p.recall, p.y, p.ef, p.sha]),
    emphasis: { focus: "series" as const },
  }));
  const yIsLog = yMetric.value === "qps" && qpsLogScale.value;
  const yTitle = yMetric.value === "qps" ? "QPS (single-thread)" : "P99 latency (µs)";
  return {
    tooltip: {
      trigger: "item",
      formatter: (params: any) => {
        const [recall, y, ef, sha] = params.data as [number, number, number, string];
        const yLabel = yMetric.value === "qps" ? "QPS" : "P99 µs";
        return [
          `<b>${params.seriesName}</b> (${sha})`,
          `Recall @ 10: ${recall.toFixed(4)}`,
          `${yLabel}: ${y.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
          `ef_search: ${ef}`,
        ].join("<br>");
      },
    },
    legend: {
      type: "scroll",
      top: 0,
      textStyle: { color: "var(--vp-c-text-1)" },
    },
    grid: { top: 50, left: 60, right: 24, bottom: 56 },
    xAxis: {
      type: "value",
      name: "Recall @ 10 (higher → more accurate)",
      nameLocation: "middle",
      nameGap: 30,
      // ECharts auto-fits — never hard-coded bounds, otherwise data
      // outside the window silently disappears.
      scale: true,
    },
    yAxis: {
      type: yIsLog ? "log" : "value",
      name: yTitle,
      nameLocation: "middle",
      nameGap: 50,
    },
    series,
  };
});

// Speedup vs baselines — X is the CoordiNode commit timeline (one
// dot per CN commit), Y is the ratio (CN metric / competitor metric)
// for the selected metric.  Direction is normalised so that values
// above 1.0 always mean "CoordiNode wins" — for QPS/recall that's
// CN/baseline, for p99 latency that's baseline/CN.  One line per
// toggled competitor; competitor baselines are pinned (a single
// scalar per competitor) so each line is the CN curve scaled by the
// inverse of that competitor's pinned scalar.
const speedupOption = computed(() => {
  const m = ratioMetric.value;
  const dir = ratioDirection(m);

  // CoordiNode commits, sorted by commit time.
  const cnRuns = reports.value
    .filter((r) => r.subject.toLowerCase() === COORDINODE)
    .map((r) => ({
      ts: r.timestamp,
      sha: r.git.sha_short,
      value: metricValue(r, m),
      recall: commitRecall(r, m),
    }))
    .filter(
      (x): x is { ts: string; sha: string; value: number; recall: number | null } =>
        x.value !== null,
    )
    .sort((a, b) => a.ts.localeCompare(b.ts));

  // Pinned competitor scalar — there's exactly one JSON per competitor
  // in bench-results/ (replaced on re-bench, never timelined).  If
  // multiple files exist, pick the most recent.
  const baselineValue = new Map<string, number>();
  for (const r of reports.value) {
    if (r.subject.toLowerCase() === COORDINODE) continue;
    if (!ratioBaselines.value.has(r.subject)) continue;
    const v = metricValue(r, m);
    if (v === null) continue;
    const existing = baselineValue.get(r.subject);
    if (existing === undefined) {
      baselineValue.set(r.subject, v);
    }
  }

  // X category axis — commit short SHA, ordered left-to-right by time.
  const xCategories = cnRuns.map((r) => r.sha);

  const series = [...baselineValue.entries()].map(([competitor, baseValue]) => ({
    name: `vs ${competitor}`,
    type: "line" as const,
    smooth: false,
    symbol: "circle",
    symbolSize: 8,
    data: cnRuns.map((r) => {
      const ratio = dir === "higher-better" ? r.value / baseValue : baseValue / r.value;
      return [r.sha, ratio, r.ts, r.recall];
    }),
    emphasis: { focus: "series" as const },
  }));

  const metricLabel =
    m === "qps_at_0_95" ? "QPS @ recall ≥ 0.95" : "P99 latency @ recall ≥ 0.95";

  return {
    tooltip: {
      trigger: "item",
      formatter: (params: any) => {
        const [sha, ratio, ts, recall] = params.data as [
          string,
          number,
          string,
          number | null,
        ];
        const verb = ratio >= 1 ? "faster than" : "slower than";
        const factor = ratio >= 1 ? ratio.toFixed(2) : (1 / ratio).toFixed(2);
        const lines = [
          `<b>${params.seriesName}</b>`,
          `Commit <code>${sha}</code> — ${new Date(ts).toLocaleString()}`,
          `Ratio: ${ratio.toFixed(3)}× (${factor}× ${verb} baseline)`,
          `Metric: ${metricLabel}`,
        ];
        if (recall !== null) {
          lines.push(`Recall @ 10 at this point: ${recall.toFixed(4)}`);
        }
        return lines.join("<br>");
      },
    },
    legend: {
      type: "scroll",
      top: 0,
      textStyle: { color: "var(--vp-c-text-1)" },
    },
    grid: { top: 50, left: 60, right: 24, bottom: 60 },
    xAxis: {
      type: "category",
      data: xCategories,
      name: "CoordiNode commit (oldest → newest)",
      nameLocation: "middle",
      nameGap: 32,
      axisLabel: {
        rotate: xCategories.length > 12 ? 35 : 0,
        fontFamily: "var(--vp-font-family-mono)",
      },
    },
    yAxis: {
      type: "value",
      name: "Ratio (>1 = CoordiNode wins)",
      nameLocation: "middle",
      nameGap: 50,
    },
    series: series.map((s) => ({
      ...s,
      markLine: {
        symbol: "none",
        silent: true,
        lineStyle: { color: "var(--vp-c-text-3)", type: "dashed" },
        data: [{ yAxis: 1, label: { formatter: "parity (1.0×)" } }],
      },
    })),
  };
});

// Bar — QPS @ recall ≥ 0.95 (latest run per subject).
const barOption = computed(() => {
  const latestBySubject = new Map<string, BenchReport>();
  for (const r of visibleReports.value) {
    if (typeof r.metrics.qps_at_recall_0_95 !== "number") continue;
    const existing = latestBySubject.get(r.subject);
    if (!existing || r.timestamp > existing.timestamp) {
      latestBySubject.set(r.subject, r);
    }
  }
  const data = [...latestBySubject.values()]
    .map((r) => ({
      name: r.subject,
      value: r.metrics.qps_at_recall_0_95 as number,
      sha: r.git.sha_short,
    }))
    .sort((a, b) => b.value - a.value);
  return {
    tooltip: {
      trigger: "item",
      formatter: (params: any) => {
        const { name, value, data } = params;
        const sha = (data as { sha: string }).sha;
        return `<b>${name}</b> (${sha})<br>QPS @ recall ≥ 0.95: ${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
      },
    },
    grid: { top: 24, left: 60, right: 24, bottom: 50 },
    xAxis: {
      type: "category",
      data: data.map((d) => d.name),
      axisLabel: { interval: 0 },
    },
    yAxis: {
      type: "value",
      name: "QPS @ recall ≥ 0.95",
      nameLocation: "middle",
      nameGap: 50,
    },
    series: [
      {
        type: "bar",
        data: data,
        itemStyle: { borderRadius: 4 },
        emphasis: { focus: "series" },
        label: {
          show: true,
          position: "top",
          // ECharts defaults a contrasting text-border around bar labels;
          // in dark mode that border reads as a doubled / blurry outline
          // around white digits. Force the colour to follow the VitePress
          // text variable and kill the border so it stays a clean single
          // stroke in both themes.
          color: "var(--vp-c-text-1)",
          textBorderWidth: 0,
          textBorderColor: "transparent",
          formatter: (p: any) =>
            (p.data.value as number).toLocaleString(undefined, {
              maximumFractionDigits: 0,
            }),
        },
      },
    ],
  };
});

async function render() {
  try {
    const manifest = await loadManifest();
    const entries = manifest.entries.filter(
      (e) => e.modality === "vector" && e.dataset === DATASET,
    );
    if (entries.length === 0) {
      status.value = "empty";
      return;
    }
    const fetched: BenchReport[] = [];
    for (const e of entries) {
      try {
        fetched.push(await loadReport(e.path));
      } catch (err) {
        console.warn(`skipping ${e.path}: ${(err as Error).message}`);
      }
    }
    if (fetched.length === 0) {
      status.value = "empty";
      return;
    }
    reports.value = fetched;
    status.value = "ok";
  } catch (e) {
    status.value = "error";
    errorMsg.value = (e as Error).message;
  }
}

onMounted(render);
</script>

<template>
  <div class="bench-block">
    <div v-if="status === 'loading'" class="bench-status">
      Loading bench results…
    </div>
    <div v-else-if="status === 'empty'" class="bench-status">
      No SIFT1M results published yet. The first bench run will populate
      this view.
    </div>
    <div v-else-if="status === 'error'" class="bench-status bench-error">
      {{ errorMsg }}
    </div>

    <template v-if="status === 'ok'">
      <!-- Y-axis control row (engine toggles live in the legend table below) -->
      <div class="controls">
        <div class="control-group">
          <span class="control-label">Y-axis:</span>
          <button
            :class="['chip', yMetric === 'qps' ? 'on' : 'off']"
            @click="yMetric = 'qps'"
          >
            QPS
          </button>
          <button
            :class="['chip', yMetric === 'latency_us_p99' ? 'on' : 'off']"
            @click="yMetric = 'latency_us_p99'"
          >
            P99 latency
          </button>
          <button
            v-if="yMetric === 'qps'"
            :class="['chip', qpsLogScale ? 'on' : 'off']"
            @click="qpsLogScale = !qpsLogScale"
          >
            {{ qpsLogScale ? "log" : "linear" }}
          </button>
        </div>
      </div>

      <h3>Pareto frontier — Recall @ 10 vs {{ yMetric === "qps" ? "QPS" : "P99 latency" }}</h3>
      <v-chart class="echart" :option="paretoOption" autoresize />

      <h3>Speedup vs baselines — by commit</h3>
      <p class="hint">
        Each dot is one CoordiNode commit (X-axis = commits left → right
        by time). Y is the ratio of CoordiNode's metric to the pinned
        competitor's metric — normalised so <strong>&gt; 1.0 always
        means CoordiNode wins</strong>. Pick the metric to compare in
        the tabs, and toggle which competitor lines you want on the
        chart.
      </p>
      <div class="controls">
        <div class="control-group">
          <span class="control-label">Metric:</span>
          <button
            :class="['chip', ratioMetric === 'qps_at_0_95' ? 'on' : 'off']"
            @click="ratioMetric = 'qps_at_0_95'"
          >
            QPS @ recall ≥ 0.95
          </button>
          <button
            :class="['chip', ratioMetric === 'p99_at_0_95' ? 'on' : 'off']"
            @click="ratioMetric = 'p99_at_0_95'"
          >
            P99 latency @ recall ≥ 0.95
          </button>
        </div>
        <div class="control-group">
          <span class="control-label">Compare against:</span>
          <button
            v-for="c in competitors"
            :key="c"
            :class="['chip', ratioBaselines.has(c) ? 'on' : 'off']"
            @click="toggleBaseline(c)"
          >
            {{ c }}
          </button>
        </div>
      </div>
      <v-chart class="echart" :option="speedupOption" autoresize />

      <h3>QPS @ recall ≥ 0.95 — latest run per engine</h3>
      <v-chart class="echart bar" :option="barOption" autoresize />

      <!-- Clickable engine legend. Each row toggles that engine across
           every chart above. Status pill ('live' / 'planned') shows whether
           we have JSON yet; planned rows still toggle the chip but have no
           series to render. -->
      <h4 class="legend-title">Engines &amp; pinned versions <span class="legend-hint">— click row to toggle</span></h4>
      <table class="engine-legend">
        <thead>
          <tr>
            <th class="col-toggle">On</th>
            <th>Engine</th>
            <th>Type</th>
            <th>Transport</th>
            <th>License</th>
            <th>Status</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="e in ENGINES"
            :key="e.name"
            :class="['legend-row', enabledSubjects.has(e.name) ? 'is-on' : 'is-off', e.status === 'planned' ? 'is-planned' : '']"
            @click="e.status === 'live' && toggleSubject(e.name)"
          >
            <td class="col-toggle">
              <span :class="['toggle-dot', enabledSubjects.has(e.name) ? 'on' : 'off']" :title="enabledSubjects.has(e.name) ? 'visible — click to hide' : 'hidden — click to show'">
                {{ enabledSubjects.has(e.name) ? "●" : "○" }}
              </span>
            </td>
            <td class="engine-name">{{ e.name }}</td>
            <td>
              <span :class="['kind-pill', `kind-${e.kind}`]">
                {{ e.kind === "coordinode" ? "this engine" : e.kind }}
              </span>
            </td>
            <td>
              <span
                :class="['transport-pill', `transport-${e.transport}`]"
                :title="e.transport === 'server' ? 'answers each query over the wire (gRPC/HTTP); QPS comparable only to other server-tier engines' : 'runs in-process, no network; QPS comparable only to other embedded-tier engines'"
              >
                {{ e.transport }}
              </span>
            </td>
            <td><code>{{ e.license }}</code></td>
            <td>
              <span :class="['status-pill', `status-${e.status}`]">
                {{ e.status }}
              </span>
            </td>
            <td class="note-cell">{{ e.note ?? "—" }}</td>
          </tr>
        </tbody>
      </table>
    </template>
  </div>
</template>

<style scoped>
.bench-block {
  margin: 1.5rem 0;
}
.bench-status {
  padding: 1rem;
  background: var(--vp-c-bg-soft);
  border-radius: 6px;
  color: var(--vp-c-text-2);
  margin-bottom: 1.5rem;
}
.bench-error {
  color: var(--vp-c-danger-1);
  border: 1px solid var(--vp-c-danger-2);
}
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 1.5rem;
  padding: 0.75rem 0;
  margin-bottom: 1rem;
  border-bottom: 1px solid var(--vp-c-divider);
  font-size: 0.88rem;
}
.control-group {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
}
.control-label {
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
  opacity: 0.55;
}
.echart {
  width: 100%;
  height: 420px;
  margin: 0.5rem 0 2rem;
}
.echart.bar {
  height: 320px;
}
.hint {
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
  margin: 0 0 0.5rem;
}
h3 {
  margin-top: 2rem;
}

/* Clickable engine legend under the charts. Whole row is the toggle
 * surface — cursor:pointer + hover bg + a leading dot indicate state.
 * Planned engines are visibly dimmed but still toggle the chip (when
 * a JSON arrives the chart picks up the series automatically). */
.legend-title {
  margin-top: 2.5rem;
  font-size: 1rem;
  color: var(--vp-c-text-1);
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  flex-wrap: wrap;
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
.legend-row.is-planned {
  cursor: not-allowed;
  opacity: 0.45;
}
.legend-row.is-planned:hover {
  background: transparent;
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
.engine-name {
  font-weight: 500;
  color: var(--vp-c-text-1);
}
.kind-pill {
  display: inline-block;
  font-size: 0.78rem;
  font-weight: 500;
  padding: 0.12rem 0.5rem;
  border-radius: 10px;
  white-space: nowrap;
}
.kind-pill.kind-coordinode {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
}
.kind-pill.kind-specialist {
  background: rgba(120, 144, 156, 0.16);
  color: var(--vp-c-text-2);
}
.kind-pill.kind-multi-model {
  background: var(--vp-c-purple-soft, rgba(159, 122, 234, 0.16));
  color: var(--vp-c-purple-1, #9f7aea);
}
.status-pill {
  display: inline-block;
  font-size: 0.78rem;
  font-weight: 500;
  padding: 0.12rem 0.5rem;
  border-radius: 10px;
  white-space: nowrap;
}
.status-pill.status-live {
  background: rgba(46, 160, 67, 0.16);
  color: rgb(63, 185, 80);
}
.status-pill.status-planned {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-3);
  border: 1px dashed var(--vp-c-divider);
}
.transport-pill {
  display: inline-block;
  font-size: 0.78rem;
  font-weight: 500;
  padding: 0.12rem 0.5rem;
  border-radius: 10px;
  white-space: nowrap;
}
.transport-pill.transport-embedded {
  background: rgba(56, 139, 253, 0.16);
  color: var(--vp-c-brand-1, #388bfd);
}
.transport-pill.transport-server {
  background: rgba(219, 109, 40, 0.16);
  color: rgb(219, 109, 40);
}
.note-cell {
  color: var(--vp-c-text-2);
  font-size: 0.88rem;
}
</style>
