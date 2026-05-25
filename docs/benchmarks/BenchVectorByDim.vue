<script setup lang="ts">
// Product-facing vector benchmark view: "QPS at recall ≥ X across dimensions".
//
// A developer arrives with an embedding model (BERT-768, Cohere-v3 1024,
// text-embedding-3-large 3072) and wants to answer ONE question: "at MY
// dimension, is CoordiNode faster than $alternative?".  This component
// plots that — and nothing else.
//
// Data path:
//   /bench-data/index.json   ← manifest (paths + minimal metadata)
//      ↓ fetch each entry
//   /bench-data/bench-results/vector/<dataset>/<sha>-<subject>-M<m>-<ts>.json
//      ↓ collapse to single (subject, dataset, dim, qps@recall, ...) datapoint
//   chart.
//
// Hard rules (per repo policy):
//   * NO drawn / cited / leaderboard numbers — every chart point comes
//     from a JSON committed to the `bench-data` branch by our own runner.
//   * CoordiNode is per-commit (timeline view in BenchVectorAnn handles
//     that; here we pick ONE CoordiNode SHA to show — defaults to latest).
//   * Competitor engines are one-shot snapshots — most-recent JSON wins.
import { computed, onMounted, ref } from "vue";
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

// ── Types matching the canonical bench-results JSON schema (v1) ──────────────

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
  schema_version: 1;
  timestamp: string;
  git: { sha: string; sha_short: string; commit_date: string };
  modality: string;
  benchmark: string;
  dataset: string;
  subject: string;
  codec: string;
  version: string;
  metrics: {
    sweep?: SweepPoint[];
    hnsw_m?: number;
    hnsw_ef_construction?: number;
    dataset_dim?: number;
    dataset_n_train?: number;
    build_secs?: number;
    recall_at_k_peak?: number;
    qps_at_recall_peak?: number;
    k?: number;
  };
}

// Pending engines list: rendered in the legend with a "pending bench run"
// pill when we have no JSON for them yet.  Lets readers see what's in flight
// without us drawing numbers we don't have.
//
// Names MUST match the lowercase `subject` field of bench-results JSONs
// produced by ``scripts/ann-benchmarks-to-json.py`` — otherwise the
// "already measured, hide from pending" filter sees "Qdrant" ≠ "qdrant"
// and shows the engine in BOTH measured + pending lists.
const PENDING_ENGINES: { name: string; suite: string }[] = [
  { name: "hnswlib", suite: "ann-benchmarks" },
  { name: "faiss-hnsw", suite: "ann-benchmarks" },
  { name: "scann", suite: "ann-benchmarks" },
  { name: "annoy", suite: "ann-benchmarks" },
  { name: "pgvector", suite: "ann-benchmarks + VDBBench" },
  { name: "qdrant", suite: "ann-benchmarks + VDBBench" },
  { name: "milvus", suite: "ann-benchmarks + VDBBench" },
  { name: "weaviate", suite: "ann-benchmarks + VDBBench" },
  { name: "elasticsearch", suite: "ann-benchmarks + VDBBench" },
  { name: "opensearch", suite: "VDBBench" },
  { name: "surrealdb", suite: "VDBBench" },
  { name: "arangodb", suite: "VDBBench" },
  { name: "mongodb", suite: "VDBBench" },
];

// ── State ────────────────────────────────────────────────────────────────────

const status = ref<"loading" | "empty" | "ok" | "error">("loading");
const errorMsg = ref<string>("");
const reports = ref<BenchReport[]>([]);

// Controls
const recallTarget = ref<0.9 | 0.95 | 0.99>(0.95);
const selectedCoordinodeSha = ref<string>("");   // empty = latest

// ── Manifest + report loading ────────────────────────────────────────────────

async function loadManifest(): Promise<Manifest> {
  const base = (import.meta.env.BASE_URL ?? "/").replace(/\/+$/, "/");
  const url = `${base}bench-data/index.json`;
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) {
    throw new Error(`Manifest fetch failed: ${url} (HTTP ${r.status})`);
  }
  return r.json();
}

async function loadReport(path: string): Promise<BenchReport> {
  const base = (import.meta.env.BASE_URL ?? "/").replace(/\/+$/, "/");
  const r = await fetch(`${base}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

async function render(): Promise<void> {
  status.value = "loading";
  errorMsg.value = "";
  try {
    const manifest = await loadManifest();
    const vectorEntries = manifest.entries.filter((e) => e.modality === "vector");
    const fetched: BenchReport[] = [];
    for (const e of vectorEntries) {
      try {
        const r = await loadReport(e.path);
        if (r.schema_version === 1) fetched.push(r);
      } catch (err) {
        // Skip individual broken reports; surface count at the end.
        console.warn(`skipped ${e.path}:`, err);
      }
    }
    reports.value = fetched;
    status.value = fetched.length === 0 ? "empty" : "ok";
  } catch (err) {
    status.value = "error";
    errorMsg.value = err instanceof Error ? err.message : String(err);
  }
}

onMounted(render);

// ── Slicing the reports for the chart ────────────────────────────────────────

// CoordiNode SHAs sorted by commit timestamp (newest first) for the picker.
const coordinodeSHAs = computed(() => {
  const seen = new Map<string, string>();   // sha_short → commit timestamp
  for (const r of reports.value) {
    if (r.subject === "coordinode") {
      const ts = r.git.commit_date ?? r.timestamp;
      const prev = seen.get(r.git.sha_short);
      if (!prev || ts > prev) seen.set(r.git.sha_short, ts);
    }
  }
  return [...seen.entries()]
    .sort((a, b) => (a[1] > b[1] ? -1 : 1))
    .map(([sha]) => sha);
});

// Effective CoordiNode SHA — selected one, or latest if no selection / selection vanished.
const effectiveCoordinodeSha = computed(() => {
  const all = coordinodeSHAs.value;
  if (!all.length) return null;
  if (selectedCoordinodeSha.value && all.includes(selectedCoordinodeSha.value)) {
    return selectedCoordinodeSha.value;
  }
  return all[0];
});

/** Returns the operating-point QPS at recall ≥ target from a sweep,
 *  or null if no row clears the target.  Picks the highest-QPS row
 *  among those meeting the bar — the canonical operating point. */
function operatingQps(sweep: SweepPoint[] | undefined, target: number): number | null {
  if (!sweep) return null;
  const ok = sweep.filter((s) => s.recall_at_k >= target);
  if (!ok.length) return null;
  return Math.max(...ok.map((s) => s.qps));
}

interface ChartPoint {
  subject: string;
  dataset: string;
  dim: number;
  qps: number;
  m: number;
  sha_short?: string;
  is_coordinode: boolean;
}

// Build chart points by collapsing reports → one point per (subject, dataset).
// For each (subject, dataset) we KEEP the report with the highest operating-
// point QPS at the active recall target across M sweeps — that's the engine's
// best at that recall level.
const chartPoints = computed<ChartPoint[]>(() => {
  const cnSha = effectiveCoordinodeSha.value;
  const groups = new Map<string, BenchReport[]>();
  for (const r of reports.value) {
    if (r.subject === "coordinode" && r.git.sha_short !== cnSha) continue;
    const key = `${r.subject}::${r.dataset}`;
    const arr = groups.get(key) ?? [];
    arr.push(r);
    groups.set(key, arr);
  }
  const points: ChartPoint[] = [];
  for (const [, reps] of groups) {
    const best: { qps: number; rep: BenchReport } | null = reps.reduce<
      { qps: number; rep: BenchReport } | null
    >((acc, r) => {
      const q = operatingQps(r.metrics.sweep, recallTarget.value);
      if (q === null) return acc;
      if (acc === null || q > acc.qps) return { qps: q, rep: r };
      return acc;
    }, null);
    if (best === null) continue;
    const dim = best.rep.metrics.dataset_dim;
    if (dim === undefined || dim === null) continue;
    points.push({
      subject: best.rep.subject,
      dataset: best.rep.dataset,
      dim,
      qps: best.qps,
      m: best.rep.metrics.hnsw_m ?? 0,
      sha_short: best.rep.git.sha_short,
      is_coordinode: best.rep.subject === "coordinode",
    });
  }
  return points;
});

// Subjects seen in the data, plus any pending engines we don't have data for.
const measuredSubjects = computed(() => {
  const set = new Set<string>();
  for (const p of chartPoints.value) set.add(p.subject);
  return [...set];
});
const pendingSubjects = computed(() =>
  PENDING_ENGINES.map((p) => p.name).filter((n) => !measuredSubjects.value.includes(n))
);
const allSubjects = computed(() => {
  const cn = measuredSubjects.value.filter((s) => s === "coordinode");
  const others = measuredSubjects.value.filter((s) => s !== "coordinode").sort();
  return [...cn, ...others, ...pendingSubjects.value];
});

const enabledSubjects = ref<Set<string>>(new Set());
function toggleSubject(s: string): void {
  const next = new Set(enabledSubjects.value);
  if (next.has(s)) next.delete(s);
  else next.add(s);
  enabledSubjects.value = next;
}
// Auto-enable everything we have data for, the first time a render lands.
function autoEnable(): void {
  enabledSubjects.value = new Set(measuredSubjects.value);
}
onMounted(autoEnable);

function subjectColor(subject: string): string {
  if (subject === "coordinode") return "#c94d20";
  if (subject === "hnswlib") return "#7f8c8d";
  if (subject.startsWith("faiss")) return "#3b5998";
  if (subject === "scann") return "#34a853";
  if (subject === "annoy") return "#e8a33d";
  if (subject.startsWith("pgvector")) return "#336791";
  if (subject === "qdrant") return "#dc382d";
  if (subject === "milvus") return "#00a1ea";
  if (subject === "weaviate") return "#1a9e6e";
  if (subject.startsWith("elastic")) return "#f6b352";
  if (subject.startsWith("opensearch")) return "#005eb8";
  if (subject.startsWith("surreal")) return "#ff00a0";
  if (subject.startsWith("arango")) return "#5b9b3f";
  if (subject.startsWith("mongo")) return "#13aa52";
  return "#34495e";
}

function suiteFor(s: string): string {
  return PENDING_ENGINES.find((p) => p.name === s)?.suite ?? "—";
}

function isPending(s: string): boolean {
  return pendingSubjects.value.includes(s);
}

function displaySubject(s: string): string {
  // Subjects serialised as lowercase kebab in JSON but displayed with
  // canonical capitalisation.  Keep in sync with PENDING_ENGINES.name.
  const map: Record<string, string> = {
    coordinode: "CoordiNode",
    hnswlib: "hnswlib",
    "faiss-hnsw": "FAISS-HNSW",
    scann: "ScaNN",
    annoy: "Annoy",
    pgvector: "pgvector",
    qdrant: "Qdrant",
    milvus: "Milvus",
    weaviate: "Weaviate",
    elasticsearch: "Elasticsearch",
    opensearch: "OpenSearch",
    surrealdb: "SurrealDB",
    arangodb: "ArangoDB",
    mongodb: "MongoDB",
  };
  return map[s] ?? s;
}

// ── Chart options ────────────────────────────────────────────────────────────

const qpsChartOption = computed(() => {
  const bySubject = new Map<string, ChartPoint[]>();
  for (const p of chartPoints.value) {
    if (!enabledSubjects.value.has(p.subject)) continue;
    const arr = bySubject.get(p.subject) ?? [];
    arr.push(p);
    bySubject.set(p.subject, arr);
  }
  for (const arr of bySubject.values()) arr.sort((a, b) => a.dim - b.dim);

  const series = [...bySubject.entries()].map(([subject, points]) => ({
    name: displaySubject(subject),
    type: "line" as const,
    data: points.map((p) => [p.dim, p.qps]),
    showSymbol: true,
    symbolSize: 10,
    lineStyle: {
      color: subjectColor(subject),
      width: subject === "coordinode" ? 3 : 2,
    },
    itemStyle: { color: subjectColor(subject) },
    emphasis: { focus: "series" as const },
  }));

  return {
    grid: { left: 60, right: 30, top: 30, bottom: 60 },
    legend: { type: "scroll" as const, top: 0, textStyle: { color: "var(--vp-c-text-1)" } },
    tooltip: {
      trigger: "axis" as const,
      axisPointer: { type: "cross" as const },
      formatter: (params: { seriesName: string; data: [number, number]; color: string }[]) => {
        const dim = params[0]?.data[0];
        const head = `<strong>Dimension ${dim}</strong><br/>`;
        const rows = params
          .map((p) => {
            const point = chartPoints.value.find(
              (cp) => displaySubject(cp.subject) === p.seriesName && cp.dim === dim
            );
            const meta = point
              ? ` <span style="opacity:.55">— M=${point.m}${point.is_coordinode && point.sha_short ? `, ${point.sha_short}` : ""}</span>`
              : "";
            return `<span style="color:${p.color}">●</span> ${p.seriesName}: <strong>${p.data[1].toLocaleString()}</strong> QPS${meta}`;
          })
          .join("<br/>");
        return head + rows;
      },
    },
    xAxis: {
      type: "log" as const,
      name: "Embedding dimension",
      nameLocation: "middle" as const,
      nameGap: 30,
      data: [25, 50, 100, 128, 200, 256, 384, 768, 784, 960, 1024, 1536, 3072],
      axisLabel: { formatter: (v: number) => `${v}d` },
    },
    yAxis: {
      type: "log" as const,
      name: `QPS at recall@10 ≥ ${recallTarget.value}`,
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
    <p class="prompt">
      Pick your embedding model's dimension, then read off the chart.
      Higher is better.  Every line is measured on our bench host
      (Intel i9-9900K, 8C/16T, 64 GB RAM) under the
      <a href="https://github.com/erikbern/ann-benchmarks">ann-benchmarks</a>
      Docker harness — no leaderboard citations, no projections.
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
        <span class="ctrl-label">CoordiNode commit:</span>
        <select v-model="selectedCoordinodeSha" class="select">
          <option value="">latest ({{ coordinodeSHAs[0] ?? "—" }})</option>
          <option v-for="s in coordinodeSHAs" :key="s" :value="s">{{ s }}</option>
        </select>
        <span class="muted" v-if="coordinodeSHAs.length">
          {{ coordinodeSHAs.length }} commits available · for full timeline see the
          "Engineering timeline" tab
        </span>
      </div>
    </div>

    <!-- Status -->
    <p v-if="status === 'loading'" class="status">Loading bench data…</p>
    <p v-else-if="status === 'error'" class="status status-error">
      Could not load bench data: {{ errorMsg }}
    </p>
    <p v-else-if="status === 'empty'" class="status status-empty">
      No bench data on disk yet. Once
      <a href="https://github.com/structured-world/coordinode/tree/bench-data">the
      bench-data branch</a> picks up the first CoordiNode + competitor JSONs from
      the ann-benchmarks Docker run on our bench host, the chart will fill in.
    </p>

    <!-- Chart -->
    <template v-if="status === 'ok'">
      <h3>QPS at recall@10 ≥ {{ recallTarget }} across dimensions</h3>
      <v-chart class="echart" :option="qpsChartOption" autoresize />
    </template>

    <!-- Engine roster -->
    <h4 class="legend-title">
      Engines <span class="legend-hint">— click row to toggle on/off</span>
    </h4>
    <table class="engine-legend">
      <thead>
        <tr>
          <th class="col-toggle">On</th>
          <th>Engine</th>
          <th>Status</th>
          <th>Suite</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="s in allSubjects"
          :key="s"
          :class="['legend-row', enabledSubjects.has(s) ? 'is-on' : 'is-off', s === 'coordinode' ? 'is-coordinode' : '']"
          @click="toggleSubject(s)"
        >
          <td class="col-toggle">
            <span :class="['toggle-dot', enabledSubjects.has(s) ? 'on' : 'off']">
              {{ enabledSubjects.has(s) ? "●" : "○" }}
            </span>
          </td>
          <td class="engine-name">{{ displaySubject(s) }}</td>
          <td>
            <span v-if="s === 'coordinode'" class="status-pill status-current">measured per commit</span>
            <span v-else-if="isPending(s)" class="status-pill status-pending">pending bench run</span>
            <span v-else class="status-pill status-competitor">measured (one-shot)</span>
          </td>
          <td class="source">
            <span v-if="s === 'coordinode'">ann-benchmarks via CoordiNode adapter (per-commit timeline)</span>
            <span v-else-if="isPending(s)">{{ suiteFor(s) }}</span>
            <span v-else>ann-benchmarks on bench-host (one-shot snapshot)</span>
          </td>
        </tr>
      </tbody>
    </table>

    <p class="footnote">
      <strong>What's on this chart.</strong> Each line is the engine's best
      QPS at recall@10 ≥ {{ recallTarget }} per dataset, picked from the M / ef
      sweep. CoordiNode moves per push to <code>main</code> — use the commit
      picker above to snapshot the chart at a specific SHA.  Competitor lines
      are one-shot snapshots — re-run when we bump the competitor's pinned
      version.
    </p>
    <p class="footnote">
      <strong>How this works.</strong> The Cypher-bypass PyO3 binding the
      harness imports lives in
      <a href="https://github.com/structured-world/coordinode-python">coordinode-python</a>
      under <code>coordinode-embedded</code>.  Bench orchestrator and
      ann-benchmarks Docker integration are at
      <code>benches/ann-benchmarks-adapter/</code> +
      <code>scripts/run-coordinode-ann-benchmarks.sh</code>.
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
.select {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  padding: 0.25rem 0.55rem;
  font: inherit;
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  font-size: 0.85rem;
}
.muted {
  color: var(--vp-c-text-3);
  font-size: 0.85rem;
}
.status {
  margin: 1.5rem 0;
  padding: 1rem 1.5rem;
  border-radius: 6px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
}
.status-error {
  background: var(--vp-c-danger-soft, rgba(220, 60, 60, 0.1));
  color: var(--vp-c-danger-1, #c62828);
}
.status-empty {
  background: var(--vp-c-warning-soft, rgba(232, 160, 60, 0.12));
  color: var(--vp-c-warning-1, #b45a00);
}
.echart {
  height: 420px;
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
.engine-name {
  font-weight: 500;
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
</style>
