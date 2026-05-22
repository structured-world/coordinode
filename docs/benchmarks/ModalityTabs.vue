<script setup lang="ts">
import { computed, defineAsyncComponent, ref } from "vue";

const BenchVectorAnn = defineAsyncComponent(() => import("./BenchVectorAnn.vue"));

interface Competitor {
  name: string;
  kind: "specialist" | "multi-model";
  license: string;
  // Exact version benched. Specialist baselines are pinned: when we re-run
  // a specialist on a newer release we DELETE the previous JSON from
  // bench-results/ rather than keeping a timeline. The repo always
  // reflects "CoordiNode <latest commit> vs <this exact competitor build>".
  version?: string;
  note?: string;
}

interface Modality {
  id: string;
  label: string;
  status: "live" | "coming-soon";
  milestone?: string;
  suite: string;
  blurb: string;
  competitors: Competitor[];
}

const modalities: Modality[] = [
  {
    id: "vector",
    label: "Vector",
    status: "live",
    suite: "ann-benchmarks · SIFT1M",
    blurb:
      "1 000 000 SIFT1M base vectors, 10 000 queries, ground-truth top-100. HNSW M=32, ef_construction=200, ef_search sweep.",
    competitors: [
      { name: "hnswlib", kind: "specialist", license: "Apache-2.0", version: "0.8.0", note: "reference implementation" },
      { name: "Faiss (HNSW)", kind: "specialist", license: "MIT", version: "planned" },
      { name: "Qdrant", kind: "specialist", license: "Apache-2.0", version: "planned" },
      { name: "Milvus", kind: "specialist", license: "Apache-2.0", version: "planned" },
      { name: "pgvector", kind: "multi-model", license: "PostgreSQL", version: "planned" },
      { name: "MongoDB Atlas Vector", kind: "multi-model", license: "SSPL", version: "planned" },
    ],
  },
  {
    id: "graph",
    label: "Graph",
    status: "coming-soon",
    milestone: "v0.5",
    suite: "LDBC SNB Interactive",
    blurb:
      "LDBC Social Network Benchmark, Interactive workload (short reads + complex reads + updates) at scale factor 1 and 10. Power@k throughput and tail latency.",
    competitors: [
      { name: "Neo4j CE", kind: "specialist", license: "GPLv3", version: "planned" },
      { name: "Memgraph CE", kind: "specialist", license: "BSL", version: "planned" },
      { name: "JanusGraph", kind: "specialist", license: "Apache-2.0", version: "planned" },
      { name: "ArangoDB CE", kind: "multi-model", license: "Apache-2.0 / BSL", version: "planned" },
    ],
  },
  {
    id: "document",
    label: "Document",
    status: "coming-soon",
    milestone: "v0.5",
    suite: "YCSB workloads A & C",
    blurb:
      "YCSB-A (50/50 read/update) and YCSB-C (100% read). Multi-region replication latency tracked separately. Json document model, secondary indexes.",
    competitors: [
      { name: "MongoDB CE", kind: "specialist", license: "SSPL", version: "planned" },
      { name: "Couchbase CE", kind: "specialist", license: "Apache-2.0", version: "planned" },
      { name: "ArangoDB CE", kind: "multi-model", license: "Apache-2.0 / BSL", version: "planned" },
    ],
  },
  {
    id: "timeseries",
    label: "Time-Series",
    status: "coming-soon",
    milestone: "v0.6",
    suite: "TSBS DevOps",
    blurb:
      "TSBS DevOps workload — load throughput plus 15 canonical query patterns. Ingest rate at sustained backpressure.",
    competitors: [
      { name: "TimescaleDB", kind: "specialist", license: "Apache-2.0 / TSL", version: "planned" },
      { name: "InfluxDB OSS", kind: "specialist", license: "MIT", version: "planned" },
      { name: "QuestDB", kind: "specialist", license: "Apache-2.0", version: "planned" },
      { name: "ClickHouse", kind: "multi-model", license: "Apache-2.0", version: "planned" },
    ],
  },
  {
    id: "spatial",
    label: "Spatial",
    status: "coming-soon",
    milestone: "v0.6",
    suite: "PostGIS-shape geo workload",
    blurb:
      "Geometry containment, k-NN over geo points, bounding-box scans. R-tree index over Cartesian and geographic coordinates.",
    competitors: [
      { name: "PostGIS", kind: "specialist", license: "GPLv2", version: "planned" },
      { name: "MongoDB CE (geo)", kind: "multi-model", license: "SSPL", version: "planned" },
    ],
  },
  {
    id: "fulltext",
    label: "Full-Text",
    status: "coming-soon",
    milestone: "v0.5",
    suite: "Search Benchmark Game",
    blurb:
      "Tantivy's published Search Benchmark Game — Wikipedia corpus, BM25 ranking, 50 query patterns. Index size and query latency at the 95th percentile.",
    competitors: [
      { name: "OpenSearch", kind: "specialist", license: "Apache-2.0", version: "planned" },
      { name: "Tantivy (stand-alone)", kind: "specialist", license: "MIT", version: "planned" },
      { name: "Meilisearch", kind: "specialist", license: "MIT", version: "planned" },
    ],
  },
];

const active = ref(modalities[0]!.id);
const current = computed(() => modalities.find((m) => m.id === active.value)!);
</script>

<template>
  <div class="modality-tabs">
    <div class="tab-strip" role="tablist">
      <button
        v-for="m in modalities"
        :key="m.id"
        :class="{ tab: true, 'tab-active': active === m.id }"
        :aria-selected="active === m.id"
        role="tab"
        @click="active = m.id"
      >
        <span class="tab-label">{{ m.label }}</span>
        <span v-if="m.status === 'coming-soon'" class="tab-badge">soon</span>
      </button>
    </div>

    <div class="tab-panel" role="tabpanel">
      <div class="panel-header">
        <h3 class="panel-title">
          {{ current.label }}
          <span class="suite-chip">{{ current.suite }}</span>
        </h3>
        <p class="panel-blurb">{{ current.blurb }}</p>
      </div>

      <!-- Live data slot -->
      <div v-if="current.id === 'vector'" class="panel-body">
        <ClientOnly>
          <BenchVectorAnn />
        </ClientOnly>
      </div>

      <!-- Coming-soon placeholder -->
      <div v-else class="panel-body coming-soon">
        <div class="placeholder">
          <div class="placeholder-icon">⏳</div>
          <div class="placeholder-text">
            <strong>Coming soon</strong>
            — first run scheduled at
            <code>{{ current.milestone }}</code>
          </div>
        </div>
      </div>

      <!-- Competitor matrix -->
      <div class="competitors">
        <h4 class="competitors-title">Competitors in this modality</h4>
        <p class="competitors-policy">
          Each competitor is benched at a single pinned version, recorded
          in the JSON. When we re-run a baseline on a newer build we
          <strong>replace</strong> the previous JSON in
          <code>bench-results/</code> — no historical competitor timeline
          accumulates. CoordiNode results, by contrast, advance on every
          push to <code>main</code>.
        </p>
        <table class="competitor-table">
          <thead>
            <tr>
              <th>Engine</th>
              <th>Type</th>
              <th>Version benched</th>
              <th>License</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="c in current.competitors" :key="c.name">
              <td class="engine-name">{{ c.name }}</td>
              <td>
                <span :class="['kind-pill', `kind-${c.kind}`]">
                  {{ c.kind === "specialist" ? "specialist" : "multi-model" }}
                </span>
              </td>
              <td>
                <code v-if="c.version && c.version !== 'planned'">{{ c.version }}</code>
                <span v-else class="version-pending">planned</span>
              </td>
              <td><code>{{ c.license }}</code></td>
              <td class="note-cell">{{ c.note ?? "—" }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<style scoped>
.modality-tabs {
  margin: 1.5rem 0 2rem;
}

.tab-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
  border-bottom: 1px solid var(--vp-c-divider);
  margin-bottom: 1.25rem;
}

.tab {
  appearance: none;
  background: transparent;
  border: 1px solid transparent;
  border-bottom: none;
  border-radius: 6px 6px 0 0;
  padding: 0.55rem 1rem;
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--vp-c-text-2);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  transition: background-color 0.15s, color 0.15s, border-color 0.15s;
}

.tab:hover {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
}

.tab-active {
  background: var(--vp-c-bg);
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-divider);
  border-bottom-color: var(--vp-c-bg);
  margin-bottom: -1px;
}

.tab-badge {
  font-size: 0.7rem;
  font-weight: 600;
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
  background: var(--vp-c-warning-soft);
  color: var(--vp-c-warning-1);
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.panel-header {
  margin-bottom: 1rem;
}

.panel-title {
  margin: 0 0 0.5rem;
  display: flex;
  align-items: baseline;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.suite-chip {
  font-size: 0.85rem;
  font-weight: 500;
  padding: 0.15rem 0.55rem;
  border-radius: 4px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
}

.panel-blurb {
  margin: 0;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

.panel-body {
  margin: 1.25rem 0 1.5rem;
}

.coming-soon .placeholder {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 2rem;
  border: 1px dashed var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
}

.placeholder-icon {
  font-size: 2rem;
}

.placeholder-text {
  font-size: 1rem;
  color: var(--vp-c-text-2);
}

.placeholder-text strong {
  color: var(--vp-c-text-1);
}

.competitors-title {
  margin: 1.5rem 0 0.75rem;
  font-size: 1rem;
  color: var(--vp-c-text-1);
}

.competitor-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
}

.competitor-table th,
.competitor-table td {
  text-align: left;
  padding: 0.55rem 0.75rem;
  border-bottom: 1px solid var(--vp-c-divider);
}

.competitor-table th {
  font-weight: 600;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg-soft);
}

.engine-name {
  font-weight: 500;
}

.kind-pill {
  display: inline-block;
  font-size: 0.78rem;
  font-weight: 500;
  padding: 0.12rem 0.5rem;
  border-radius: 10px;
  white-space: nowrap;
}

.kind-specialist {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
}

.kind-multi-model {
  background: var(--vp-c-purple-soft, rgba(159, 122, 234, 0.16));
  color: var(--vp-c-purple-1, #9f7aea);
}

.note-cell {
  color: var(--vp-c-text-2);
  font-size: 0.88rem;
}

.competitors-policy {
  margin: 0 0 1rem;
  padding: 0.75rem 1rem;
  border-left: 3px solid var(--vp-c-brand-1);
  background: var(--vp-c-bg-soft);
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
  line-height: 1.55;
}

.version-pending {
  color: var(--vp-c-text-3);
  font-style: italic;
  font-size: 0.85rem;
}
</style>
