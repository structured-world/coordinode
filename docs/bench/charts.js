// Bench dashboard renderer — reads BenchReport JSON files from
// the repo's `bench-results/` directory (relative to this page in
// the deployed site), groups by (modality, dataset), and renders
// three charts per dataset:
//
//   1. Pareto frontier (recall vs QPS) — current snapshot
//   2. Timeline of recall_at_k_peak across CN commits
//   3. Bar chart of qps_at_recall_0_95 at the latest CN SHA
//
// The data feed is `data/index.json` — a flat manifest generated
// at deploy time by `.github/workflows/gh-pages.yml`. The
// dashboard never crawls bench-results/ at runtime (single fetch
// per page load).

const INDEX_URL = "data/index.json";

async function loadManifest() {
  const r = await fetch(INDEX_URL, { cache: "no-store" });
  if (!r.ok) {
    throw new Error(
      `failed to load ${INDEX_URL} (${r.status}). The gh-pages workflow must run at least once with bench-results present for the dashboard to render.`,
    );
  }
  return r.json();
}

async function loadReport(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
  return r.json();
}

/** Pareto frontier: per-subject, the (recall, QPS) curve of the
 *  latest run on the most-recent CN commit. */
function paretoSpec(reports, dataset) {
  const points = [];
  for (const rep of reports) {
    if (!rep.metrics || !Array.isArray(rep.metrics.sweep)) continue;
    for (const p of rep.metrics.sweep) {
      points.push({
        subject: rep.subject,
        recall: p.recall_at_k,
        qps: p.qps,
        ef: p.ef_search,
        sha: rep.git.sha_short,
      });
    }
  }
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    description: `Pareto frontier — ${dataset}`,
    width: "container",
    height: 380,
    data: { values: points },
    layer: [
      {
        mark: { type: "line", interpolate: "monotone" },
        encoding: {
          x: { field: "recall", type: "quantitative", title: "Recall@10", scale: { domain: [0.7, 1.0] } },
          y: { field: "qps", type: "quantitative", title: "QPS (single-thread)", scale: { type: "log" } },
          color: { field: "subject", type: "nominal", title: "Engine" },
        },
      },
      {
        mark: { type: "point", filled: true, size: 80 },
        encoding: {
          x: { field: "recall", type: "quantitative" },
          y: { field: "qps", type: "quantitative" },
          color: { field: "subject", type: "nominal" },
          tooltip: [
            { field: "subject", type: "nominal" },
            { field: "ef", type: "ordinal", title: "ef_search" },
            { field: "recall", type: "quantitative", format: ".4f" },
            { field: "qps", type: "quantitative", format: ",.0f" },
            { field: "sha", type: "nominal", title: "CN SHA" },
          ],
        },
      },
    ],
  };
}

/** Timeline: recall_at_k_peak over CN commits, one line per subject. */
function timelineSpec(reports, dataset) {
  const rows = [];
  for (const rep of reports) {
    const peak = rep.metrics && rep.metrics.recall_at_k_peak;
    if (typeof peak !== "number") continue;
    rows.push({
      timestamp: rep.timestamp,
      sha: rep.git.sha_short,
      subject: rep.subject,
      recall_peak: peak,
    });
  }
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    description: `Recall@10 peak timeline — ${dataset}`,
    width: "container",
    height: 320,
    data: { values: rows },
    mark: { type: "line", point: true, interpolate: "monotone" },
    encoding: {
      x: { field: "timestamp", type: "temporal", title: "Commit timestamp" },
      y: {
        field: "recall_peak",
        type: "quantitative",
        title: "Peak recall@10",
        scale: { domain: [0.85, 1.0] },
      },
      color: { field: "subject", type: "nominal", title: "Engine" },
      tooltip: [
        { field: "subject", type: "nominal" },
        { field: "sha", type: "nominal", title: "CN SHA" },
        { field: "timestamp", type: "temporal" },
        { field: "recall_peak", type: "quantitative", format: ".4f" },
      ],
    },
  };
}

/** Bar chart: QPS at recall≥0.95 at the LATEST CN commit. */
function barSpec(reports, dataset) {
  const latestByCommit = new Map();
  for (const rep of reports) {
    const sha = rep.git.sha;
    if (!latestByCommit.has(sha) || rep.timestamp > latestByCommit.get(sha).timestamp) {
      latestByCommit.set(sha, rep);
    }
  }
  const allReports = [...latestByCommit.values()];
  allReports.sort((a, b) => b.git.commit_date.localeCompare(a.git.commit_date));
  if (allReports.length === 0) return null;
  const latestSha = allReports[0].git.sha;
  const sameSha = allReports.filter((r) => r.git.sha === latestSha);
  const bars = sameSha
    .map((r) => ({
      subject: r.subject,
      qps_recall_95: r.metrics ? r.metrics.qps_at_recall_0_95 : null,
    }))
    .filter((r) => typeof r.qps_recall_95 === "number");
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    description: `QPS @ recall≥0.95 — ${dataset} @ ${latestSha.slice(0, 7)}`,
    width: "container",
    height: 280,
    data: { values: bars },
    mark: "bar",
    encoding: {
      x: { field: "subject", type: "nominal", title: "Engine", sort: "-y" },
      y: { field: "qps_recall_95", type: "quantitative", title: "QPS @ recall ≥ 0.95" },
      color: { field: "subject", type: "nominal", legend: null },
      tooltip: [
        { field: "subject", type: "nominal" },
        { field: "qps_recall_95", type: "quantitative", format: ",.0f" },
      ],
    },
  };
}

async function main() {
  let manifest;
  try {
    manifest = await loadManifest();
  } catch (e) {
    document.getElementById("vega-sift1m-pareto").textContent =
      `No bench results yet. ${e.message}`;
    return;
  }
  // For now we only render vector / sift-128-euclidean. As other
  // modalities land, add a section per (modality, dataset) here.
  const sift = manifest.entries.filter(
    (e) => e.modality === "vector" && e.dataset === "sift-128-euclidean",
  );
  if (sift.length === 0) {
    document.getElementById("vega-sift1m-pareto").textContent =
      "No SIFT1M results yet. The first push to main with vector-bench changes will trigger the CI run.";
    return;
  }
  const reports = [];
  for (const e of sift) {
    try {
      reports.push(await loadReport(e.path));
    } catch (err) {
      console.warn(`skipping ${e.path}: ${err.message}`);
    }
  }
  if (reports.length === 0) {
    document.getElementById("vega-sift1m-pareto").textContent =
      "Manifest references SIFT1M results but no JSON loaded successfully.";
    return;
  }
  vegaEmbed("#vega-sift1m-pareto", paretoSpec(reports, "sift-128-euclidean"));
  vegaEmbed("#vega-sift1m-timeline", timelineSpec(reports, "sift-128-euclidean"));
  const bar = barSpec(reports, "sift-128-euclidean");
  if (bar) vegaEmbed("#vega-sift1m-bar", bar);
}

main().catch((e) => {
  console.error(e);
  document.body.insertAdjacentHTML(
    "beforeend",
    `<p style="color:red;text-align:center;padding:2rem;">Dashboard failed to render: ${e.message}</p>`,
  );
});
