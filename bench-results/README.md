# Bench results

Canonical home for benchmark JSON outputs that the gh-pages
dashboard renders. Layout:

```
bench-results/
└── <modality>/                  # vector, graph, spatial, ts, doc, search
    └── <dataset>/               # sift-128-euclidean, ldbc-snb-sf1, ...
        └── <sha>-<subject>-<ts>.json
```

Where:
- `<sha>` — 7-char short SHA of the CoordiNode commit the result
  is paired with. The dashboard pairs same-`<sha>` runs across
  subjects for an apples-to-apples cell.
- `<subject>` — `coordinode` for the CN run, `hnswlib` /
  `faiss-cpu` / `mongodb-8` / `surrealdb-3` / etc. for competitors.
- `<ts>` — UTC `YYYYmmdd-HHMMSS` timestamp; chronological ordering
  within a (subject, sha) pair.

## How JSON files land here

**CN runs** — committed automatically by the
`.github/workflows/bench-vector.yml` self-hosted runner workflow
on every push to `main`. The workflow rebuilds the index, runs the
ef sweep, and pushes a single new JSON file into this directory as
part of the bench commit.

**Competitor runs** — placed manually by the engineer running the
off-cycle re-baseline. See
[`../benches/vector-ann/README.md`](../benches/vector-ann/README.md)
for the full procedure (one-time dataset placement on the runner,
Docker invocation, copy-back). Commit with:

```
git add bench-results/vector/sift-128-euclidean/*.json
git commit -m "chore(bench): refresh hnswlib SIFT1M baseline"
```

## Schema

See [`crates/coordinode-bench/src/lib.rs`](../crates/coordinode-bench/src/lib.rs)
for the `BenchReport` canonical schema. Both Rust (CN) and Python
(competitor) runners write the same shape; the gh-pages dashboard
treats them uniformly.

## What is NOT here

- **Raw datasets** — `.fvecs`, `.ivecs`, HDF5 etc. live ONLY on
  the bench runner under `/srv/coordinode-bench/datasets/`.
  Never commit dataset files: they're hundreds of MB and
  identical across the world's ANN literature.
- **Build artifacts** — index files, Docker container state etc.
  stay on the runner.
