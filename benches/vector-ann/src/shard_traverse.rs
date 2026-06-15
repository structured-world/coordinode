//! Sharded graph-traversal perf prototype.
//!
//! Demonstrates the distributed-traversal thesis: a heavy k-hop traversal whose
//! single-node cost is hundreds of milliseconds is split across N shard engines,
//! each expanding its slice of the frontier in parallel, so wall-clock per query
//! drops even after a per-hop inter-shard latency is added. Each shard is a real
//! on-disk StorageEngine holding the forward adjacency of the nodes it owns
//! (`src % shards`), so the single-node baseline reflects real posting-list read
//! cost rather than an in-memory toy.
//!
//! This is the in-process proof of the speedup. The cross-host version replaces
//! the local shard engines with gRPC calls to remote shard servers; the
//! coordinator logic (partition the frontier by owning shard, expand each slice
//! in parallel, gather and dedup) is identical.

use clap::Parser;
use coordinode_core::graph::edge::{encode_adj_key_forward, PostingList};
use coordinode_core::graph::node::NodeId;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::merge::encode_add_batch;
use coordinode_storage::engine::partition::Partition;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(name = "shard-traverse", version)]
struct Args {
    /// Person nodes in the synthetic Barabasi-Albert graph.
    #[arg(long, default_value_t = 20_000)]
    nodes: u64,

    /// BA attachment degree (edges added per new node). Mean undirected degree
    /// converges to ~2*m; both directions are materialised.
    #[arg(long, default_value_t = 8)]
    m: u64,

    /// Number of shards the graph is partitioned across (by `src % shards`).
    #[arg(long, default_value_t = 2)]
    shards: usize,

    /// Traversal depth (`*1..hops`).
    #[arg(long, default_value_t = 3)]
    hops: usize,

    /// Source nodes sampled for the measured query set.
    #[arg(long, default_value_t = 64)]
    queries: usize,

    /// Injected latency (ms) per shard call per hop, modelling the network
    /// round-trip to a remote shard. 0 = in-process (no network).
    #[arg(long, default_value_t = 0)]
    latency_ms: u64,

    /// If set, write the generated BA graph as Dgraph RDF n-quads to this path
    /// and exit (no storage load, no measurement). Lets an external engine load
    /// the identical graph for an apples-to-apples comparison. Each node carries
    /// an `nid` int (its node id, indexed) and `knows` uid edges.
    #[arg(long)]
    export_rdf: Option<std::path::PathBuf>,

    /// Diagnostic: build one hub adjacency key via many merge operands (the
    /// loader's write path) and time snapshot reads before and after a forced
    /// compaction. Confirms whether merge-on-read of uncompacted operands is the
    /// traversal-read bottleneck. Runs and exits; ignores graph args.
    #[arg(long)]
    confirm_merge: bool,

    /// Diagnostic: load the BA graph via per-edge merge operands (the server's
    /// write path), time a full k-hop BFS, collapse operands, then re-time. Shows
    /// the end-to-end traversal speedup from operand collapse. Uses --nodes/--m/
    /// --hops/--queries. Runs and exits.
    #[arg(long)]
    confirm_traverse: bool,
}

/// Deterministic splitmix64.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

fn disk_engine(dir: &std::path::Path) -> StorageEngine {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Ssd,
        Durability::Durable,
        Tier::Warm,
    )]);
    StorageEngine::open(&config).expect("open shard engine")
}

/// Forward neighbours of `src` from one engine (its owning shard, or the
/// combined engine for the baseline).
fn out_neighbors(engine: &StorageEngine, src: u64) -> Vec<u64> {
    let key = encode_adj_key_forward("KNOWS", NodeId::from_raw(src));
    match engine.get(Partition::Adj, &key).expect("get adj") {
        Some(bytes) => PostingList::from_bytes(&bytes)
            .expect("decode posting")
            .iter()
            .collect(),
        None => Vec::new(),
    }
}

/// Single-engine level-synchronous BFS reachable set within `[1..=hops]`.
fn reachable_single(engine: &StorageEngine, source: u64, hops: usize) -> BTreeSet<u64> {
    let mut reached = BTreeSet::new();
    let mut expanded = HashSet::new();
    let mut frontier = vec![source];
    for _ in 0..hops {
        let mut next = Vec::new();
        for &n in &frontier {
            if !expanded.insert(n) {
                continue;
            }
            for t in out_neighbors(engine, n) {
                if reached.insert(t) {
                    next.push(t);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    reached
}

/// Sharded coordinator BFS: each hop, partition the not-yet-expanded frontier by
/// owning shard, expand every shard's slice in parallel (one thread per shard,
/// each paying one `latency` round), then gather and dedup. Mirrors the
/// scatter-gather a distributed coordinator runs over remote shards.
fn reachable_sharded(
    shards: &[Arc<StorageEngine>],
    source: u64,
    hops: usize,
    latency: Duration,
) -> BTreeSet<u64> {
    let n = shards.len();
    let mut reached = BTreeSet::new();
    let mut expanded = HashSet::new();
    let mut frontier = vec![source];
    for _ in 0..hops {
        // Partition the new frontier by owning shard.
        let mut to_expand: Vec<u64> = frontier
            .into_iter()
            .filter(|&node| expanded.insert(node))
            .collect();
        if to_expand.is_empty() {
            break;
        }
        let mut buckets: Vec<Vec<u64>> = vec![Vec::new(); n];
        for node in to_expand.drain(..) {
            buckets[(node % n as u64) as usize].push(node);
        }

        // Scatter: expand each shard's slice concurrently. Each shard pays one
        // network round (latency) regardless of slice size, like one RPC/hop.
        let triples: Vec<u64> = std::thread::scope(|scope| {
            let handles: Vec<_> = buckets
                .iter()
                .enumerate()
                .map(|(sid, bucket)| {
                    let engine = Arc::clone(&shards[sid]);
                    scope.spawn(move || {
                        if !latency.is_zero() {
                            std::thread::sleep(latency);
                        }
                        let mut out = Vec::new();
                        for &node in bucket {
                            out.extend(out_neighbors(&engine, node));
                        }
                        out
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("shard thread"))
                .collect()
        });

        // Gather + dedup into the next frontier.
        let mut next = Vec::new();
        for tgt in triples {
            if reached.insert(tgt) {
                next.push(tgt);
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    reached
}

/// Diagnostic: prove whether merge-on-read of uncompacted adjacency operands is
/// the traversal-read cost. Builds one hub key via `degree` single-neighbour
/// merge operands (an incrementally-loaded hub) and a second hub via one put of
/// the full posting list (the compacted shape), then times snapshot reads of
/// each before and after a forced compaction.
fn confirm_merge_read() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = disk_engine(dir.path());
    let hub_merge = encode_adj_key_forward("KNOWS", NodeId::from_raw(1));
    let hub_put = encode_adj_key_forward("KNOWS", NodeId::from_raw(2));
    let degree: u64 = 1000;
    let iters: usize = 2000;

    // Key A: built from `degree` separate merge operands (worst case of a hub
    // touched once per loaded batch with no compaction since).
    for t in 0..degree {
        engine
            .merge(Partition::Adj, &hub_merge, &encode_add_batch(&[100 + t]))
            .expect("merge");
    }
    // Key B: built from a single put of the full posting list.
    let mut pl = PostingList::new();
    for t in 0..degree {
        pl.insert(100 + t);
    }
    engine
        .put(Partition::Adj, &hub_put, &pl.to_bytes().expect("pl bytes"))
        .expect("put");

    let read_us = |snap: &_, key: &[u8]| -> f64 {
        let _ = engine
            .snapshot_get(snap, Partition::Adj, key)
            .expect("warm");
        let t0 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(engine.snapshot_get(snap, Partition::Adj, key).expect("get"));
        }
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    };

    let snap = engine.snapshot();
    let before_merge = read_us(&snap, &hub_merge);
    let before_put = read_us(&snap, &hub_put);
    eprintln!("--- merge-on-read confirm (degree={degree}, {iters} reads) ---");
    eprintln!("BEFORE compaction:");
    eprintln!("  merge-operand hub: {before_merge:.2} us/read");
    eprintln!("  single-put hub:    {before_put:.2} us/read");
    eprintln!("  ratio: {:.1}x", before_merge / before_put.max(1e-9));

    engine
        .force_compaction(Partition::Adj)
        .expect("force_compaction");

    let snap2 = engine.snapshot();
    let after_merge = read_us(&snap2, &hub_merge);
    let after_put = read_us(&snap2, &hub_put);
    eprintln!("AFTER force_compaction(Adj):");
    eprintln!("  merge-operand hub: {after_merge:.2} us/read");
    eprintln!("  single-put hub:    {after_put:.2} us/read");
    eprintln!("  ratio: {:.1}x", after_merge / after_put.max(1e-9));

    // Apply the shipped fix: collapse merge operands into single stored values.
    let rewritten = engine
        .collapse_merge_operands(Partition::Adj)
        .expect("collapse");
    let snap3 = engine.snapshot();
    let after_repair = read_us(&snap3, &hub_merge);
    eprintln!("AFTER collapse_merge_operands(Adj) (rewrote {rewritten} keys):");
    eprintln!("  merge-operand hub: {after_repair:.2} us/read");
    eprintln!(
        "verdict: collapse sped the merged hub by {:.1}x (vs before)",
        before_merge / after_repair.max(1e-9)
    );
}

/// Diagnostic: load a BA graph through per-edge merge operands (the server's
/// incremental write path), time a full k-hop BFS, collapse the operands, then
/// re-time. Shows the end-to-end traversal speedup the operand collapse delivers.
fn confirm_traverse(args: &Args) {
    let nodes = args.nodes;
    let m = args.m.min(nodes.saturating_sub(1)).max(1);
    let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
    let mut targets: Vec<u64> = Vec::new();
    let mut rng = Rng(0x5EED_1234_ABCD);
    for i in 0..=m {
        for j in 0..=m {
            if i != j {
                adj.entry(i).or_default().push(j);
                targets.push(i);
            }
        }
    }
    for v in (m + 1)..nodes {
        let mut picked = HashSet::new();
        while picked.len() < m as usize {
            let t = if targets.is_empty() {
                rng.below(v)
            } else {
                targets[rng.below(targets.len() as u64) as usize]
            };
            if t != v {
                picked.insert(t);
            }
        }
        for &t in &picked {
            adj.entry(v).or_default().push(t);
            adj.entry(t).or_default().push(v);
            targets.push(v);
            targets.push(t);
        }
    }
    let edges: usize = adj.values().map(|v| v.len()).sum();
    eprintln!(
        "--- confirm-traverse: nodes={nodes} m={m} directed_edges={edges} hops={} queries={} ---",
        args.hops, args.queries
    );

    // Load via per-edge merge operands, the shape the server accumulates.
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = disk_engine(dir.path());
    for (&src, tgts) in &adj {
        let key = encode_adj_key_forward("KNOWS", NodeId::from_raw(src));
        for &t in tgts {
            engine
                .merge(Partition::Adj, &key, &encode_add_batch(&[t]))
                .expect("merge");
        }
    }

    let mut qrng = Rng(0xC0FF_EE00_2222);
    let sources: Vec<u64> = (0..args.queries).map(|_| qrng.below(nodes)).collect();

    let bfs_ms = |engine: &StorageEngine| -> (f64, f64) {
        let _ = reachable_single(engine, sources[0], args.hops);
        let t0 = Instant::now();
        let mut reach = 0usize;
        for &s in &sources {
            reach += reachable_single(engine, s, args.hops).len();
        }
        let per = t0.elapsed().as_secs_f64() * 1000.0 / sources.len() as f64;
        (per, reach as f64 / sources.len() as f64)
    };

    let (before, mean_reach) = bfs_ms(&engine);
    eprintln!("BEFORE collapse: {before:.1} ms/query (mean_reachable={mean_reach:.0})");

    let rewritten = engine
        .collapse_merge_operands(Partition::Adj)
        .expect("collapse");
    let (after, _) = bfs_ms(&engine);
    eprintln!("AFTER collapse_merge_operands ({rewritten} keys): {after:.1} ms/query");
    eprintln!(
        "verdict: collapse sped the traversal by {:.1}x",
        before / after.max(1e-9)
    );
}

fn main() {
    let args = Args::parse();

    if args.confirm_merge {
        confirm_merge_read();
        return;
    }

    if args.confirm_traverse {
        confirm_traverse(&args);
        return;
    }

    if args.confirm_traverse {
        confirm_traverse(&args);
        return;
    }

    let n = args.shards.max(1);

    // Generate a Barabasi-Albert graph: adjacency in memory first, then write
    // each source's posting list ONCE per engine (fast bulk load).
    eprintln!(
        "generating BA graph: nodes={} m={} shards={}",
        args.nodes, args.m, n
    );
    let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
    let mut targets: Vec<u64> = Vec::new(); // attachment multiset (preferential)
    let mut rng = Rng(0x5EED_1234_ABCD);
    let m = args.m.min(args.nodes.saturating_sub(1)).max(1);
    // Seed clique.
    for i in 0..=m {
        for j in 0..=m {
            if i != j {
                adj.entry(i).or_default().push(j);
                targets.push(i);
            }
        }
    }
    for v in (m + 1)..args.nodes {
        let mut picked = HashSet::new();
        while picked.len() < m as usize {
            let t = if targets.is_empty() {
                rng.below(v)
            } else {
                let idx = rng.below(targets.len() as u64) as usize;
                targets[idx]
            };
            if t != v {
                picked.insert(t);
            }
        }
        for &t in &picked {
            // Materialise both directions (social KNOWS is symmetric).
            adj.entry(v).or_default().push(t);
            adj.entry(t).or_default().push(v);
            targets.push(v);
            targets.push(t);
        }
    }
    let edge_count: usize = adj.values().map(|v| v.len()).sum();
    eprintln!("graph built: directed edges={edge_count}");

    // RDF export path: write the identical graph as Dgraph n-quads and exit.
    if let Some(path) = &args.export_rdf {
        use std::io::Write;
        let file = std::fs::File::create(path).expect("create rdf file");
        let mut w = std::io::BufWriter::new(file);
        let mut nodes: Vec<u64> = adj.keys().copied().collect();
        nodes.sort_unstable();
        for &v in &nodes {
            writeln!(
                w,
                "_:n{v} <nid> \"{v}\"^^<http://www.w3.org/2001/XMLSchema#int> ."
            )
            .expect("write nid");
        }
        for &src in &nodes {
            for &dst in &adj[&src] {
                writeln!(w, "_:n{src} <knows> _:n{dst} .").expect("write edge");
            }
        }
        w.flush().expect("flush rdf");
        eprintln!(
            "exported {} nodes + {edge_count} edges to {}",
            nodes.len(),
            path.display()
        );
        return;
    }

    // Combined (single-node baseline) + N shard engines.
    let combined_dir = tempfile::tempdir().expect("combined tempdir");
    let combined = disk_engine(combined_dir.path());
    let shard_dirs: Vec<tempfile::TempDir> = (0..n)
        .map(|_| tempfile::tempdir().expect("shard tempdir"))
        .collect();
    let shards: Vec<Arc<StorageEngine>> = shard_dirs
        .iter()
        .map(|d| Arc::new(disk_engine(d.path())))
        .collect();

    for (&src, tgts) in &adj {
        let mut list = PostingList::new();
        for &t in tgts {
            list.insert(t);
        }
        let key = encode_adj_key_forward("KNOWS", NodeId::from_raw(src));
        let bytes = list.to_bytes().expect("serialize posting");
        combined
            .put(Partition::Adj, &key, &bytes)
            .expect("put combined");
        let sid = (src % n as u64) as usize;
        shards[sid]
            .put(Partition::Adj, &key, &bytes)
            .expect("put shard");
    }
    eprintln!("loaded {} sources into combined + {} shards", adj.len(), n);

    // Sample sources for the measured query set.
    let mut qrng = Rng(0xC0FF_EE00_1111);
    let sources: Vec<u64> = (0..args.queries).map(|_| qrng.below(args.nodes)).collect();
    let latency = Duration::from_millis(args.latency_ms);

    // Warm both paths once (cold cache should not skew the comparison).
    let _ = reachable_single(&combined, sources[0], args.hops);
    let _ = reachable_sharded(&shards, sources[0], args.hops, Duration::ZERO);

    // Measure single-node baseline.
    let t0 = Instant::now();
    let mut single_reach = 0usize;
    for &s in &sources {
        single_reach += reachable_single(&combined, s, args.hops).len();
    }
    let single_total = t0.elapsed();
    let single_per_q = single_total.as_secs_f64() * 1000.0 / sources.len() as f64;

    // Correctness gate (untimed): the parallel sharded expansion must reach the
    // exact same node set as the sequential single-engine BFS for every source.
    for &s in &sources {
        let single = reachable_single(&combined, s, args.hops);
        let sharded = reachable_sharded(&shards, s, args.hops, latency);
        assert_eq!(
            single, sharded,
            "sharded reachable set diverged from single-node at source {s}"
        );
    }

    // Measure the sharded (parallel scatter-gather) path on its own.
    let t2 = Instant::now();
    for &s in &sources {
        let _ = reachable_sharded(&shards, s, args.hops, latency);
    }
    let sharded_total = t2.elapsed();
    let sharded_per_q = sharded_total.as_secs_f64() * 1000.0 / sources.len() as f64;

    let mean_reach = single_reach as f64 / sources.len() as f64;
    let speedup = single_per_q / sharded_per_q.max(f64::EPSILON);

    println!("--- sharded-traversal prototype ---");
    println!(
        "nodes={} m={} shards={} hops={} queries={} latency_ms={}",
        args.nodes,
        args.m,
        n,
        args.hops,
        sources.len(),
        args.latency_ms
    );
    println!("mean reachable per query: {mean_reach:.0}");
    println!("single-node:  {single_per_q:.2} ms/query");
    println!(
        "sharded ({n}): {sharded_per_q:.2} ms/query  (+{}ms latency/shard/hop)",
        args.latency_ms
    );
    println!(
        "speedup: {speedup:.2}x  ({})",
        if speedup > 1.0 {
            "sharding helps"
        } else {
            "sharding does NOT help"
        }
    );
}
