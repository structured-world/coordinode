//! The `serve` subcommand: open storage, join or bootstrap Raft, wire every
//! protocol frontend onto its port and run until a shutdown signal arrives.

use std::net::SocketAddr;
use std::sync::Arc;

use tonic::transport::Server;
use tracing::info;

use crate::checkpoint;
use crate::config;
use crate::grpc;
use crate::logging;
use crate::ops;
use crate::pg;
use crate::proto;
use crate::registry;
use crate::services;

/// Raise the process open-file-descriptor soft limit before opening storage.
///
/// `target = Some(n)` requests `n` descriptors (clamped to the hard limit);
/// `None` raises the soft limit to the current hard limit. Returns the
/// effective `(soft, hard)` pair, or `None` when the syscall fails. The storage
/// engine keeps many files open at once, so a low limit surfaces as
/// "too many open files" under load.
#[cfg(unix)]
fn set_nofile_limit(target: Option<u64>) -> Option<(u64, u64)> {
    // `None` means "raise to the hard limit": request u64::MAX, which the helper
    // clamps to the current hard limit.
    let want = target.unwrap_or(u64::MAX);
    match rlimit::increase_nofile_limit(want) {
        Ok(soft) => {
            let hard = rlimit::Resource::NOFILE
                .get()
                .map(|(_, hard)| hard)
                .unwrap_or(soft);
            Some((soft, hard))
        }
        Err(_) => None,
    }
}

/// No descriptor limit to manage on non-unix platforms.
#[cfg(not(unix))]
fn set_nofile_limit(_target: Option<u64>) -> Option<(u64, u64)> {
    None
}

/// Run the server until SIGTERM or Ctrl+C.
///
/// `config_path` selects the YAML config file (absent = built-in defaults);
/// `overrides` are the command-line flags, folded over the file last so the
/// command line wins.
pub(crate) async fn serve(
    extensions: crate::builder::ServerBuilder,
    config_path: Option<String>,
    overrides: Box<config::CliOverrides>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Resolve the single config gate: built-in defaults, overlaid by the
    // YAML config file (if `--config` given), overlaid last by the
    // command-line flags. A malformed / unreadable config file is a
    // startup error rather than a silent fallback.
    let mut cfg = config::ServerConfig::load(config_path.as_deref())
        .map_err(|e| format!("config error: {e}"))?;
    cfg.apply_overrides(&overrides);

    // Set the inter-node wire compression level before any gRPC service
    // starts; the transport codec reads it per message.
    coordinode_wire::set_wire_zstd_level(cfg.wire_compression_level);

    // Install the pure-Rust TLS crypto provider as the process default so
    // tonic's TLS builders use it (no C FFI). Must precede any TLS config.
    coordinode_wire::tls::install_ce_crypto_provider();

    // Resolve the operational mode from the merged string value, so a
    // mode set in the config file is validated exactly like a CLI flag.
    //
    // `full` is the built-in and parses here. A downstream distribution makes
    // further values legal by registering a handler for them; a value nobody
    // registered keeps the built-in rejection, message and exit code.
    let extension_mode = extensions.serve_modes.get(&cfg.mode).cloned();
    let mode: String = match config::ServeMode::parse(&cfg.mode) {
        Ok(m) => m.to_string(),
        Err(_) if extension_mode.is_some() => cfg.mode.clone(),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    // Resolve the storage topology (an explicit multi-endpoint list or
    // the single-endpoint `data_dir` desugar) and the page-ECC request
    // once, while the config is still whole — the destructure below
    // moves it field-by-field.
    let mut storage_config = cfg.resolve_storage_config();
    let page_ecc_requested = cfg.page_ecc_requested();

    // Bind the resolved settings into the local names the rest of the
    // handler uses. `peers` becomes `None` when empty (= standalone),
    // matching the cluster-detection contract below.
    // Capture the scrub config before destructuring moves `cfg`.
    let scrub_cfg = cfg.scrub_config();
    // Capture checkpoint settings before destructuring moves `cfg`.
    let checkpoint_enabled = cfg.checkpoint_enabled;
    let checkpoint_interval_secs = cfg.checkpoint_interval_secs;
    let checkpoint_keep = cfg.checkpoint_keep;
    let checkpoint_dir = cfg.checkpoint_directory();
    // Capture AFTER COMMIT trigger dispatch settings before the move
    // (config-file surface; applied to the Database / worker below).
    let trigger_dispatch_cfg = cfg.trigger_dispatch_config();
    let trigger_dispatch_interval = cfg.trigger_dispatch_interval();
    let config::ServerConfig {
        node_id,
        grpc_addr,
        advertise_addr,
        rest_addr,
        ops_addr,
        pg_addr,
        data_dir,
        storage: _,
        nofile,
        max_connections,
        max_request_size_mb,
        request_timeout_secs,
        http2_keepalive_secs,
        cache_size_mb,
        write_buffer_mb,
        retention_window_secs,
        registry_heartbeat_ms,
        registry_eviction_ms,
        cdc_consumer_ttl_secs,
        interactive_txn_idle_timeout_secs,
        interactive_txn_max_bytes,
        peers: peers_vec,
        mode: _,
        // Already consumed above via set_wire_zstd_level before serving.
        wire_compression_level: _,
        tls_cert,
        tls_key,
        tls_ca,
        tls_require_client_auth,
        // Already captured above via scrub_cfg before the move.
        scrub_enabled: _,
        scrub_interval_secs: _,
        scrub_throttle_ms: _,
        // Already captured above before the move.
        checkpoint_enabled: _,
        checkpoint_interval_secs: _,
        checkpoint_dir: _,
        checkpoint_keep: _,
        // Already captured above (trigger_dispatch_cfg / _interval) before the move.
        trigger_max_cascade_depth: _,
        trigger_default_retry_attempts: _,
        trigger_default_backoff_ms: _,
        trigger_dispatch_interval_ms: _,
        extensions: extension_config,
    } = cfg;
    let peers = if peers_vec.is_empty() {
        None
    } else {
        Some(peers_vec)
    };
    #[cfg(not(feature = "rest-proxy"))]
    let _ = rest_addr;

    // Cross-field validation, deferred from CLI parse because the peer
    // list can arrive from the config file: a node id above 1 only makes
    // sense as a member of a multi-node cluster.
    if node_id > 1 && peers.is_none() {
        eprintln!(
            "error: node_id={node_id} requires peers. \
                     Single-node deployments always use node-id=1."
        );
        std::process::exit(1);
    }

    logging::init_logging();

    // Raise the open-file-descriptor limit before opening storage: the
    // engine keeps many files open at once. Honour an explicit target or
    // raise the soft limit to the hard limit.
    if let Some((soft, hard)) = set_nofile_limit(nofile) {
        info!(soft, hard, "file-descriptor limit");
    }

    let addr: SocketAddr = grpc_addr.parse()?;
    // Advertise address is what peers use to connect to this node.
    // Falls back to grpc_addr when not explicitly set.
    let effective_advertise = advertise_addr.unwrap_or_else(|| grpc_addr.clone());
    let cluster_mode = peers.is_some();
    info!(
        data_dir = %data_dir,
        mode = %mode,
        node_id = node_id,
        cluster = cluster_mode,
        advertise = %effective_advertise,
        "coordinode v{} starting on {addr}",
        env!("CARGO_PKG_VERSION")
    );

    coordinode_vector::metrics::log_simd_capabilities();

    // All modes use RaftProposalPipeline — unified write path.
    //
    // - Standalone (no --peers): single-node Raft (node_id=1, StubNetwork).
    //   Writes go through Raft → oplog always populated → CDC works in both modes.
    // - Cluster (--peers): multi-node Raft (GrpcNetwork, leader election).
    //   Writes replicated to followers before commit.
    //
    // `raft_node_shared` provides the read fence (R141), ClusterService
    // administration, and ensures consistent apply ordering via oracle.

    // Common setup: open storage engine + timestamp oracle. The storage
    // topology was resolved from config above (a multi-endpoint list or
    // the single-endpoint `data_dir` desugar); apply the cache / write-
    // buffer size overrides on top.
    if let Some(mb) = cache_size_mb {
        storage_config.block_cache_bytes = mb.saturating_mul(1024 * 1024);
    }
    if let Some(mb) = write_buffer_mb {
        storage_config.max_write_buffer_bytes = mb.saturating_mul(1024 * 1024);
    }
    // Surface the page-ECC build/config mismatch: an operator who asked
    // for per-block ECC on a binary built without the feature gets a
    // no-op, not a silent one.
    if page_ecc_requested && !cfg!(feature = "page_ecc") {
        tracing::warn!(
            "a storage endpoint requests per-block ECC (page_ecc) but \
                     this binary was built without the `page_ecc` feature — the \
                     request has no on-disk effect; rebuild with \
                     `--features page_ecc` to enable it"
        );
    }
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine = coordinode_storage::engine::core::StorageEngine::open_with_oracle(
        &storage_config,
        oracle.clone(),
    )
    .map_err(|e| format!("failed to open storage: {e}"))?;
    let engine = Arc::new(engine);

    // Shared slot for this node's RaftNode, filled once it is built below.
    // The scrub task (spawned now) reads it for WAL-replay repair; its
    // first run is jitter-delayed by minutes, long after the slot is set.
    let raft_slot: Arc<std::sync::OnceLock<Arc<coordinode_raft::cluster::RaftNode>>> =
        Arc::new(std::sync::OnceLock::new());

    // Background integrity scrub. Each node verifies its OWN local
    // storage independently (no leader election — silent bit rot is a
    // per-node, per-disk concern), so this is multi-instance-safe with
    // no shared state. The scan is blocking file I/O, kept off the async
    // runtime via spawn_blocking and throttled per config so it yields to
    // production traffic.
    {
        if scrub_cfg.enabled {
            let scrub_engine = Arc::clone(&engine);
            let interval = scrub_cfg.interval;
            // For WAL-replay repair: the Raft oplog source + the checkpoint
            // base directory.
            let scrub_raft = Arc::clone(&raft_slot);
            let scrub_ckpt_dir = checkpoint_dir.clone();
            // Peers to pull a fresh copy from when scrub finds corruption
            // (CE basic replica-fetch repair). Normalised to URIs; empty
            // when standalone (nothing to repair from). Each node repairs
            // its own corruption independently.
            let repair_peers: Vec<String> = peers
                .as_ref()
                .map(|ps| {
                    ps.iter()
                        .map(|p| {
                            if p.contains("://") {
                                p.clone()
                            } else {
                                format!("http://{p}")
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            let repair_installer = Arc::new(coordinode_replicate::SegmentInstaller::new(
                Arc::clone(&engine),
            ));
            // Stagger the first run by node id so a fleet does not scrub
            // in lockstep and saturate I/O cluster-wide at once.
            let jitter = interval
                .checked_div(16)
                .map(|slice| slice.saturating_mul(u32::try_from(node_id % 16).unwrap_or(0)))
                .unwrap_or_default();
            tokio::spawn(async move {
                tokio::time::sleep(jitter).await;
                let mut ticker = tokio::time::interval(interval);
                ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    ticker.tick().await;
                    let eng = Arc::clone(&scrub_engine);
                    let cfg2 = scrub_cfg.clone();
                    match tokio::task::spawn_blocking(move || {
                        coordinode_storage::scrub::scrub_all(&eng, &cfg2)
                    })
                    .await
                    {
                        Ok(Ok(report)) => {
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_secs_f64())
                                .unwrap_or(0.0);
                            metrics::gauge!("coordinode_scrub_last_timestamp_seconds").set(now);
                            metrics::gauge!("coordinode_scrub_duration_seconds")
                                .set(report.duration.as_secs_f64());
                            metrics::gauge!("coordinode_scrub_blocks_checked")
                                .set(report.blocks_checked as f64);
                            metrics::counter!("coordinode_scrub_pages_scanned_total")
                                .increment(report.blocks_checked);
                            if report.has_errors() {
                                metrics::counter!("coordinode_scrub_errors_total")
                                    .increment(report.errors.len() as u64);
                                let mut corrupt = std::collections::HashSet::new();
                                for err in &report.errors {
                                    tracing::error!(
                                        partition = err.partition.name(),
                                        detail = %err.message,
                                        "scrub detected corruption"
                                    );
                                    corrupt.insert(err.partition);
                                }
                                // Basic replica-fetch repair: re-pull each
                                // affected partition from healthy peers over
                                // the swarm transport and re-install it. A
                                // standalone node has no peer to repair from.
                                for part in corrupt {
                                    // 1) Replica-fetch repair from healthy peers.
                                    let from_peers = if repair_peers.is_empty() {
                                        None
                                    } else {
                                        Some(
                                            repair_installer
                                                .repair_partition(
                                                    &repair_peers,
                                                    part,
                                                    1 << 20,
                                                    coordinode_replicate::PieceEncoding::None,
                                                )
                                                .await,
                                        )
                                    };
                                    match from_peers {
                                        Some(Ok(bytes)) => {
                                            metrics::counter!("coordinode_scrub_repairs_total")
                                                .increment(1);
                                            tracing::info!(
                                                partition = part.name(),
                                                bytes,
                                                "repaired partition from peers"
                                            );
                                            continue;
                                        }
                                        // No reachable replica (or none configured) → fall
                                        // through to WAL-replay repair below.
                                        None
                                        | Some(Err(coordinode_replicate::RepairError::NoSource(
                                            _,
                                        ))) => {}
                                        Some(Err(e)) => {
                                            tracing::warn!(
                                                partition = part.name(),
                                                %e,
                                                "partition repair failed"
                                            );
                                            continue;
                                        }
                                    }

                                    // 2) WAL-replay repair: rebuild from the latest local
                                    // checkpoint + Raft oplog replay (needs the Raft oplog —
                                    // cluster mode; a standalone node has neither replica nor
                                    // oplog, see the single-node gap).
                                    let Some(raft) = scrub_raft.get() else {
                                        tracing::warn!(
                                                    partition = part.name(),
                                                    "no replica and no Raft oplog (standalone) — cannot repair"
                                                );
                                        continue;
                                    };
                                    let Some(ckpt) = checkpoint::latest_checkpoint(&scrub_ckpt_dir)
                                    else {
                                        tracing::warn!(
                                            partition = part.name(),
                                            "no checkpoint available for WAL-replay repair"
                                        );
                                        continue;
                                    };
                                    let from =
                                        coordinode_raft::storage::checkpoint_oplog_last_index(
                                            &ckpt,
                                        )
                                        .map_or(0, |i| i + 1);
                                    let oplog_since = match raft.read_oplog_since(from) {
                                        Ok(v) => v,
                                        Err(e) => {
                                            tracing::warn!(partition = part.name(), %e, "read oplog for WAL-replay failed");
                                            continue;
                                        }
                                    };
                                    let inst = Arc::clone(&repair_installer);
                                    let ckpt2 = ckpt.clone();
                                    match tokio::task::spawn_blocking(move || {
                                        inst.wal_replay_repair(&ckpt2, &oplog_since, part)
                                    })
                                    .await
                                    {
                                        Ok(Ok(bytes)) => {
                                            metrics::counter!("coordinode_scrub_wal_repairs_total")
                                                .increment(1);
                                            tracing::info!(
                                                partition = part.name(),
                                                bytes,
                                                "repaired partition by WAL replay from checkpoint"
                                            );
                                        }
                                        Ok(Err(e)) => {
                                            tracing::warn!(partition = part.name(), %e, "WAL-replay repair failed")
                                        }
                                        Err(e) => {
                                            tracing::warn!(partition = part.name(), %e, "WAL-replay repair task panicked")
                                        }
                                    }
                                }
                            } else {
                                tracing::info!(
                                    blocks = report.blocks_checked,
                                    ssts = report.sst_files_checked,
                                    duration_ms = report.duration.as_millis(),
                                    "background scrub clean"
                                );
                            }
                        }
                        Ok(Err(e)) => tracing::warn!(%e, "background scrub failed"),
                        Err(e) => tracing::warn!(%e, "background scrub task panicked"),
                    }
                }
            });
        }
    }

    // Periodic local checkpoints: the base WAL-replay repair rebuilds a
    // corrupt partition from when no healthy replica can serve it. Per
    // node, no leader election; blocking I/O kept off the runtime.
    {
        if checkpoint_enabled {
            let ckpt_engine = Arc::clone(&engine);
            let interval = std::time::Duration::from_secs(checkpoint_interval_secs.max(1));
            let jitter = interval
                .checked_div(16)
                .map(|slice| slice.saturating_mul(u32::try_from(node_id % 16).unwrap_or(0)))
                .unwrap_or_default();
            tokio::spawn(async move {
                tokio::time::sleep(jitter).await;
                let mut ticker = tokio::time::interval(interval);
                ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    ticker.tick().await;
                    let eng = Arc::clone(&ckpt_engine);
                    let dir = checkpoint_dir.clone();
                    let now_secs = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    match tokio::task::spawn_blocking(move || {
                        checkpoint::run_checkpoint_cycle(&eng, &dir, checkpoint_keep, now_secs)
                    })
                    .await
                    {
                        Ok(Ok(path)) => {
                            metrics::counter!("coordinode_checkpoint_total").increment(1);
                            metrics::gauge!("coordinode_checkpoint_last_timestamp_seconds")
                                .set(now_secs as f64);
                            tracing::info!(checkpoint = %path.display(), "checkpoint written");
                        }
                        Ok(Err(e)) => {
                            metrics::counter!("coordinode_checkpoint_failures_total").increment(1);
                            tracing::warn!(%e, "periodic checkpoint failed");
                        }
                        Err(e) => tracing::warn!(%e, "checkpoint task panicked"),
                    }
                }
            });
        }
    }

    // Open Raft node and build database — both modes use RaftProposalPipeline.
    //
    // Three construction paths:
    //
    // 1. Standalone (no --peers): single-node Raft via StubNetworkFactory.
    //    No gRPC Raft handler needed — no peers can connect.
    //
    // 2. Cluster, node_id == 1 (bootstrap leader): open_cluster_embedded().
    //    Calls initialize(), returns a RaftGrpcHandler for the main router.
    //
    // 3. Cluster, node_id > 1 (joining node): open_joining_embedded().
    //    Does NOT call initialize(). Waits for leader to add it via
    //    `coordinode admin node join`. Returns a RaftGrpcHandler for the
    //    main router.
    //
    // In cases 2 and 3, RaftServiceServer is registered at the end of
    // router construction so inter-node Raft RPCs share the :7080 port.
    let (raft_node, raft_grpc_handler) = if let Some(ref peers_list) = peers {
        let peer_count = peers_list.len();
        if node_id == 1 {
            info!(
                peers = peer_count,
                node_id, "cluster mode: bootstrap leader (open_cluster_embedded)"
            );
            let (rn, handler) = coordinode_raft::cluster::RaftNode::open_cluster_embedded(
                node_id,
                Arc::clone(&engine),
                effective_advertise,
            )
            .await
            .map_err(|e| format!("failed to open cluster Raft node: {e}"))?;
            (rn, Some(handler))
        } else {
            info!(
                peers = peer_count,
                node_id, "cluster mode: joining node (open_joining_embedded)"
            );
            let (rn, handler) = coordinode_raft::cluster::RaftNode::open_joining_embedded(
                node_id,
                Arc::clone(&engine),
            )
            .await
            .map_err(|e| format!("failed to open joining Raft node: {e}"))?;
            (rn, Some(handler))
        }
    } else {
        info!(node_id, "standalone mode: single-node Raft (StubNetwork)");
        let rn = coordinode_raft::cluster::RaftNode::open_with_oracle(
            node_id,
            Arc::clone(&engine),
            Some(Arc::clone(&oracle)),
        )
        .await
        .map_err(|e| format!("failed to open Raft node: {e}"))?;
        (rn, None)
    };

    let raft_node = Arc::new(raft_node);

    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> = Arc::new(
        coordinode_raft::proposal::RaftProposalPipeline::new(Arc::clone(raft_node.raft())),
    );

    // no-std: spin::RwLock (drop-in).
    let database = Arc::new(parking_lot::RwLock::new(
        coordinode_embed::Database::from_engine(
            &data_dir,
            Arc::clone(&engine),
            oracle.clone(),
            Arc::clone(&pipeline),
        )
        .map_err(|e| format!("failed to open database: {e}"))?,
    ));

    // Live session registry for operational introspection
    // (SHOW SESSIONS / SHOW TRANSACTIONS). Shared between the session
    // binding (which updates it as sessions open and transactions
    // begin/end) and the query engine (which reads a snapshot). Its
    // transaction auto-abort countdown uses the same idle timeout as the
    // interactive-transaction reaper.
    let session_registry = Arc::new(coordinode_session::SessionRegistry::new(
        std::time::Duration::from_secs(interactive_txn_idle_timeout_secs),
    ));

    // Interactive-transaction tunables (ADR-042). Always resolved (the
    // config gate carries the built-in defaults: 30s idle timeout,
    // 256 MiB buffered-write ceiling per open transaction).
    {
        let mut db = database.write();
        db.set_interactive_idle_timeout(std::time::Duration::from_secs(
            interactive_txn_idle_timeout_secs,
        ));
        db.set_max_interactive_txn_bytes(interactive_txn_max_bytes as usize);
        // AFTER COMMIT trigger dispatch knobs (R192) from the config file.
        // The same setter is the runtime `setParameters` seam.
        db.set_trigger_dispatch_config(trigger_dispatch_cfg);
        // Let SHOW SESSIONS / SHOW TRANSACTIONS read the live registry.
        // The annotated binding coerces the concrete `Arc<SessionRegistry>`
        // to the trait object the setter expects.
        let ops_view: Arc<dyn coordinode_core::operations::OperationsView> =
            session_registry.clone();
        db.set_operations_view(ops_view);
        // Extension-op handlers registered by a downstream distribution. The
        // planner resolves an op to its handler by name while building the
        // plan, so this costs nothing on the row path. CE registers none and
        // the registry stays empty.
        for (name, handler) in &extensions.query_extensions {
            db.register_extension(name.clone(), Arc::clone(handler));
            info!(extension = %name, "query extension registered");
        }
    }

    // Idle reaper: periodically drop interactive transactions left
    // untouched past the idle timeout, so an abandoned client cannot pin
    // a transaction (and its snapshot) forever. The countdown surfaced by
    // SHOW TRANSACTIONS hits zero exactly when a transaction is reaped
    // here.
    {
        let reaper_registry = Arc::clone(&session_registry);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(1));
            loop {
                tick.tick().await;
                let _ = reaper_registry.reap_idle();
            }
        });
    }

    // Per-shard consumer-retention registry (ADR-028). Constructing it
    // makes the registry the source of the MVCC GC watermark: it
    // activates the documented retention window (`now - 7d`, so
    // `AS OF TIMESTAMP` history is retained) and, once CDC / backup
    // consumers register, holds older versions / oplog segments back
    // for them. The background service runs batched heartbeats + TTL
    // eviction and advances the window with the wall clock. Both
    // standalone and cluster modes drive it through the same Raft
    // pipeline. Held for the process lifetime.
    // Operator overrides for the retention window + background
    // cadences arrive as `coordinode serve` flags; `None` keeps the
    // built-in defaults (7-day window, 100 ms heartbeat, 1 s eviction).
    let (consumer_registry, _registry_bg) = registry::build_consumer_registry(
        Arc::clone(&engine),
        Arc::clone(&pipeline),
        node_id,
        registry::RegistryTuning {
            retention_window_secs,
            heartbeat_window_ms: registry_heartbeat_ms,
            eviction_interval_ms: registry_eviction_ms,
        },
    );

    // Vector index observability: publish per-index
    // serving state + freshness lag as Prometheus gauges. Scrape-style
    // periodic collector — the lag needs the engine's current
    // committed HLC, which is only meaningful at sample time.
    {
        let db_metrics = Arc::clone(&database);
        let engine_metrics = Arc::clone(&engine);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(15));
            loop {
                tick.tick().await;
                let committed = engine_metrics.snapshot();
                let health = db_metrics.read().vector_index_registry().all_health();
                for (label, property, state) in health {
                    let code = match &state {
                        coordinode_vector::health::IndexHealthState::Ready { .. } => 0.0,
                        coordinode_vector::health::IndexHealthState::Rebuilding { .. } => 1.0,
                        coordinode_vector::health::IndexHealthState::Offline { .. } => 2.0,
                    };
                    metrics::gauge!(
                        "coordinode_vector_index_state",
                        "label" => label.clone(),
                        "property" => property.clone(),
                    )
                    .set(code);
                    let lag = state
                        .indexed_hlc()
                        .map(|h| committed.saturating_sub(h))
                        .unwrap_or(0);
                    metrics::gauge!(
                        "coordinode_vector_index_lag_hlc",
                        "label" => label,
                        "property" => property,
                    )
                    .set(lag as f64);
                }
            }
        });
    }

    let raft_node_shared: Option<Arc<coordinode_raft::cluster::RaftNode>> =
        Some(Arc::clone(&raft_node));

    // Hand the RaftNode to the scrub task so WAL-replay repair can read
    // the oplog. Set-once; the scrub's first (jitter-delayed) run is well
    // after this point.
    let _ = raft_slot.set(Arc::clone(&raft_node));

    // Refresh node-local derived state when replicated entries
    // apply: property values are encoded against interner ids,
    // and a follower that never refreshes its in-memory interner
    // resolves every replicated property to null. The refresh is
    // a cheap length pre-check unless the mapping actually grew.
    if peers.is_some() {
        let mut applied_rx = raft_node.subscribe_applied();
        let db = Arc::clone(&database);
        tokio::spawn(async move {
            while applied_rx.changed().await.is_ok() {
                let guard = db.read();
                if let Err(e) = guard.refresh_field_interner() {
                    tracing::warn!(%e, "field interner refresh failed");
                }
                // Replicated CREATE VECTOR INDEX definitions are
                // brought live here: register + local HNSW rebuild
                // (the graph itself is never replicated).
                match guard.refresh_vector_indexes() {
                    Ok(0) => {}
                    Ok(n) => tracing::info!(n, "vector indexes brought live from apply"),
                    Err(e) => tracing::warn!(%e, "vector index refresh failed"),
                }
            }
        });
    }

    // Drive AFTER COMMIT trigger dispatch on the Raft leader (R192,
    // ADR-026). The event queue (`trigger_pending:`) is Raft-replicated,
    // so every node sees the same backlog; gating execution on the lease
    // holder makes each event fire exactly once cluster-wide (the body's
    // writes have to go through the leader's pipeline anyway). Woken by
    // each applied entry (covers fresh enqueues) and a periodic tick
    // (covers retry backoff timers). The blocking dispatch runs off the
    // async runtime so a long body never stalls consensus.
    if peers.is_some() {
        let db = Arc::clone(&database);
        let rn = Arc::clone(&raft_node);
        let mut applied_rx = rn.subscribe_applied();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(trigger_dispatch_interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    changed = applied_rx.changed() => {
                        if changed.is_err() {
                            break; // RaftNode dropped — shut the worker down.
                        }
                    }
                    _ = tick.tick() => {}
                }
                if rn.current_leader() != Some(rn.node_id()) {
                    continue;
                }
                let db2 = Arc::clone(&db);
                match tokio::task::spawn_blocking(move || {
                    db2.read().dispatch_after_commit_triggers()
                })
                .await
                {
                    Ok(report) => {
                        for e in &report.errors {
                            tracing::warn!("after-commit trigger dispatch: {e}");
                        }
                    }
                    Err(e) => {
                        tracing::warn!(%e, "after-commit dispatch task join error")
                    }
                }
            }
        });
    }

    let query_registry = Arc::new(coordinode_query::advisor::QueryRegistry::new());
    let nplus1_detector = Arc::new(coordinode_query::advisor::nplus1::NPlus1Detector::new());

    let graph_service = services::graph::GraphServiceImpl::new(Arc::clone(&database));
    let schema_service = services::schema::SchemaServiceImpl::new(Arc::clone(&database));
    let cypher_service = {
        let svc = services::cypher::CypherServiceImpl::new(
            Arc::clone(&database),
            Arc::clone(&query_registry),
            Arc::clone(&nplus1_detector),
        );
        if let Some(ref rn) = raft_node_shared {
            svc.with_raft_node(Arc::clone(rn))
        } else {
            svc
        }
    };
    let vector_service = services::vector::VectorServiceImpl::new(Arc::clone(&database));
    let text_service = services::text::TextServiceImpl::new(Arc::clone(&database));
    let health_service = services::health::HealthServiceImpl;
    // CDC service: tails oplog/<shard>/ dir. Empty stream in embedded mode
    // (no oplog); populated in Raft cluster mode (LogStore writes oplog).
    let cdc_service = services::cdc::ChangeEventServiceImpl::new(
        std::path::PathBuf::from(&data_dir),
        consumer_registry,
        // Operator-tunable CDC consumer TTL (seconds → ms); saturating
        // so an absurdly large window means "effectively never reclaim".
        cdc_consumer_ttl_secs
            .map(|s| s.saturating_mul(1000))
            .unwrap_or(services::cdc::DEFAULT_CONSUMER_TTL_MS),
    );

    // ClusterService: cluster join/leave lifecycle.
    // Available only in cluster mode (requires a RaftNode).
    let cluster_service = raft_node_shared
        .as_ref()
        .map(|rn| services::cluster::ClusterServiceImpl::new(Arc::clone(rn)));

    // BlobService shares the same storage engine as the Database.
    // Read guard is dropped immediately — only need engine_shared().
    let blob_engine = database.read().engine_shared();
    let blob_service = services::blob::BlobServiceImpl::new(blob_engine);

    // Spawn operational HTTP server (default :7084, configurable via --ops-addr).
    let ops_sock: SocketAddr = ops_addr.parse()?;
    tokio::spawn(async move {
        if let Err(e) = ops::start_ops_server(ops_sock).await {
            tracing::error!("ops server error: {e}");
        }
    });

    // Spawn embedded REST/JSON proxy (default :7081, configurable via --rest-addr).
    // Transcodes HTTP/JSON requests to gRPC via google.api.http annotations.
    // Compiled only when the `rest-proxy` feature is enabled (default).
    // Disable for embedded/mobile builds: --no-default-features --features vector,full-text
    #[cfg(feature = "rest-proxy")]
    {
        use structured_proxy::config::{
            DescriptorSource, HealthConfig, ListenConfig, MetricsConfig, ProxyConfig,
            ServiceConfig, UpstreamConfig,
        };
        static DESCRIPTOR_BYTES: &[u8] = include_bytes!("../../../coordinode.descriptor.bin");
        let grpc_upstream = format!("http://127.0.0.1:{}", addr.port());
        // The proxy would otherwise mount its own /health and /metrics on the
        // REST port, reporting proxy state. CoordiNode publishes those for the
        // database itself on the ops port, which is where the documented
        // endpoints live, so keep the proxy off both paths. Both structs are
        // #[non_exhaustive], so start from the default and clear the flag.
        let mut proxy_health = HealthConfig::default();
        proxy_health.enabled = false;
        let mut proxy_metrics = MetricsConfig::default();
        proxy_metrics.enabled = false;
        // structured-proxy 2.0.1 makes the embedded-constructed config structs
        // hand-buildable again (no longer #[non_exhaustive]), so the config is built
        // programmatically. serve() is gone in 2.x; the proxy exposes an axum Router
        // that we bind and serve here.
        let config = ProxyConfig {
            upstream: UpstreamConfig {
                default: grpc_upstream,
            },
            descriptors: vec![DescriptorSource::Embedded {
                bytes: DESCRIPTOR_BYTES,
            }],
            listen: ListenConfig {
                http: rest_addr.clone(),
            },
            service: ServiceConfig {
                name: "coordinode".into(),
            },
            health: proxy_health,
            metrics: proxy_metrics,
            aliases: vec![],
            openapi: None,
            auth: None,
            shield: None,
            oidc_discovery: None,
            maintenance: Default::default(),
            cors: Default::default(),
            logging: Default::default(),
            streaming: Default::default(),
            metrics_classes: vec![],
            forwarded_headers: vec![
                "authorization".into(),
                "dpop".into(),
                "x-request-id".into(),
                "x-forwarded-for".into(),
                "x-forwarded-proto".into(),
                "x-real-ip".into(),
                "user-agent".into(),
                "accept-language".into(),
                "idempotency-key".into(),
            ],
        };
        let proxy = structured_proxy::ProxyServer::from_config(config);
        tokio::spawn(async move {
            match proxy.router() {
                Ok(router) => match tokio::net::TcpListener::bind(rest_addr.as_str()).await {
                    Ok(listener) => {
                        if let Err(e) = axum::serve(listener, router).await {
                            tracing::error!("REST proxy serve error: {e}");
                        }
                    }
                    Err(e) => tracing::error!("REST proxy bind error: {e}"),
                },
                Err(e) => tracing::error!("REST proxy router build error: {e}"),
            }
        });
    }

    // PostgreSQL wire-protocol frontend: opt-in (only when an address is
    // configured), trust authentication. Shares the same database handle
    // as the gRPC/REST services so SQL over the wire sees identical state.
    if let Some(pg_addr) = pg_addr {
        match pg_addr.parse::<SocketAddr>() {
            Ok(pg_sockaddr) => {
                let pg_db = Arc::clone(&database);
                tokio::spawn(async move {
                    if let Err(e) = pg::serve(pg_sockaddr, pg_db).await {
                        tracing::error!("PostgreSQL wire server error: {e}");
                    }
                });
            }
            Err(e) => tracing::error!(addr = %pg_addr, "invalid --pg-addr: {e}"),
        }
    }

    info!(
        port = addr.port(),
        node_id,
        mode = %mode,
        "gRPC server listening"
    );

    // Graceful shutdown: wait for SIGTERM (Docker / test harness) or Ctrl+C.
    // When the signal fires, `serve_with_shutdown` stops accepting new
    // connections and waits for in-flight RPCs to complete before returning.
    // The returned future resolves → all Arc<Database> / Arc<RaftNode> drop
    // → StorageEngine::Drop flushes all memtables to SST files.
    #[cfg(unix)]
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .map_err(|e| format!("failed to install SIGTERM handler: {e}"))?;

    let shutdown = async move {
        #[cfg(unix)]
        tokio::select! {
            _ = sigterm.recv() => {
                info!("SIGTERM received — initiating graceful shutdown");
            }
            _ = tokio::signal::ctrl_c() => {
                info!("Ctrl+C received — initiating graceful shutdown");
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
            info!("Ctrl+C received — initiating graceful shutdown");
        }
    };

    // Network limits: per-request timeout, per-connection in-flight cap,
    // and HTTP/2 keepalive pings. Each applies only when configured.
    let mut server = Server::builder();
    if let Some(secs) = request_timeout_secs {
        server = server.timeout(std::time::Duration::from_secs(secs));
    }
    if let Some(n) = max_connections {
        server = server.concurrency_limit_per_connection(n);
    }
    if let Some(secs) = http2_keepalive_secs {
        server = server.http2_keepalive_interval(Some(std::time::Duration::from_secs(secs)));
    }

    // TLS / mTLS for inter-node + client gRPC. Enabled when a cert+key are
    // configured; the pure-Rust crypto provider was installed above so
    // tonic's rustls config uses it (no C FFI). With require-client-auth,
    // verify peer certs against the CA (mutual TLS).
    if let (Some(cert_path), Some(key_path)) = (tls_cert.as_ref(), tls_key.as_ref()) {
        use tonic::transport::{Certificate, Identity, ServerTlsConfig};
        let cert =
            std::fs::read(cert_path).map_err(|e| format!("read tls cert {cert_path}: {e}"))?;
        let key = std::fs::read(key_path).map_err(|e| format!("read tls key {key_path}: {e}"))?;
        // Read the CA once if configured: it verifies connecting clients
        // (server-side mTLS) and the peers we dial (outbound client side).
        let ca = match tls_ca.as_ref() {
            Some(ca_path) => {
                Some(std::fs::read(ca_path).map_err(|e| format!("read tls ca {ca_path}: {e}"))?)
            }
            None => None,
        };
        let mut tls = ServerTlsConfig::new().identity(Identity::from_pem(&cert, &key));
        if tls_require_client_auth {
            let ca = ca
                .as_ref()
                .ok_or("--tls-require-client-auth requires --tls-ca")?;
            tls = tls.client_ca_root(Certificate::from_pem(ca));
        }
        server = server
            .tls_config(tls)
            .map_err(|e| format!("tls config: {e}"))?;
        // Outbound inter-node TLS (Raft network + segment drain): verify
        // peers against the CA and present our identity for mutual TLS.
        // Without a CA the node serves TLS but cannot verify peers, so a
        // TLS cluster would not interconnect; keep outbound plaintext and
        // warn rather than dial unverified.
        match ca {
            Some(ca) => {
                let client_tls = coordinode_wire::build_client_tls(&ca, Some((cert, key)));
                coordinode_wire::set_wire_client_tls(client_tls);
            }
            None => tracing::warn!(
                "gRPC TLS enabled without --tls-ca: outbound peer connections stay \
                         plaintext; set --tls-ca to interconnect a TLS cluster"
            ),
        }
        info!(mtls = tls_require_client_auth, "gRPC TLS enabled");
    }

    // Cap the decoded size of any single request to guard against
    // unbounded-allocation messages. Applied to every service.
    let max_req_bytes = max_request_size_mb.saturating_mul(1024 * 1024);

    // Publish the running server to everything registered on the builder.
    // Placement defaults to the single-shard, single-node strategy; a
    // downstream distribution replaces it via `with_placement`.
    let (routing, topology) = extensions.placement.clone().unwrap_or_else(|| {
        (
            Arc::new(coordinode_cluster::SingleShardRouting::new()),
            Arc::new(coordinode_cluster::SingleNodeTopology::from_storage(
                &storage_config,
            )),
        )
    });
    let ctx = crate::builder::ServerContext::new(
        node_id,
        data_dir.clone(),
        cluster_mode,
        max_req_bytes,
        Arc::clone(&database),
        Arc::clone(&engine),
        Arc::clone(&raft_node),
        Arc::clone(&session_registry),
        routing,
        topology,
        extension_config,
    );

    // Bring the node into a registered non-built-in mode before it accepts
    // traffic. `full` has no handler and nothing runs here.
    if let Some(handler) = extension_mode {
        handler.start(&ctx)?;
        info!(mode = %mode, "serve mode handler started");
    }

    for task in &extensions.background_tasks {
        task.start(&ctx);
    }

    // Services are collected into a `Routes` first, so a downstream
    // distribution can contribute its own before the router is assembled.
    // `Server::add_routes` and `Server::add_service` both end at
    // `Router::new`, so registration order is what it always was.
    let mut routes = tonic::service::Routes::builder();
    routes
        .add_service(
            proto::graph::graph_service_server::GraphServiceServer::new(graph_service)
                .max_decoding_message_size(max_req_bytes),
        )
        .add_service(
            proto::graph::schema_service_server::SchemaServiceServer::new(schema_service)
                .max_decoding_message_size(max_req_bytes),
        )
        .add_service(
            proto::query::cypher_service_server::CypherServiceServer::new(cypher_service)
                .max_decoding_message_size(max_req_bytes),
        )
        .add_service(
            proto::session::session_service_server::SessionServiceServer::new(
                services::session::SessionSvc::new(
                    Arc::clone(&database),
                    Arc::clone(&session_registry),
                ),
            )
            .max_decoding_message_size(max_req_bytes),
        )
        .add_service(
            proto::query::vector_service_server::VectorServiceServer::new(vector_service)
                .max_decoding_message_size(max_req_bytes),
        )
        .add_service(
            proto::query::text_service_server::TextServiceServer::new(text_service)
                .max_decoding_message_size(max_req_bytes),
        )
        .add_service(
            proto::health::health_service_server::HealthServiceServer::new(health_service)
                .max_decoding_message_size(max_req_bytes),
        )
        .add_service(
            proto::graph::blob_service_server::BlobServiceServer::new(blob_service)
                .max_decoding_message_size(max_req_bytes),
        )
        .add_service(
            proto::replication::cdc::change_stream_service_server::ChangeStreamServiceServer::new(
                cdc_service,
            )
            .max_decoding_message_size(max_req_bytes),
        );

    // Register ClusterService only in cluster mode (requires Raft node).
    if let Some(cs) = cluster_service {
        routes.add_service(
            proto::admin::cluster::cluster_service_server::ClusterServiceServer::new(cs)
                .max_decoding_message_size(max_req_bytes),
        );
        info!("ClusterService registered — cluster join/leave management available");
    }

    // Register cluster-only inter-node services in cluster mode — embedded
    // into :7080 so inter-node RPCs share the main gRPC port (no separate
    // server). Gated on cluster mode: raft_grpc_handler is Some only when
    // peers are configured.
    if let Some(handler) = raft_grpc_handler {
        use coordinode_raft::proto::replication::raft_service_server::RaftServiceServer;
        routes.add_service(RaftServiceServer::new(handler));
        info!(node_id, "RaftService registered on :7080 (shared port)");

        // SegmentTransferService: receive bulk segment pushes (replication
        // repair, operator-commanded migration, node resync) and install
        // them into local storage via the engine-backed sink.
        use coordinode_replicate::segment_store::SegmentInstaller;
        use coordinode_replicate::transfer::proto::segment_transfer_service_server::SegmentTransferServiceServer;
        use coordinode_replicate::transfer::SegmentTransferHandler;
        let segment_handler =
            SegmentTransferHandler::new(Arc::new(SegmentInstaller::new(Arc::clone(&engine))));
        routes.add_service(
            SegmentTransferServiceServer::new(segment_handler)
                .max_decoding_message_size(max_req_bytes),
        );
        info!(
            node_id,
            "SegmentTransferService registered on :7080 (shared port)"
        );
    }

    // Services contributed by a downstream distribution, added after the
    // built-in ones. CE registers no providers and this loop does nothing.
    for provider in &extensions.grpc_services {
        provider.register(&ctx, &mut routes);
    }

    // NodeInfoLayer: inject x-coordinode-node / x-coordinode-hops /
    // x-coordinode-load response headers on every gRPC response.
    let mut server = server.layer(grpc::NodeInfoLayer::new(node_id));
    let router = server.add_routes(routes.routes());

    router.serve_with_shutdown(addr, shutdown).await?;

    Ok(())
}
