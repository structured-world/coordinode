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

/// Delay before a periodic task's first run: `node_id mod 16` sixteenths of
/// its interval, so a fleet does not run it in lockstep. The product is at
/// most the interval itself, so it cannot overflow.
fn startup_jitter(interval: std::time::Duration, node_id: u64) -> std::time::Duration {
    // Below 16, so the conversion is lossless.
    let slot = (node_id % 16) as u32;
    interval / 16 * slot
}

/// The URL the in-process REST proxy reaches the gRPC listener bound at
/// `bound` by: that address, or the loopback of its family when the listener
/// takes every interface. A fixed `127.0.0.1` misses a listener bound to an
/// IPv6 or a single interface address.
#[cfg(feature = "rest-proxy")]
fn local_upstream(bound: SocketAddr) -> String {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    let ip = match bound.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(ip) if ip.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        ip => ip,
    };
    format!("http://{}", SocketAddr::new(ip, bound.port()))
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

    // Select the TLS crypto provider before any TLS config is built. The stock
    // server takes the pure-Rust one (no C FFI); a downstream distribution can
    // have registered another on the builder.
    coordinode_wire::tls::install_crypto_provider(
        extensions
            .crypto_provider
            .clone()
            .unwrap_or_else(coordinode_wire::tls::rustcrypto_provider),
    );

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
    let mut storage_config = match cfg.resolve_storage_config() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: storage endpoints: {e}");
            std::process::exit(1);
        }
    };
    let page_ecc_requested = cfg.page_ecc_requested();
    let sizes = match cfg.byte_sizes() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

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
    let statement_defaults = cfg.statement_defaults();
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
        request_timeout_secs,
        http2_keepalive_secs,
        // Converted to bytes above (`sizes`).
        max_request_size_mb: _,
        cache_size_mb: _,
        write_buffer_mb: _,
        // Applied to the engine through resolve_storage_config above.
        retention_window_secs: _,
        max_invariant_claims: _,
        max_commits_in_flight: _,
        snapshot_wait_ms: _,
        min_free_bytes: _,
        resume_free_bytes: _,
        node_shard: _,
        registry_heartbeat_ms,
        registry_eviction_ms,
        cdc_heartbeat_interval_ms,
        cdc_batch_size,
        cdc_buffer_bytes,
        // Already captured above (statement_defaults) before the move.
        default_read_concern: _,
        default_read_preference: _,
        default_write_concern: _,
        interactive_txn_idle_timeout_secs,
        interactive_txn_max_bytes,
        peers: peers_vec,
        membership_change_timeout_secs,
        join_readiness_lag_entries,
        join_timeout_secs,
        raft_snapshot_entries,
        raft_snapshot_log_bytes,
        raft_snapshot_min_interval_secs,
        planner_stats_ttl_secs,
        vector_build_wait_ms,
        vector_retired_bytes_budget,
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

    // A consumer's own liveness timeout is checked against the heartbeat
    // interval when it registers; nothing is shared to check here.
    let cdc_tuning = {
        let default = services::cdc::CdcStreamTuning::default();
        services::cdc::CdcStreamTuning {
            heartbeat_interval: cdc_heartbeat_interval_ms
                .map_or(default.heartbeat_interval, |ms| {
                    std::time::Duration::from_millis(ms.get())
                }),
            batch_size: cdc_batch_size.unwrap_or(default.batch_size),
            buffer_bytes: cdc_buffer_bytes.unwrap_or(default.buffer_bytes),
        }
    };

    logging::init_logging();

    // Raise the open-file-descriptor limit before opening storage: the
    // engine keeps many files open at once. Honour an explicit target or
    // raise the soft limit to the hard limit.
    if let Some((soft, hard)) = set_nofile_limit(nofile) {
        info!(soft, hard, "file-descriptor limit");
    }

    let addr: SocketAddr = grpc_addr.parse()?;
    // Bind the gRPC port before opening storage: a port already in use fails
    // the start at once, rather than after the store has been opened (and
    // possibly recovered) only to be abandoned. A client that connects before
    // the services are wired waits in the accept backlog. TCP_NODELAY is the
    // value `serve_with_shutdown(addr)` would have applied.
    let grpc_incoming = tonic::transport::server::TcpIncoming::bind(addr)
        .map_err(|e| format!("cannot bind the gRPC address {addr}: {e}"))?
        .with_nodelay(Some(true));
    #[cfg(feature = "rest-proxy")]
    let grpc_upstream = local_upstream(grpc_incoming.local_addr()?);
    // The ops and REST ports are claimed here too. A node running without its
    // ops listener has no /ready of its own, so a health check against that
    // port would get its answer from whatever holds it; one running without
    // REST looks healthy while that API reaches nothing.
    let ops_sock: SocketAddr = ops_addr.parse()?;
    let ops_listener = tokio::net::TcpListener::bind(ops_sock)
        .await
        .map_err(|e| format!("cannot bind the ops address {ops_sock}: {e}"))?;
    #[cfg(feature = "rest-proxy")]
    let rest_listener = tokio::net::TcpListener::bind(rest_addr.as_str())
        .await
        .map_err(|e| format!("cannot bind the REST address {rest_addr}: {e}"))?;
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
    // `raft_node_shared` provides the follower-read fence, ClusterService
    // administration, and ensures consistent apply ordering via oracle.

    // Common setup: open storage engine + timestamp oracle. The storage
    // topology was resolved from config above (a multi-endpoint list or
    // the single-endpoint `data_dir` desugar); apply the cache / write-
    // buffer size overrides on top.
    if let Some(bytes) = sizes.cache_bytes {
        storage_config.block_cache_bytes = bytes;
    }
    if let Some(bytes) = sizes.write_buffer_bytes {
        storage_config.max_write_buffer_bytes = bytes;
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
            let jitter = startup_jitter(interval, node_id);
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
                            let mut corrupt = std::collections::HashSet::new();
                            if report.has_errors() {
                                metrics::counter!("coordinode_scrub_errors_total")
                                    .increment(report.errors.len() as u64);
                                for err in &report.errors {
                                    tracing::error!(
                                        partition = err.partition.name(),
                                        detail = %err.message,
                                        "scrub detected corruption"
                                    );
                                    corrupt.insert(err.partition);
                                }
                            }
                            // A rebuild a crash interrupted left its tree with
                            // an unknown part of its data; the tree scrubs
                            // clean, so only its intent says so.
                            match scrub_engine.pending_rebuilds() {
                                Ok(pending) => {
                                    for part in pending {
                                        tracing::error!(
                                            partition = part.name(),
                                            "partition rebuild was interrupted"
                                        );
                                        corrupt.insert(part);
                                    }
                                }
                                Err(e) => tracing::warn!(%e, "reading rebuild intents failed"),
                            }
                            if !corrupt.is_empty() {
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

                                    // 2) Rebuild from the latest local checkpoint and
                                    // the Raft log, with the applies paused.
                                    let Some(raft) = scrub_raft.get() else {
                                        tracing::warn!(
                                            partition = part.name(),
                                            "no replica and no Raft log yet — cannot repair"
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
                                    let raft = Arc::clone(raft);
                                    match tokio::task::spawn_blocking(move || {
                                        raft.rebuild_partition_from_checkpoint(&ckpt, part)
                                    })
                                    .await
                                    {
                                        Ok(Ok(())) => {
                                            metrics::counter!("coordinode_scrub_wal_repairs_total")
                                                .increment(1);
                                            tracing::info!(
                                                partition = part.name(),
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
            let jitter = startup_jitter(interval, node_id);
            tokio::spawn(async move {
                tokio::time::sleep(jitter).await;
                let mut ticker = tokio::time::interval(interval);
                ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    ticker.tick().await;
                    // A checkpoint copies the store to disk; with the disk
                    // below its reserve it waits for the next tick.
                    if ckpt_engine.space().is_paused() {
                        tracing::warn!(
                            "periodic checkpoint skipped: disk below its free-space reserve"
                        );
                        continue;
                    }
                    let eng = Arc::clone(&ckpt_engine);
                    let dir = checkpoint_dir.clone();
                    let now_secs = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    match tokio::task::spawn_blocking(move || {
                        let path = checkpoint::run_checkpoint_cycle(
                            &eng,
                            &dir,
                            checkpoint_keep,
                            now_secs,
                        )?;
                        // A rebuild from this checkpoint replays the Raft log
                        // from what its trees lack; the log keeps that.
                        let floor =
                            coordinode_storage::engine::core::StorageEngine::checkpoint_raft_floor(
                                &path,
                            )
                            .map_err(|e| format!("read the checkpoint's raft floor: {e}"))?;
                        eng.set_raft_log_keep_from(floor);
                        Ok::<_, String>(path)
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
    let defaults = coordinode_raft::cluster::SnapshotTriggerConfig::default();
    let snapshots = coordinode_raft::cluster::SnapshotTriggerConfig {
        logs_since_last: raft_snapshot_entries.map_or(defaults.logs_since_last, |n| n.get()),
        log_bytes: raft_snapshot_log_bytes.map_or(defaults.log_bytes, |n| n.get()),
        min_interval: raft_snapshot_min_interval_secs.map_or(defaults.min_interval, |n| {
            std::time::Duration::from_secs(n.get())
        }),
    };
    // A server is not embedded in an application with a format of its own:
    // its host epoch is zero. It hosts the one group a deployment is formed
    // with.
    let options = coordinode_raft::cluster::NodeOptions {
        snapshots,
        host_epoch: 0,
        group: coordinode_raft::cluster::GroupId::FORMING,
        connections: Default::default(),
    };
    let (raft_node, raft_grpc_handler) = if let Some(ref peers_list) = peers {
        let peer_count = peers_list.len();
        if node_id == 1 {
            info!(
                peers = peer_count,
                node_id, "cluster mode: bootstrap leader (open_cluster_embedded)"
            );
            let (rn, handler) =
                coordinode_raft::cluster::RaftNode::open_cluster_embedded_with_options(
                    node_id,
                    Arc::clone(&engine),
                    effective_advertise,
                    options,
                )
                .await
                .map_err(|e| format!("failed to open cluster Raft node: {e}"))?;
            (rn, Some(handler))
        } else {
            info!(
                peers = peer_count,
                node_id, "cluster mode: joining node (open_joining_embedded)"
            );
            let (rn, handler) =
                coordinode_raft::cluster::RaftNode::open_joining_embedded_with_options(
                    node_id,
                    Arc::clone(&engine),
                    options,
                )
                .await
                .map_err(|e| format!("failed to open joining Raft node: {e}"))?;
            (rn, Some(handler))
        }
    } else {
        info!(node_id, "standalone mode: single-node Raft (StubNetwork)");
        let rn = coordinode_raft::cluster::RaftNode::open_with_oracle_and_options(
            node_id,
            Arc::clone(&engine),
            Some(Arc::clone(&oracle)),
            options,
        )
        .await
        .map_err(|e| format!("failed to open Raft node: {e}"))?;
        (rn, None)
    };

    if let Some(secs) = membership_change_timeout_secs {
        raft_node.set_membership_settle_timeout(std::time::Duration::from_secs(secs));
    }
    if let Some(entries) = join_readiness_lag_entries {
        raft_node.set_join_readiness_lag(entries);
    }
    if let Some(secs) = join_timeout_secs {
        raft_node.set_join_timeout(std::time::Duration::from_secs(secs.get()));
    }
    let raft_node = Arc::new(raft_node);

    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(raft_node.pipeline());

    let mut database = coordinode_embed::Database::from_engine(
        &data_dir,
        Arc::clone(&engine),
        oracle.clone(),
        Arc::clone(&pipeline),
    )
    .map_err(|e| format!("failed to open database: {e}"))?;
    if let Some(secs) = planner_stats_ttl_secs {
        database.set_stats_ttl(std::time::Duration::from_secs(secs));
    }
    if let Some(ms) = vector_build_wait_ms {
        database.set_vector_build_wait(std::time::Duration::from_millis(ms));
    }
    if let Some(bytes) = vector_retired_bytes_budget {
        // A budget past the address space is no bound at all, which is what
        // `usize::MAX` means.
        database.set_vector_retired_bytes_budget(usize::try_from(bytes).unwrap_or(usize::MAX));
    }
    // What a statement executes under when neither it nor its session names a
    // concern.
    database.set_read_concern(statement_defaults.read_concern);
    database.set_write_concern(statement_defaults.write_concern);
    // no-std: spin::RwLock (drop-in).
    let database = Arc::new(parking_lot::RwLock::new(database));

    // Live session registry for operational introspection
    // (SHOW SESSIONS / SHOW TRANSACTIONS). Shared between the session
    // binding (which updates it as sessions open and transactions
    // begin/end) and the query engine (which reads a snapshot). Its
    // transaction auto-abort countdown uses the same idle timeout as the
    // interactive-transaction reaper.
    let session_registry = Arc::new(coordinode_session::SessionRegistry::new(
        std::time::Duration::from_secs(interactive_txn_idle_timeout_secs),
    ));

    // Interactive-transaction tunables. Always resolved (the
    // config gate carries the built-in defaults: 30s idle timeout,
    // 256 MiB buffered-write ceiling per open transaction).
    let interactive_begun = Arc::new(tokio::sync::Notify::new());
    {
        let mut db = database.write();
        db.set_interactive_idle_timeout(std::time::Duration::from_secs(
            interactive_txn_idle_timeout_secs,
        ));
        let begun = Arc::clone(&interactive_begun);
        db.set_interactive_begun_hook(Arc::new(move || begun.notify_one()));
        db.set_max_interactive_txn_bytes(interactive_txn_max_bytes as usize);
        // AFTER COMMIT trigger dispatch knobs from the config file.
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
        // Procedures a downstream distribution adds to the CE catalog. A name
        // clash stops startup: serving a different procedure than the one
        // registered would be worse than not serving.
        for procedure in &extensions.procedures {
            db.register_procedure(Arc::clone(procedure))?;
            info!(procedure = %procedure.signature().name, "procedure registered");
        }
    }

    // Idle reaper: roll back interactive transactions left untouched past
    // the idle timeout, so an abandoned client cannot pin a transaction (and
    // its snapshot) forever. Asleep while none is open.
    crate::txn_reaper::spawn(
        Arc::clone(&database),
        Arc::clone(&session_registry),
        std::time::Duration::from_secs(interactive_txn_idle_timeout_secs),
        interactive_begun,
        std::time::Duration::from_secs(1),
    );

    // Per-shard consumer-retention registry: every change-stream consumer is
    // registered here with its retention policy, over the history this node
    // holds (the Raft log, the MVCC store). The background service flushes
    // heartbeats and ends a BOUNDED registration whose bound is crossed. Both
    // standalone and cluster modes drive it through the same Raft pipeline.
    // Held for the process lifetime. Operator overrides for the background
    // cadences arrive from the config file; `None` keeps the built-in
    // defaults (1 s heartbeat window, 1 s between sweeps).
    let retention_source: Arc<dyn coordinode_replicate::RetentionSource> = {
        let applied_node = Arc::clone(&raft_node);
        Arc::new(registry::NodeRetentionSource::new(
            Arc::clone(&engine),
            coordinode_raft::storage::raft_oplog_dirs(&engine, 0)
                .map_err(|e| format!("oplog directories: {e}"))?
                .all,
            raft_node.retained_floor(),
            Arc::new(move || applied_node.applied_through()),
        ))
    };
    let (consumer_registry, _registry_bg) = registry::build_consumer_registry(
        Arc::clone(&engine),
        Arc::clone(&pipeline),
        retention_source,
        registry::RegistryTuning {
            heartbeat_window_ms: registry_heartbeat_ms,
            eviction_interval_ms: registry_eviction_ms,
        },
    );

    // Vector index observability: per-index serving state + freshness lag as
    // Prometheus gauges. The lag needs the engine's current committed HLC,
    // which is only meaningful at sample time, so a scrape samples them.
    let sample_gauges: ops::SampleGauges = {
        let db_metrics = Arc::clone(&database);
        let engine_metrics = Arc::clone(&engine);
        let version_node = Arc::clone(&raft_node);
        Arc::new(move || {
            // This member's version, its group's, and whether it is
            // read-only or its group paused: sampled per scrape.
            {
                let report = version_node.version_report();
                metrics::gauge!("coordinode_version_engine_format").set(report.pair.engine as f64);
                metrics::gauge!("coordinode_version_host_epoch").set(report.pair.host_epoch as f64);
                if let Some(group) = report.group_pair {
                    metrics::gauge!("coordinode_version_group_engine_format")
                        .set(group.pair.engine as f64);
                    metrics::gauge!("coordinode_version_group_host_epoch")
                        .set(group.pair.host_epoch as f64);
                }
                metrics::gauge!("coordinode_version_read_only")
                    .set(if report.read_only.is_some() { 1.0 } else { 0.0 });
                metrics::gauge!("coordinode_version_pause_seconds")
                    .set(report.pause_ms.map_or(0.0, |ms| ms as f64 / 1000.0));
            }
            {
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
                    // Clamped at zero on purpose: the index may pass
                    // `committed` after it was sampled, which is no lag.
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
                // Neighbour-list publication: memory replaced lists still
                // hold for running searches, the age of the oldest of those,
                // contention, and writers held off by the budget.
                let publication = db_metrics
                    .read()
                    .vector_index_registry()
                    .all_publication_stats();
                for (label, property, s) in publication {
                    let gauges: [(&'static str, f64); 5] = [
                        (
                            "coordinode_vector_index_retired_bytes",
                            s.retired_bytes as f64,
                        ),
                        (
                            "coordinode_vector_index_retired_lists",
                            s.retired_lists as f64,
                        ),
                        (
                            "coordinode_vector_index_oldest_operation_seconds",
                            s.oldest_operation.as_secs_f64(),
                        ),
                        (
                            "coordinode_vector_index_retired_nodes",
                            s.retired_nodes as f64,
                        ),
                        ("coordinode_vector_index_free_slots", s.free_slots as f64),
                    ];
                    for (name, value) in gauges {
                        metrics::gauge!(
                            name,
                            "label" => label.clone(),
                            "property" => property.clone(),
                        )
                        .set(value);
                    }
                    // Monotonic since the index was created.
                    let counters: [(&'static str, u64); 3] = [
                        ("coordinode_vector_index_lost_cas_total", s.lost_cas),
                        (
                            "coordinode_vector_index_admission_waits_total",
                            s.admission_waits,
                        ),
                        (
                            "coordinode_vector_index_admission_wait_microseconds_total",
                            s.admission_wait.as_micros() as u64,
                        ),
                    ];
                    for (name, value) in counters {
                        metrics::counter!(
                            name,
                            "label" => label.clone(),
                            "property" => property.clone(),
                        )
                        .absolute(value);
                    }
                }
            }
        })
    };

    let raft_node_shared: Option<Arc<coordinode_raft::cluster::RaftNode>> =
        Some(Arc::clone(&raft_node));

    // Hand the RaftNode to the scrub task so WAL-replay repair can read
    // the oplog. Set-once; the scrub's first (jitter-delayed) run is well
    // after this point.
    let _ = raft_slot.set(Arc::clone(&raft_node));

    // Bring index definitions live as their entries apply: a replica's copy of
    // a leader's CREATE / DROP INDEX, and on a single node the definitions the
    // log replays after the database opened. Woken only by applied Schema
    // entries that write an index definition (or replace the partition), so
    // an ordinary write costs nothing here. The field dictionary needs no such
    // follower: each statement refreshes its view whenever a binding applied.
    let index_definitions = IndexDefinitionFollower::spawn(&engine, Arc::downgrade(&database));

    // Drive AFTER COMMIT trigger dispatch on the Raft leader. The event
    // queue (`trigger_pending:`) is Raft-replicated, so every node sees the
    // same backlog; gating execution on the lease holder means only one
    // node runs the backlog at a time (the body's writes have to go through
    // the leader's pipeline anyway). A leader change before an event is
    // acknowledged runs it again on the new leader, so bodies run at least
    // once. Woken by each applied entry (covers fresh enqueues, a trigger
    // enabled, and the entry a new leader appends) and by the earliest
    // retry the last pass left scheduled; with nothing queued it sleeps.
    // Passes are at least `trigger_dispatch_interval` apart. The blocking
    // dispatch runs off the async runtime so a long body never stalls
    // consensus.
    if peers.is_some() {
        let db = Arc::clone(&database);
        let rn = Arc::clone(&raft_node);
        let mut applied_rx = rn.subscribe_applied();
        tokio::spawn(async move {
            let mut due: Option<tokio::time::Instant> = None;
            loop {
                tokio::select! {
                    changed = applied_rx.changed() => {
                        if changed.is_err() {
                            break; // RaftNode dropped — shut the worker down.
                        }
                    }
                    _ = tokio::time::sleep_until(due.unwrap_or_else(tokio::time::Instant::now)), if due.is_some() => {}
                }
                due = None;
                if rn.current_leader() != Some(rn.node_id()) {
                    continue;
                }
                let passed_at = tokio::time::Instant::now();
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
                        // A bookkeeping failure leaves its event due: try again.
                        let retry_in = if report.errors.is_empty() {
                            report.next_due_us.map(|at| {
                                // A retry already due waits for nothing.
                                std::time::Duration::from_micros(at.saturating_sub(unix_now_us()))
                            })
                        } else {
                            Some(std::time::Duration::ZERO)
                        };
                        due = retry_in.map(|wait| passed_at + wait.max(trigger_dispatch_interval));
                    }
                    Err(e) => {
                        tracing::warn!(%e, "after-commit dispatch task join error");
                        due = Some(passed_at + trigger_dispatch_interval);
                    }
                }
            }
        });
    }

    // Rebuild the B-tree indexes a store kept in the entry layout that
    // preceded transactional entries, and finish the builds an earlier
    // process left unfinished. Both are written through the log, so the
    // leader runs them; a member that is not leading looks again at each
    // applied entry (a new leader's first entry among them) until it leads or
    // another member's work reaches it through apply.
    {
        let db = Arc::clone(&database);
        let rn = Arc::clone(&raft_node);
        let mut applied_rx = rn.subscribe_applied();
        tokio::spawn(async move {
            // A failed rebuild is tried again after this, not at the next apply.
            const RETRY: std::time::Duration = std::time::Duration::from_secs(5);
            let mut first = true;
            loop {
                if !std::mem::take(&mut first) && applied_rx.changed().await.is_err() {
                    break; // RaftNode dropped.
                }
                if rn.current_leader() != Some(rn.node_id()) {
                    continue;
                }
                let db2 = Arc::clone(&db);
                match tokio::task::spawn_blocking(move || {
                    let db = db2.read();
                    let rebuilt = db.rebuild_legacy_btree_indexes()?;
                    let resumed = db.resume_interrupted_index_builds()?;
                    // After the rebuild: an index of an earlier release is
                    // ready, in the current layout, before it is adopted.
                    db.adopt_unowned_unique_indexes()
                        .map(|adopted| (rebuilt, resumed, adopted))
                })
                .await
                {
                    Ok(Ok((rebuilt, resumed, adopted))) => {
                        if rebuilt > 0 {
                            tracing::info!(rebuilt, "B-tree indexes rebuilt in the current layout");
                        }
                        if resumed > 0 {
                            tracing::info!(resumed, "interrupted B-tree index builds finished");
                        }
                        if adopted > 0 {
                            tracing::info!(
                                adopted,
                                "unique indexes became the constraints owning them"
                            );
                        }
                        break;
                    }
                    Ok(Err(e)) => tracing::warn!(%e, "B-tree index rebuild failed; retrying"),
                    Err(e) => tracing::warn!(%e, "B-tree index rebuild task join error"),
                }
                tokio::time::sleep(RETRY).await;
                first = true;
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
        )
        .with_statement_defaults(statement_defaults);
        if let Some(ref rn) = raft_node_shared {
            svc.with_raft_node(Arc::clone(rn))
        } else {
            svc
        }
    };
    let vector_service = services::vector::VectorServiceImpl::new(Arc::clone(&database));
    let text_service = services::text::TextServiceImpl::new(Arc::clone(&database));
    let health_service = services::health::HealthServiceImpl;
    // CDC service: tails the Raft log up to the entries this node applied.
    // Empty stream in embedded mode: there is no Raft log. Shared with the
    // session service, whose subscriptions read through it.
    let cdc_service = Arc::new(
        match raft_node_shared {
            Some(ref rn) => services::cdc::ChangeEventServiceImpl::for_raft_node(
                &database.read().engine_shared(),
                Arc::clone(rn),
                consumer_registry,
            )?,
            None => services::cdc::ChangeEventServiceImpl::new(
                0,
                Vec::new(),
                consumer_registry,
                Arc::new(|| 0),
                None,
            ),
        }
        .with_tuning(cdc_tuning),
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
    let readiness = ops::Readiness::default();
    // Consensus that stopped on a fatal error commits nothing more: the node
    // stops reporting ready, so a balancer and an operator see it.
    if let Some(node) = raft_node_shared.clone() {
        let consensus_readiness = readiness.clone();
        tokio::spawn(async move {
            if let Some(fatal) = node.consensus_stopped().await {
                tracing::error!(%fatal, "consensus stopped; the node no longer reports ready");
                consensus_readiness.consensus_failed();
            }
        });
    }
    let ops_readiness = readiness.clone();
    let version_view: ops::VersionView = {
        let node = Arc::clone(&raft_node);
        Arc::new(move || {
            serde_json::to_string(&node.version_report())
                .unwrap_or_else(|e| format!(r#"{{"error":"version report: {e}"}}"#))
        })
    };
    tokio::spawn(async move {
        if let Err(e) =
            ops::start_ops_server(ops_listener, ops_readiness, sample_gauges, version_view).await
        {
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
        static DESCRIPTOR_BYTES: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/coordinode.descriptor.bin"));
        // The proxy would otherwise mount its own /health and /metrics on the
        // REST port, reporting proxy state. CoordiNode publishes those for the
        // database itself on the ops port, which is where the documented
        // endpoints live, so keep the proxy off both paths. Both structs are
        // #[non_exhaustive], so start from the default and clear the flag.
        let mut proxy_health = HealthConfig::default();
        proxy_health.enabled = false;
        let mut proxy_metrics = MetricsConfig::default();
        proxy_metrics.enabled = false;
        // The wiring structs are hand-buildable for embedders; everything not
        // named here keeps the proxy's own default, including the forwarded
        // headers (authorization, dpop, request id, forwarding and client
        // headers, idempotency key). The proxy exposes an axum Router that is
        // bound and served here.
        let config = ProxyConfig {
            upstream: Some(UpstreamConfig {
                default: grpc_upstream,
            }),
            descriptors: vec![DescriptorSource::Embedded {
                bytes: DESCRIPTOR_BYTES,
            }],
            listen: ListenConfig {
                http: rest_addr.clone(),
                ..ListenConfig::default()
            },
            service: ServiceConfig {
                name: "coordinode".into(),
            },
            health: proxy_health,
            metrics: proxy_metrics,
            ..ProxyConfig::default()
        };
        let proxy = structured_proxy::ProxyServer::from_config(config);
        tokio::spawn(async move {
            match proxy.router() {
                Ok(router) => {
                    if let Err(e) = axum::serve(rest_listener, router).await {
                        tracing::error!("REST proxy serve error: {e}");
                    }
                }
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

    let shutdown_readiness = readiness.clone();
    let shutdown = async move {
        #[cfg(unix)]
        tokio::select! {
            _ = sigterm.recv() => {
                info!("SIGTERM received, initiating graceful shutdown");
            }
            _ = tokio::signal::ctrl_c() => {
                info!("Ctrl+C received, initiating graceful shutdown");
            }
        }
        // A Windows console process is asked to stop by a console control
        // event, not a signal: Ctrl+C at a terminal, CTRL_BREAK from a
        // supervisor that started it in its own process group, CTRL_CLOSE
        // when its console goes away (`docker stop` on a Windows container)
        // and CTRL_SHUTDOWN when the system shuts down. Each one gets the
        // same graceful shutdown as SIGTERM.
        #[cfg(windows)]
        {
            use tokio::signal::windows;
            match (
                windows::ctrl_c(),
                windows::ctrl_break(),
                windows::ctrl_close(),
                windows::ctrl_shutdown(),
            ) {
                (Ok(mut c), Ok(mut brk), Ok(mut close), Ok(mut shutdown)) => {
                    let event = tokio::select! {
                        _ = c.recv() => "CTRL_C",
                        _ = brk.recv() => "CTRL_BREAK",
                        _ = close.recv() => "CTRL_CLOSE",
                        _ = shutdown.recv() => "CTRL_SHUTDOWN",
                    };
                    info!(
                        event,
                        "console control event received, initiating graceful shutdown"
                    );
                }
                // Without the handlers the process would still end on these
                // events, only without draining; Ctrl+C alone is what is left.
                _ => {
                    tracing::warn!(
                        "could not install the console control handlers; only Ctrl+C \
                         shuts down gracefully"
                    );
                    if let Err(e) = tokio::signal::ctrl_c().await {
                        tracing::warn!(%e, "Ctrl+C handler failed");
                    }
                }
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            if let Err(e) = tokio::signal::ctrl_c().await {
                tracing::warn!(%e, "Ctrl+C handler failed");
            }
            info!("Ctrl+C received, initiating graceful shutdown");
        }
        // First, so /ready turns a balancer away while in-flight RPCs drain.
        shutdown_readiness.set(false);
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
    let max_req_bytes = sizes.max_request_bytes;

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
            proto::v2::graph::schema_service_server::SchemaServiceServer::new(schema_service)
                .max_decoding_message_size(max_req_bytes),
        )
        .add_service(
            proto::query::cypher_service_server::CypherServiceServer::new(cypher_service)
                .max_decoding_message_size(max_req_bytes),
        )
        .add_service(
            proto::session::session_service_server::SessionServiceServer::new({
                let svc = services::session::SessionSvc::new(
                    Arc::clone(&database),
                    Arc::clone(&session_registry),
                    raft_node_shared
                        .as_ref()
                        .map(|raft| Arc::clone(raft.version())),
                )
                .with_change_streams(Arc::clone(&cdc_service));
                // In a cluster, a session reports what its node can
                // actually do: whether a leader is reachable, and so
                // whether writes go through. Standalone keeps the
                // always-writable default, which there is the truth.
                match raft_node_shared.as_ref() {
                    Some(raft) => svc.with_cluster(Arc::clone(raft)),
                    None => svc,
                }
            })
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
            proto::replication::cdc::change_stream_service_server::ChangeStreamServiceServer::from_arc(
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
        // The frozen exchange a member of another version is answered with.
        routes.add_service(handler.handshake_service());
        routes.add_service(RaftServiceServer::new(handler));
        info!(node_id, "RaftService registered on :7080 (shared port)");

        // SegmentTransferService: receive bulk segment pushes (replication
        // repair, operator-commanded migration, node resync) and install
        // them into local storage via the engine-backed sink.
        use coordinode_replicate::segment_store::SegmentInstaller;
        use coordinode_replicate::transfer::SegmentTransferHandler;
        use coordinode_replicate::transfer::proto::segment_transfer_service_server::SegmentTransferServiceServer;
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

    // The listener is bound and every service registered: connections that
    // arrive now are accepted as soon as the server below starts polling.
    readiness.set(true);
    router
        .serve_with_incoming_shutdown(grpc_incoming, shutdown)
        .await?;
    drop(index_definitions);

    Ok(())
}

/// Brings this member's B-tree, vector and text indexes in line with the
/// index definitions as their entries apply.
struct IndexDefinitionFollower {
    stop: coordinode_storage::engine::applied::AppliedStop,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// Queue of applied Schema events the follower may fall behind on; a full
/// queue arrives as "replaced" and costs one refresh, never a missed one.
const INDEX_DEFINITION_QUEUE: usize = 1024;

impl IndexDefinitionFollower {
    fn spawn(
        engine: &Arc<coordinode_storage::engine::core::StorageEngine>,
        database: std::sync::Weak<parking_lot::RwLock<coordinode_embed::Database>>,
    ) -> Self {
        use coordinode_storage::engine::applied::AppliedEvent;
        let applied = engine.subscribe_applied(
            coordinode_storage::engine::partition::Partition::Schema,
            INDEX_DEFINITION_QUEUE,
        );
        let stop = applied.stopper();
        let thread = std::thread::Builder::new()
            .name("index-def-follower".to_string())
            .spawn(move || {
                // Asleep until a Schema entry applies; `None` once stopped.
                while let Some(first) = applied.next(None) {
                    let mut event = Some(first);
                    let mut definitions_changed = false;
                    while let Some(current) = event {
                        definitions_changed |= match current {
                            AppliedEvent::Keys { keys, .. } => {
                                keys.iter().any(|k| k.starts_with(b"schema:idx:"))
                            }
                            // A command or a replaced partition: which keys
                            // changed is not known.
                            AppliedEvent::Replaced => true,
                        };
                        event = applied.try_next();
                    }
                    if !definitions_changed {
                        continue;
                    }
                    let Some(db) = database.upgrade() else {
                        break;
                    };
                    refresh_index_definitions(&db.read());
                }
            })
            .map_err(|e| tracing::error!(%e, "could not start the index definition follower"))
            .ok();
        Self { stop, thread }
    }
}

/// Stopped on every way out of `serve`, an early error included.
impl Drop for IndexDefinitionFollower {
    fn drop(&mut self) {
        self.stop.stop();
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                tracing::error!("the index definition follower panicked");
            }
        }
    }
}

/// Register, rebuild or drop this member's indexes to match the stored
/// definitions.
fn refresh_index_definitions(db: &coordinode_embed::Database) {
    // Register + local HNSW rebuild (the graph itself is never replicated).
    match db.refresh_vector_indexes() {
        Ok(0) => {}
        Ok(n) => tracing::info!(n, "vector indexes brought live from apply"),
        Err(e) => tracing::warn!(%e, "vector index refresh failed"),
    }
    // Text index definitions another member created or dropped: registered
    // and rebuilt from the store here; the text worker keeps them current
    // from then on.
    match db.refresh_text_indexes() {
        Ok(0) => {}
        Ok(n) => tracing::info!(n, "text indexes brought in line from apply"),
        Err(e) => tracing::warn!(%e, "text index refresh failed"),
    }
    // B-tree definitions another member created or dropped: their entries
    // arrive in the log, the definitions tell this member to maintain and use
    // them.
    if let Err(e) = db.refresh_btree_indexes() {
        tracing::warn!(%e, "B-tree index refresh failed");
    }
}

/// Wall-clock microseconds since the Unix epoch, the clock the AFTER COMMIT
/// queue schedules retries on.
fn unix_now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_micros() as u64)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
