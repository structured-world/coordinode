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
    let peer_timeouts = cfg.peer_timeouts();
    // Refused at start: a timing that keeps a healthy group electing is not
    // one to run with.
    let election_timing = cfg.election_timing()?;
    let election_preference = cfg.election_preference();
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
        index_feed_capacity: _,
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
        autocommit_retry_attempts,
        autocommit_retry_backoff_ms,
        peers: peers_vec,
        membership_change_timeout_secs,
        join_readiness_lag_entries,
        join_timeout_secs,
        raft_snapshot_entries,
        raft_snapshot_log_bytes,
        raft_snapshot_min_interval_secs,
        // Already captured above (election_timing / _preference) before the move.
        election_profile: _,
        election_heartbeat_ms: _,
        election_timeout_min_ms: _,
        election_timeout_max_ms: _,
        election_preference: _,
        planner_stats_ttl_secs,
        slow_query_ms,
        vector_build_wait_ms,
        query_memory_limit_mb,
        vector_retired_bytes_budget,
        index_build_max_running,
        index_build_older_transactions_wait_secs,
        index_build_unique_read_limit,
        index_build_statement_wait_secs,
        index_check_interval_secs,
        index_check_page,
        index_check_max_repairs,
        mode: _,
        // Already consumed above via set_wire_zstd_level before serving.
        wire_compression_level: _,
        tls_cert,
        tls_key,
        tls_ca,
        tls_require_client_auth,
        node_identity: node_identity_settings,
        // Already captured above (peer_timeouts) before the move.
        peer_connect_timeout_ms: _,
        peer_call_timeout_ms: _,
        peer_handshake_timeout_ms: _,
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
    if tls_require_client_auth && tls_ca.is_none() {
        return Err("tls_require_client_auth requires tls_ca".into());
    }
    // A node with peers authenticates every inter-node call under its node
    // identity; there is no plaintext peer channel to fall back to. Its port
    // presents that certificate to clients too, so a second server
    // certificate would be ambiguous.
    let node_identity = match (&peers, node_identity_settings) {
        (Some(_), None) => {
            return Err(
                "a node with peers needs node_identity: the certificate, key and \
                        trust bundles it proves itself to its peers with"
                    .into(),
            );
        }
        (Some(_), Some(_)) if tls_cert.is_some() || tls_key.is_some() => {
            return Err(
                "tls_cert / tls_key are not used with peers: the gRPC port presents \
                        the node_identity certificate, which may carry the DNS names clients \
                        verify"
                    .into(),
            );
        }
        (Some(_), Some(settings)) => Some(Arc::new(
            crate::node_identity::NodeIdentity::load(settings)
                .map_err(|e| format!("node_identity: {e}"))?,
        )),
        (None, Some(_)) => {
            // Logging starts below; this line must not be lost.
            eprintln!("warning: node_identity is configured without peers and is not used");
            None
        }
        (None, None) => None,
    };

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
    let grpc_listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("cannot bind the gRPC address {addr}: {e}"))?;
    #[cfg(feature = "rest-proxy")]
    let grpc_upstream = local_upstream(grpc_listener.local_addr()?);
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
    // The member: the store, its consensus node and the database over the
    // replicated pipeline, assembled by the embedded library the same way an
    // embedding host gets it. Every mode writes through the log: a standalone
    // server is a group of one, which is what lets it grow later.
    let defaults = coordinode_raft::cluster::SnapshotTriggerConfig::default();
    let mut member_options = coordinode_embed::replication::MemberOptions::new(node_id);
    member_options.snapshots = coordinode_raft::cluster::SnapshotTriggerConfig {
        logs_since_last: raft_snapshot_entries.map_or(defaults.logs_since_last, |n| n.get()),
        log_bytes: raft_snapshot_log_bytes.map_or(defaults.log_bytes, |n| n.get()),
        min_interval: raft_snapshot_min_interval_secs.map_or(defaults.min_interval, |n| {
            std::time::Duration::from_secs(n.get())
        }),
    };
    member_options.timing = election_timing;
    member_options.preference = election_preference;
    // A server is not embedded in an application with a format of its own:
    // its host epoch is zero.
    member_options.host_epoch = 0;
    if let (Some(peers_list), Some(identity)) = (&peers, &node_identity) {
        info!(identity = %identity.context().identity(), "node identity");
        let founding = if node_id == 1 {
            info!(
                peers = peers_list.len(),
                node_id, "cluster mode: the member the group is formed around"
            );
            // A deployment named here is the one an existing deployment's
            // node forms its group in; none founds a new one.
            coordinode_embed::replication::Founding::Form {
                deployment: identity.deployment()?,
            }
        } else {
            info!(
                peers = peers_list.len(),
                node_id, "cluster mode: joining node"
            );
            coordinode_embed::replication::Founding::Join(identity.enrollment()?)
        };
        member_options.peering = Some(coordinode_embed::replication::Peering {
            security: Arc::clone(identity.context()),
            advertise: effective_advertise.clone(),
            founding,
            timeouts: peer_timeouts,
        });
    } else {
        info!(node_id, "standalone mode: a group of one");
    }
    let mut member = coordinode_embed::replication::Member::open(&storage_config, member_options)
        .await
        .map_err(|e| format!("failed to open the member: {e}"))?;
    let engine = Arc::clone(member.engine());
    let raft_node = Arc::clone(member.node());

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
            let scrub_raft = Arc::clone(&raft_node);
            let scrub_ckpt_dir = checkpoint_dir.clone();
            // Peers to pull a fresh copy from when scrub finds corruption
            // (CE basic replica-fetch repair): the members at repair time,
            // none when standalone. Each node repairs its own corruption
            // independently.
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
                                    let repair_peers =
                                        coordinode_embed::replication::segment_peers(&scrub_raft);
                                    let from_peers = match &repair_peers {
                                        None => None,
                                        Some((links, peers)) => Some(
                                            repair_installer
                                                .repair_partition(
                                                    links,
                                                    peers,
                                                    part,
                                                    1 << 20,
                                                    coordinode_replicate::PieceEncoding::None,
                                                )
                                                .await,
                                        ),
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
                                    let Some(ckpt) = checkpoint::latest_checkpoint(&scrub_ckpt_dir)
                                    else {
                                        tracing::warn!(
                                            partition = part.name(),
                                            "no checkpoint available for WAL-replay repair"
                                        );
                                        continue;
                                    };
                                    let raft = Arc::clone(&scrub_raft);
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

    // The deployment a node joining this one's is configured with.
    if let Some(deployment) = raft_node.deployment() {
        info!(%deployment, "deployment");
    }
    if let Some(secs) = membership_change_timeout_secs {
        raft_node.set_membership_settle_timeout(std::time::Duration::from_secs(secs));
    }
    if let Some(entries) = join_readiness_lag_entries {
        raft_node.set_join_readiness_lag(entries);
    }
    if let Some(secs) = join_timeout_secs {
        raft_node.set_join_timeout(std::time::Duration::from_secs(secs.get()));
    }

    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(raft_node.pipeline());

    // The database is configured before the member's background work
    // starts, below: an interrupted index build resumes under these settings.
    let database = Arc::clone(member.database());
    {
        let mut database = database.write();
        if let Some(secs) = planner_stats_ttl_secs {
            database.set_stats_ttl(std::time::Duration::from_secs(secs));
        }
        if let Some(ms) = vector_build_wait_ms {
            database.set_vector_build_wait(std::time::Duration::from_millis(ms));
        }
        if let Some(mib) = query_memory_limit_mb {
            // Refused at start rather than clamped: an operator who wrote a
            // limit past the ceiling meant something the server cannot honour.
            let bytes = mib
                .checked_mul(1 << 20)
                .ok_or_else(|| format!("query_memory_limit_mb = {mib} does not fit in bytes"))?;
            database
                .set_query_memory_limit(bytes)
                .map_err(|e| format!("query_memory_limit_mb = {mib}: {e}"))?;
        }
        if let Some(bytes) = vector_retired_bytes_budget {
            // A budget past the address space is no bound at all, which is what
            // `usize::MAX` means.
            database.set_vector_retired_bytes_budget(usize::try_from(bytes).unwrap_or(usize::MAX));
        }
        // Before any build resumes: this process takes up interrupted builds only
        // once it leads, below.
        let defaults = database.index_build_config();
        database.set_index_build_config(coordinode_query::index::IndexBuildConfig {
            max_running: index_build_max_running.map_or(defaults.max_running, |n| n.get()),
            older_transactions_wait: index_build_older_transactions_wait_secs.map_or(
                defaults.older_transactions_wait,
                std::time::Duration::from_secs,
            ),
            unique_admission_read_limit: index_build_unique_read_limit
                .unwrap_or(defaults.unique_admission_read_limit),
            statement_wait: index_build_statement_wait_secs
                .map_or(defaults.statement_wait, std::time::Duration::from_secs),
            check_page: index_check_page.map_or(defaults.check_page, |n| n.get()),
            check_max_repairs: index_check_max_repairs.unwrap_or(defaults.check_max_repairs),
            // Zero turns the periodic check off; reports and requests still
            // check.
            check_interval: match index_check_interval_secs {
                None => defaults.check_interval,
                Some(0) => None,
                Some(secs) => Some(std::time::Duration::from_secs(secs)),
            },
        });
        // What a statement executes under when neither it nor its session names a
        // concern.
        database.set_read_concern(statement_defaults.read_concern);
        database.set_write_concern(statement_defaults.write_concern);
    }

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
        db.set_autocommit_retry(
            autocommit_retry_attempts,
            std::time::Duration::from_millis(autocommit_retry_backoff_ms),
        );
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
        let registry_metrics = consumer_registry.clone();
        Arc::new(move || {
            // How far each change-stream consumer is behind, sampled per
            // scrape.
            registry::sample_consumer_gauges(&registry_metrics);
            // Where this member stands in consensus, sampled per scrape.
            {
                let raft = &version_node;
                metrics::gauge!("coordinode_raft_term").set(raft.current_term() as f64);
                metrics::gauge!("coordinode_raft_commit_index").set(raft.commit_index() as f64);
                metrics::gauge!("coordinode_raft_applied_index").set(raft.applied_index() as f64);
                let leads = raft.current_leader() == Some(raft.node_id());
                metrics::gauge!("coordinode_raft_leader").set(if leads { 1.0 } else { 0.0 });
            }
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

    // The member's background work, now that the database is configured:
    // index definitions brought live as their entries apply, and on the
    // leader the index builds no executor finishes and AFTER COMMIT trigger
    // dispatch, passes at least `trigger_dispatch_interval` apart.
    member.start_background(trigger_dispatch_interval);

    let query_registry = Arc::new(coordinode_query::advisor::QueryRegistry::new());
    let nplus1_detector = Arc::new(coordinode_query::advisor::nplus1::NPlus1Detector::new());

    let graph_service = services::graph::GraphServiceImpl::new(Arc::clone(&database));
    let schema_service = services::schema::SchemaServiceImpl::new(Arc::clone(&database));
    // One path for every Cypher statement, whichever transport carries it:
    // the unary service and the session share the settings, fence,
    // forwarding and advisor.
    let statement_executor = {
        let executor = services::statement::StatementExecutor::new(Arc::clone(&database))
            .with_advisor(Arc::clone(&query_registry), Arc::clone(&nplus1_detector))
            .with_statement_defaults(statement_defaults)
            .with_slow_query_threshold(Some(slow_query_ms.map_or(
                services::statement::DEFAULT_SLOW_QUERY,
                std::time::Duration::from_millis,
            )));
        match raft_node_shared {
            Some(ref rn) => executor.with_raft_node(Arc::clone(rn)),
            None => executor,
        }
    };
    let cypher_service =
        services::cypher::CypherServiceImpl::from_executor(statement_executor.clone());
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
    // Consensus that stopped on a fatal error commits nothing more, and the
    // process does not try to carry on: after a failed log sync the kernel
    // may already have dropped the unsynced pages, so nothing written since
    // the last good sync can be trusted. The node stops reporting ready and
    // shuts down with an error; its supervisor restarts it and recovery
    // replays the durable state, where every acknowledged write is.
    let (fatal_tx, fatal_rx) = tokio::sync::watch::channel(None::<String>);
    if let Some(node) = raft_node_shared.clone() {
        let consensus_readiness = readiness.clone();
        tokio::spawn(async move {
            if let Some(fatal) = node.consensus_stopped().await {
                tracing::error!(
                    %fatal,
                    "consensus stopped on a fatal error; the process exits so it restarts \
                     from its durable state"
                );
                consensus_readiness.consensus_failed();
                fatal_tx.send_replace(Some(fatal));
            }
        });
    }
    let ops_readiness = readiness.clone();
    // Reading the node's files again, on `POST /reload` and on SIGHUP: a
    // renewed certificate or changed trust bundles take effect without a
    // restart, and a set that does not validate leaves the last in force.
    let reload: Option<ops::ReloadHook> = node_identity.clone().map(|identity| {
        let hook: ops::ReloadHook = Arc::new(move || {
            let outcome = identity.reload();
            match &outcome {
                Ok(()) => info!("node credentials and trust reloaded"),
                Err(e) => tracing::error!(%e, "node credentials and trust not reloaded; the last valid ones stay"),
            }
            outcome
        });
        hook
    });
    #[cfg(unix)]
    if let Some(reload) = reload.clone() {
        let mut hangup = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())
            .map_err(|e| format!("failed to install SIGHUP handler: {e}"))?;
        tokio::spawn(async move {
            while hangup.recv().await.is_some() {
                let reload = Arc::clone(&reload);
                // The outcome is logged by the hook.
                let _ = tokio::task::spawn_blocking(move || reload()).await;
            }
        });
    }
    let version_view: ops::VersionView = {
        let node = Arc::clone(&raft_node);
        Arc::new(move || {
            serde_json::to_string(&node.version_report())
                .unwrap_or_else(|e| format!(r#"{{"error":"version report: {e}"}}"#))
        })
    };
    tokio::spawn(async move {
        if let Err(e) = ops::start_ops_server(
            ops_listener,
            ops_readiness,
            sample_gauges,
            version_view,
            reload,
        )
        .await
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
        use structured_proxy::config::{DescriptorSource, ProxyConfig, UpstreamConfig};
        static DESCRIPTOR_BYTES: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/coordinode.descriptor.bin"));
        // The config types are #[non_exhaustive]: start from the proxy's
        // defaults and set what CoordiNode wires. Everything not named here
        // keeps the default, including the forwarded headers (authorization,
        // dpop, request id, forwarding and client headers, idempotency key).
        let mut config = ProxyConfig::default();
        config.upstream = Some(UpstreamConfig::new(grpc_upstream));
        config.descriptors = vec![DescriptorSource::Embedded {
            bytes: DESCRIPTOR_BYTES,
        }];
        config.listen.http = rest_addr.clone();
        config.service.name = "coordinode".into();
        // The proxy would otherwise mount its own /health and /metrics on the
        // REST port, reporting proxy state. CoordiNode publishes those for the
        // database itself on the ops port, where the documented endpoints live.
        config.health.enabled = false;
        config.metrics.enabled = false;
        config
            .validate()
            .map_err(|e| format!("REST proxy config: {e}"))?;
        // The proxy exposes an axum Router that is bound and served here.
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
    let signals = async move {
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
    };
    let mut fatal_watch = fatal_rx.clone();
    let shutdown = async move {
        tokio::select! {
            () = signals => {}
            () = fatal_reported(&mut fatal_watch) => {
                info!("shutting down after a fatal consensus error");
            }
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

    // Client TLS. A standalone node serves TLS when a cert+key are
    // configured; the crypto provider was installed above so tonic's rustls
    // config uses it. With require-client-auth, client certificates are
    // verified against the CA (mutual TLS).
    //
    // A node with peers serves one port to clients and nodes alike: it makes
    // the TLS connection itself, presenting the node certificate, and tells a
    // node's connection from a client's by the identity the certificate
    // proves. Clients may connect in plain text unless they must present a
    // certificate; nodes never.
    let shared_port = match node_identity.as_ref() {
        Some(identity) => {
            let clients = match tls_ca.as_ref() {
                Some(ca_path) => {
                    let pem = std::fs::read(ca_path)
                        .map_err(|e| format!("read tls ca {ca_path}: {e}"))?;
                    let roots = coordinode_wire::tls::load_certs(&pem)
                        .map_err(|e| format!("tls ca {ca_path}: {e}"))?;
                    Some(
                        coordinode_wire::tls::client_verifier(roots, tls_require_client_auth)
                            .map_err(|e| format!("tls ca {ca_path}: {e}"))?,
                    )
                }
                None => None,
            };
            info!(
                client_certificates = clients.is_some(),
                mtls = tls_require_client_auth,
                "gRPC port shared by clients and nodes"
            );
            Some((
                Arc::clone(identity.context()),
                coordinode_wire::peer::Accept {
                    inbound: coordinode_wire::peer::InboundAuth::Shared { clients },
                    plaintext_clients: !tls_require_client_auth,
                    handshake_timeout: peer_timeouts.handshake,
                },
            ))
        }
        None => {
            if let (Some(cert_path), Some(key_path)) = (tls_cert.as_ref(), tls_key.as_ref()) {
                use tonic::transport::{Certificate, Identity, ServerTlsConfig};
                let cert = std::fs::read(cert_path)
                    .map_err(|e| format!("read tls cert {cert_path}: {e}"))?;
                let key =
                    std::fs::read(key_path).map_err(|e| format!("read tls key {key_path}: {e}"))?;
                let mut tls = ServerTlsConfig::new().identity(Identity::from_pem(&cert, &key));
                if tls_require_client_auth {
                    if let Some(ca_path) = tls_ca.as_ref() {
                        let ca = std::fs::read(ca_path)
                            .map_err(|e| format!("read tls ca {ca_path}: {e}"))?;
                        tls = tls.client_ca_root(Certificate::from_pem(ca));
                    }
                }
                server = server
                    .tls_config(tls)
                    .map_err(|e| format!("tls config: {e}"))?;
                info!(mtls = tls_require_client_auth, "gRPC TLS enabled");
            }
            None
        }
    };

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
                    statement_executor.clone(),
                    Arc::clone(&session_registry),
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

    // The services peers call (version handshake, consensus, segment
    // transfer) share the main gRPC port with the client services; each one
    // refuses a caller that is not an admitted node before it reads the
    // message. A standalone member serves none.
    if member.add_inter_node_services(&mut routes, max_req_bytes) {
        info!(node_id, "inter-node services registered on the shared port");
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
    match shared_port {
        Some((context, accept)) => {
            let incoming = coordinode_wire::peer::incoming(grpc_listener, &context, accept)
                .map_err(|e| format!("gRPC port TLS configuration: {e}"))?;
            router
                .serve_with_incoming_shutdown(incoming, shutdown)
                .await?;
        }
        None => {
            let incoming =
                tonic::transport::server::TcpIncoming::from(grpc_listener).with_nodelay(Some(true));
            router
                .serve_with_incoming_shutdown(incoming, shutdown)
                .await?;
        }
    }
    // The member's background work stops with it.
    drop(member);

    if let Some(fatal) = fatal_rx.borrow().clone() {
        return Err(format!("consensus stopped on a fatal error: {fatal}").into());
    }
    Ok(())
}

/// Resolves once the consensus watcher reports a fatal error; never when the
/// watcher ends without one, which is a normal stop.
async fn fatal_reported(fatal: &mut tokio::sync::watch::Receiver<Option<String>>) {
    if fatal.wait_for(Option::is_some).await.is_err() {
        std::future::pending::<()>().await;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
