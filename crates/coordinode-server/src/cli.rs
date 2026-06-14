//! CLI argument parsing for coordinode subcommands.
//!
//! Subcommands:
//! - `serve` (default) — start gRPC + ops servers
//! - `version` — print version
//! - `verify --deep` — verify storage integrity
//! - `backup` — export database to file
//! - `restore` — import database from file

use coordinode_embed::backup::BackupFormat;

use crate::config::ServeMode;

/// Parsed CLI command.
pub enum Command {
    /// Start the database server (default).
    Serve {
        /// Operational mode (CE supports only "full").
        /// --mode=compute and --mode=storage require coordinode-ee.
        mode: ServeMode,
        /// Numeric node ID for this instance (default: 1).
        /// Must be unique within the cluster. In single-node deployments
        /// the default of 1 is always correct.
        node_id: u64,
        /// gRPC listen address (default: [::]:7080).
        grpc_addr: String,
        /// Advertise address for intra-cluster gRPC (default: same as --addr).
        /// Other nodes use this address to send Raft RPCs to this node.
        /// Set this when the listen address is 0.0.0.0 or [::] so peers
        /// know the actual hostname/IP.
        advertise_addr: Option<String>,
        /// REST/JSON proxy listen address (default: [::]:7081).
        /// Transcodes HTTP/JSON requests to gRPC via embedded structured-proxy.
        /// Only present when compiled with the `rest-proxy` feature.
        #[cfg(feature = "rest-proxy")]
        rest_addr: String,
        /// Operational HTTP server address for /metrics and /health (default: [::]:7084).
        /// Pass port 0 to let the OS assign an ephemeral port (useful in tests).
        ops_addr: String,
        /// Data directory (default: ./data).
        data_dir: String,
        /// Peer addresses for cluster mode (comma-separated).
        /// When provided, enables Raft consensus with the given peers.
        /// Example: --peers "node2:7080,node3:7080"
        peers: Option<Vec<String>>,
        /// Open-file-descriptor soft limit to request at startup
        /// (`setrlimit(RLIMIT_NOFILE)`). `None` raises the soft limit to the
        /// hard limit. The storage engine keeps many files open, so a high
        /// limit matters in production. Unix only; ignored elsewhere.
        nofile: Option<u64>,
        /// Maximum in-flight requests per client connection (gRPC concurrency
        /// limit). `None` leaves it unbounded. Mirrors a connection cap on a
        /// stream-multiplexed transport.
        max_connections: Option<usize>,
        /// Maximum decoded request message size, in MiB (default: 16, matching
        /// the common document-size limit). Guards against unbounded-allocation
        /// requests.
        max_request_size_mb: usize,
        /// Per-request timeout in seconds. `None` disables the server-side
        /// timeout.
        request_timeout_secs: Option<u64>,
        /// HTTP/2 keepalive ping interval in seconds. `None` disables keepalive
        /// pings. Useful to detect half-open connections across a load balancer.
        http2_keepalive_secs: Option<u64>,
        /// Block cache size in MiB. `None` keeps the engine default. The read
        /// path serves hot blocks from this cache before touching disk.
        cache_size_mb: Option<u64>,
        /// Write buffer (memtable) size in MiB. `None` keeps the engine
        /// default. Larger buffers reduce flush frequency at the cost of memory.
        write_buffer_mb: Option<u64>,
    },
    /// Print version and exit.
    Version,
    /// Verify storage integrity.
    Verify {
        /// Data directory.
        data_dir: String,
        /// Deep verification (checksums on all pages).
        deep: bool,
    },
    /// Export database to a backup file.
    ///
    /// Takes a consistent MVCC snapshot at the start — ongoing writes
    /// are not blocked during backup.
    Backup {
        /// Data directory (source database).
        data_dir: String,
        /// Output file path.
        output: String,
        /// Backup format: json, cypher, or binary.
        format: BackupFormat,
        /// Optional namespace filter (export only this namespace).
        namespace: Option<String>,
        /// Incremental snapshot boundary: with `--format snapshot`, export
        /// only changes after this seqno (from a prior backup's report).
        since: Option<u64>,
    },
    /// Restore database from a backup file.
    Restore {
        /// Data directory (target database — will be created if empty).
        data_dir: String,
        /// Input file path.
        input: String,
        /// Backup format: json, cypher, or binary.
        format: BackupFormat,
        /// Optional target namespace (restore into this namespace).
        namespace: Option<String>,
        /// Selective restore: keep only nodes carrying one of these labels
        /// (and edges between kept nodes). Applies to json / apoc-json /
        /// hetio-json. Empty = restore everything.
        only_labels: Vec<String>,
        /// Bypass binary-dump compatibility checks (format version and
        /// schema fingerprint). Use to restore a newer dump or into a
        /// database with a differing schema at your own risk.
        force: bool,
    },
    /// Create a hard-linked physical checkpoint of the whole database.
    ///
    /// Zero-copy on a single filesystem: SST files are hard-linked, so the
    /// checkpoint consumes no extra disk until the originals compact away.
    /// The result is an independently-openable database — restore by
    /// pointing `serve --data` at the checkpoint (or a copy of it). Orders
    /// of magnitude faster than a logical `backup` dump for large data.
    Checkpoint {
        /// Data directory (source database).
        data_dir: String,
        /// Output directory for the checkpoint (must not exist).
        output: String,
    },
    /// Admin commands for a running cluster.
    AdminNodeJoin {
        /// gRPC address of any cluster member (usually the leader).
        /// Example: "http://node1:7080"
        cluster_addr: String,
        /// Numeric node ID to assign to the new node. Must be unique in the cluster.
        node_id: u64,
        /// gRPC address of the new node (host:port). The node must be running.
        node_addr: String,
        /// If set, the new node has pre-seeded data from an offline snapshot backup.
        /// Three-tier recovery skips Tier 3 (full resync) and starts from Tier 1/2.
        pre_seeded: bool,
        /// If set, stream join progress until COMPLETE or FAILED.
        follow: bool,
    },
    /// Gracefully decommission a node from the cluster.
    AdminNodeDecommission {
        /// gRPC address of any cluster member (usually the leader).
        /// Example: "http://node1:7080"
        cluster_addr: String,
        /// Numeric node ID to decommission. Must be in the current voter set.
        node_id: u64,
        /// If set, mark node data for deletion after removal.
        /// In CE, operator must delete data on the decommissioned node manually.
        pruning: bool,
        /// Emergency decommission: skip quorum gate and drain checks.
        /// Forces membership remove even if the node is unreachable.
        /// May cause permanent data loss — requires --skip-confirmation.
        force: bool,
        /// Required when --force is set. Confirms awareness of potential data loss.
        skip_confirmation: bool,
    },
}

/// Parse command line arguments.
pub fn parse_args() -> Command {
    let args: Vec<String> = std::env::args().collect();
    parse_args_from(&args)
}

/// Parse from an explicit args slice (testable without std::env::args).
pub fn parse_args_from(args: &[String]) -> Command {
    if args.len() < 2 {
        return default_serve();
    }

    match args[1].as_str() {
        "serve" => {
            let mode = match find_flag(args, "--mode") {
                None => ServeMode::Full,
                Some(s) => match ServeMode::parse(&s) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("error: {e}");
                        std::process::exit(1);
                    }
                },
            };
            let node_id: u64 = match find_flag(args, "--node-id") {
                None => 1,
                Some(s) => match s.parse() {
                    Ok(id) if id > 0 => id,
                    _ => {
                        eprintln!("error: --node-id must be a positive integer, got '{s}'");
                        std::process::exit(1);
                    }
                },
            };
            let grpc_addr = find_flag(args, "--addr").unwrap_or_else(|| "[::]:7080".to_string());
            let advertise_addr = find_flag(args, "--advertise-addr");
            #[cfg(feature = "rest-proxy")]
            let rest_addr =
                find_flag(args, "--rest-addr").unwrap_or_else(|| "[::]:7081".to_string());
            let ops_addr = find_flag(args, "--ops-addr").unwrap_or_else(|| "[::]:7084".to_string());
            let data_dir = find_flag(args, "--data").unwrap_or_else(|| "./data".to_string());
            let peers = find_flag(args, "--peers").map(|p| {
                p.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            });
            // Validate: --node-id > 1 without --peers makes no sense.
            if node_id > 1 && peers.is_none() {
                eprintln!(
                    "error: --node-id={node_id} requires --peers. \
                     Single-node deployments always use node-id=1."
                );
                std::process::exit(1);
            }
            let nofile = find_flag_num(args, "--nofile");
            let max_connections = find_flag_num(args, "--max-connections");
            let max_request_size_mb = find_flag_num(args, "--max-request-size-mb").unwrap_or(16);
            let request_timeout_secs = find_flag_num(args, "--request-timeout-secs");
            let http2_keepalive_secs = find_flag_num(args, "--http2-keepalive-secs");
            let cache_size_mb = find_flag_num(args, "--cache-size-mb");
            let write_buffer_mb = find_flag_num(args, "--write-buffer-mb");
            Command::Serve {
                mode,
                node_id,
                grpc_addr,
                advertise_addr,
                #[cfg(feature = "rest-proxy")]
                rest_addr,
                ops_addr,
                data_dir,
                peers,
                nofile,
                max_connections,
                max_request_size_mb,
                request_timeout_secs,
                http2_keepalive_secs,
                cache_size_mb,
                write_buffer_mb,
            }
        }
        "version" | "--version" | "-v" => Command::Version,
        "verify" => {
            let data_dir = find_flag(args, "--data").unwrap_or_else(|| "./data".to_string());
            let deep = args.iter().any(|a| a == "--deep");
            Command::Verify { data_dir, deep }
        }
        "backup" => {
            let data_dir = find_flag(args, "--data").unwrap_or_else(|| "./data".to_string());
            let output = match find_flag(args, "--output") {
                Some(o) => o,
                None => {
                    eprintln!("error: --output is required for backup");
                    std::process::exit(1);
                }
            };
            let format = parse_format(args);
            let namespace = find_flag(args, "--namespace");
            let since = find_flag(args, "--since").map(|s| {
                s.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --since must be a non-negative integer seqno");
                    std::process::exit(1);
                })
            });
            Command::Backup {
                data_dir,
                output,
                format,
                namespace,
                since,
            }
        }
        "restore" => {
            let data_dir = find_flag(args, "--data").unwrap_or_else(|| "./data".to_string());
            let input = match find_flag(args, "--input") {
                Some(i) => i,
                None => {
                    eprintln!("error: --input is required for restore");
                    std::process::exit(1);
                }
            };
            let format = parse_format(args);
            let namespace = find_flag(args, "--namespace");
            let only_labels = find_flag(args, "--only-labels")
                .map(|s| {
                    s.split(',')
                        .map(|l| l.trim().to_string())
                        .filter(|l| !l.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            let force = args.iter().any(|a| a == "--force");
            Command::Restore {
                data_dir,
                input,
                format,
                namespace,
                only_labels,
                force,
            }
        }
        "checkpoint" => {
            let data_dir = find_flag(args, "--data").unwrap_or_else(|| "./data".to_string());
            let output = match find_flag(args, "--output") {
                Some(o) => o,
                None => {
                    eprintln!("error: --output is required for checkpoint");
                    std::process::exit(1);
                }
            };
            Command::Checkpoint { data_dir, output }
        }
        "admin" => parse_admin_args(args),
        _ => {
            eprintln!(
                "coordinode v{}\n\n\
                 Usage:\n  \
                 coordinode serve [--mode full] [--node-id N] [--addr ADDR] [--advertise-addr ADDR]\n          \
                 [--rest-addr ADDR] [--ops-addr ADDR] [--data DIR] [--peers PEERS]\n          \
                 [--nofile N] [--max-connections N] [--max-request-size-mb N] [--request-timeout-secs N]\n          \
                 [--http2-keepalive-secs N] [--cache-size-mb N] [--write-buffer-mb N]\n  \
                 coordinode backup --output FILE [--data DIR] [--format json|cypher|binary|snapshot] [--namespace NS] [--since SEQNO]\n  \
                 coordinode restore --input FILE [--data DIR] [--format json|cypher|binary|snapshot|apoc-json|apoc-cypher|hetio-json] [--namespace NS] [--only-labels L1,L2] [--force]\n  \
                 coordinode checkpoint --output DIR [--data DIR]\n  \
                 coordinode verify [--data DIR] [--deep]\n  \
                 coordinode version\n  \
                 coordinode admin node join --node CLUSTER_ADDR --id NODE_ID --addr NODE_ADDR [--pre-seeded] [--follow]\n  \
                 coordinode admin node decommission --node CLUSTER_ADDR --id NODE_ID [--pruning] [--force] [--skip-confirmation]\n",
                env!("CARGO_PKG_VERSION")
            );
            std::process::exit(1);
        }
    }
}

/// Parse `coordinode admin <subcommand> ...` arguments.
///
/// Supports:
/// - `coordinode admin node join --node ADDR --id ID --addr ADDR [--pre-seeded] [--follow]`
/// - `coordinode admin node decommission --node ADDR --id ID [--pruning] [--force] [--skip-confirmation]`
fn parse_admin_args(args: &[String]) -> Command {
    // args[1] = "admin", args[2] should be the admin object
    let object = args.get(2).map(|s| s.as_str()).unwrap_or("");
    let subcommand = args.get(3).map(|s| s.as_str()).unwrap_or("");

    match (object, subcommand) {
        ("node", "join") => {
            let cluster_addr = match find_flag(args, "--node") {
                Some(a) => a,
                None => {
                    eprintln!("error: --node CLUSTER_ADDR is required for admin node join");
                    std::process::exit(1);
                }
            };
            let node_id_str = match find_flag(args, "--id") {
                Some(s) => s,
                None => {
                    eprintln!("error: --id NODE_ID is required for admin node join");
                    std::process::exit(1);
                }
            };
            let node_id: u64 = match node_id_str.parse() {
                Ok(id) => id,
                Err(_) => {
                    eprintln!("error: --id must be a positive integer, got '{node_id_str}'");
                    std::process::exit(1);
                }
            };
            let node_addr = match find_flag(args, "--addr") {
                Some(a) => a,
                None => {
                    eprintln!("error: --addr NODE_ADDR is required for admin node join");
                    std::process::exit(1);
                }
            };
            let pre_seeded = args.iter().any(|a| a == "--pre-seeded");
            let follow = args.iter().any(|a| a == "--follow");
            Command::AdminNodeJoin {
                cluster_addr,
                node_id,
                node_addr,
                pre_seeded,
                follow,
            }
        }
        ("node", "decommission") => {
            let cluster_addr = match find_flag(args, "--node") {
                Some(a) => a,
                None => {
                    eprintln!("error: --node CLUSTER_ADDR is required for admin node decommission");
                    std::process::exit(1);
                }
            };
            let node_id_str = match find_flag(args, "--id") {
                Some(s) => s,
                None => {
                    eprintln!("error: --id NODE_ID is required for admin node decommission");
                    std::process::exit(1);
                }
            };
            let node_id: u64 = match node_id_str.parse() {
                Ok(id) => id,
                Err(_) => {
                    eprintln!("error: --id must be a positive integer, got '{node_id_str}'");
                    std::process::exit(1);
                }
            };
            let pruning = args.iter().any(|a| a == "--pruning");
            let force = args.iter().any(|a| a == "--force");
            let skip_confirmation = args.iter().any(|a| a == "--skip-confirmation");
            Command::AdminNodeDecommission {
                cluster_addr,
                node_id,
                pruning,
                force,
                skip_confirmation,
            }
        }
        _ => {
            eprintln!(
                "coordinode admin: unknown subcommand '{object} {subcommand}'\n\n\
                 Admin commands:\n  \
                 coordinode admin node join --node CLUSTER_ADDR --id NODE_ID --addr NODE_ADDR [--pre-seeded] [--follow]\n  \
                 coordinode admin node decommission --node CLUSTER_ADDR --id NODE_ID [--pruning] [--force] [--skip-confirmation]\n"
            );
            std::process::exit(1);
        }
    }
}

fn default_serve() -> Command {
    Command::Serve {
        mode: ServeMode::Full,
        node_id: 1,
        grpc_addr: "[::]:7080".to_string(),
        advertise_addr: None,
        #[cfg(feature = "rest-proxy")]
        rest_addr: "[::]:7081".to_string(),
        ops_addr: "[::]:7084".to_string(),
        data_dir: "./data".to_string(),
        peers: None,
        nofile: None,
        max_connections: None,
        max_request_size_mb: 16,
        request_timeout_secs: None,
        http2_keepalive_secs: None,
        cache_size_mb: None,
        write_buffer_mb: None,
    }
}

fn find_flag(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Parse a numeric flag value, exiting with a clear message on a bad value.
/// Absent flag returns `None`.
fn find_flag_num<T>(args: &[String], flag: &str) -> Option<T>
where
    T: std::str::FromStr,
{
    find_flag(args, flag).map(|s| {
        s.parse().unwrap_or_else(|_| {
            eprintln!("error: {flag} requires a valid number, got '{s}'");
            std::process::exit(1);
        })
    })
}

/// Parse --format flag (default: json).
fn parse_format(args: &[String]) -> BackupFormat {
    match find_flag(args, "--format").as_deref() {
        Some("json") | None => BackupFormat::Json,
        Some("cypher") => BackupFormat::Cypher,
        Some("binary") => BackupFormat::Binary,
        // Import-only Neo4j formats (restore only; backup rejects them).
        Some("apoc-json") | Some("apoc_json") => BackupFormat::ApocJson,
        Some("apoc-cypher") | Some("apoc_cypher") => BackupFormat::ApocCypher,
        // Import-only Hetionet hetnet JSON.
        Some("hetio-json") | Some("hetio") => BackupFormat::HetioJson,
        // Full Raft data snapshot (backup and restore).
        Some("snapshot") | Some("raft-snapshot") | Some("raft_snapshot") => {
            BackupFormat::RaftSnapshot
        }
        Some(other) => {
            eprintln!(
                "error: unknown format '{other}'. Use: json, cypher, binary, snapshot \
                 (backup/restore) or apoc-json, apoc-cypher, hetio-json (restore only)"
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn backup_json_default_format() {
        let cmd = parse_args_from(&args("coordinode backup --output /tmp/backup.json"));
        match cmd {
            Command::Backup {
                output,
                format,
                namespace,
                ..
            } => {
                assert_eq!(output, "/tmp/backup.json");
                assert_eq!(format, BackupFormat::Json);
                assert!(namespace.is_none());
            }
            _ => panic!("expected Backup command"),
        }
    }

    #[test]
    fn backup_binary_with_namespace() {
        let cmd = parse_args_from(&args(
            "coordinode backup --output /tmp/b.bin --format binary --namespace prod --data /db",
        ));
        match cmd {
            Command::Backup {
                data_dir,
                output,
                format,
                namespace,
                ..
            } => {
                assert_eq!(data_dir, "/db");
                assert_eq!(output, "/tmp/b.bin");
                assert_eq!(format, BackupFormat::Binary);
                assert_eq!(namespace.as_deref(), Some("prod"));
            }
            _ => panic!("expected Backup command"),
        }
    }

    #[test]
    fn backup_cypher_format() {
        let cmd = parse_args_from(&args(
            "coordinode backup --output dump.cypher --format cypher",
        ));
        match cmd {
            Command::Backup { format, .. } => assert_eq!(format, BackupFormat::Cypher),
            _ => panic!("expected Backup command"),
        }
    }

    #[test]
    fn backup_snapshot_format() {
        let cmd = parse_args_from(&args(
            "coordinode backup --output db.snap --format snapshot",
        ));
        match cmd {
            Command::Backup { format, .. } => assert_eq!(format, BackupFormat::RaftSnapshot),
            _ => panic!("expected Backup command"),
        }
    }

    #[test]
    fn restore_snapshot_format() {
        let cmd = parse_args_from(&args(
            "coordinode restore --input db.snap --format raft-snapshot",
        ));
        match cmd {
            Command::Restore { format, .. } => assert_eq!(format, BackupFormat::RaftSnapshot),
            _ => panic!("expected Restore command"),
        }
    }

    #[test]
    fn restore_json() {
        let cmd = parse_args_from(&args(
            "coordinode restore --input /tmp/backup.json --data /db2",
        ));
        match cmd {
            Command::Restore {
                data_dir,
                input,
                format,
                namespace,
                ..
            } => {
                assert_eq!(data_dir, "/db2");
                assert_eq!(input, "/tmp/backup.json");
                assert_eq!(format, BackupFormat::Json);
                assert!(namespace.is_none());
            }
            _ => panic!("expected Restore command"),
        }
    }

    #[test]
    fn restore_only_labels_parsed() {
        let cmd = parse_args_from(&args(
            "coordinode restore --input x.json --only-labels User,Post",
        ));
        match cmd {
            Command::Restore { only_labels, .. } => {
                assert_eq!(only_labels, vec!["User".to_string(), "Post".to_string()]);
            }
            _ => panic!("expected Restore command"),
        }
    }

    #[test]
    fn restore_force_flag_parsed() {
        let cmd = parse_args_from(&args(
            "coordinode restore --input db.bin --format binary --force",
        ));
        match cmd {
            Command::Restore { force, .. } => assert!(force, "--force should set force=true"),
            _ => panic!("expected Restore command"),
        }
    }

    #[test]
    fn restore_force_defaults_false() {
        let cmd = parse_args_from(&args("coordinode restore --input db.bin --format binary"));
        match cmd {
            Command::Restore { force, .. } => assert!(!force, "force defaults to false"),
            _ => panic!("expected Restore command"),
        }
    }

    #[test]
    fn restore_binary_with_namespace() {
        let cmd = parse_args_from(&args(
            "coordinode restore --input b.bin --format binary --namespace staging",
        ));
        match cmd {
            Command::Restore {
                format, namespace, ..
            } => {
                assert_eq!(format, BackupFormat::Binary);
                assert_eq!(namespace.as_deref(), Some("staging"));
            }
            _ => panic!("expected Restore command"),
        }
    }

    #[test]
    fn default_is_serve() {
        let cmd = parse_args_from(&args("coordinode"));
        assert!(matches!(cmd, Command::Serve { .. }));
    }

    #[test]
    fn serve_default_mode_is_full() {
        let cmd = parse_args_from(&args("coordinode serve"));
        match cmd {
            Command::Serve {
                mode,
                node_id,
                advertise_addr,
                ..
            } => {
                assert_eq!(mode, ServeMode::Full);
                assert_eq!(node_id, 1);
                assert!(advertise_addr.is_none());
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn serve_explicit_mode_full() {
        let cmd = parse_args_from(&args("coordinode serve --mode full"));
        match cmd {
            Command::Serve { mode, .. } => assert_eq!(mode, ServeMode::Full),
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn serve_node_id() {
        let cmd = parse_args_from(&args(
            "coordinode serve --node-id 3 --peers node1:7080,node2:7080",
        ));
        match cmd {
            Command::Serve { node_id, peers, .. } => {
                assert_eq!(node_id, 3);
                let p = peers.unwrap_or_default();
                assert_eq!(p.len(), 2);
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn serve_resource_flags_parsed() {
        let cmd = parse_args_from(&args(
            "coordinode serve --nofile 262144 --max-connections 1024 \
             --max-request-size-mb 32 --request-timeout-secs 30 \
             --http2-keepalive-secs 60 --cache-size-mb 4096 --write-buffer-mb 256",
        ));
        match cmd {
            Command::Serve {
                nofile,
                max_connections,
                max_request_size_mb,
                request_timeout_secs,
                http2_keepalive_secs,
                cache_size_mb,
                write_buffer_mb,
                ..
            } => {
                assert_eq!(nofile, Some(262144));
                assert_eq!(max_connections, Some(1024));
                assert_eq!(max_request_size_mb, 32);
                assert_eq!(request_timeout_secs, Some(30));
                assert_eq!(http2_keepalive_secs, Some(60));
                assert_eq!(cache_size_mb, Some(4096));
                assert_eq!(write_buffer_mb, Some(256));
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn serve_resource_flags_default() {
        let cmd = parse_args_from(&args("coordinode serve"));
        match cmd {
            Command::Serve {
                nofile,
                max_connections,
                max_request_size_mb,
                request_timeout_secs,
                cache_size_mb,
                ..
            } => {
                // Unset network/storage knobs stay None; the request-size cap has
                // a safe default.
                assert!(nofile.is_none());
                assert!(max_connections.is_none());
                assert_eq!(max_request_size_mb, 16);
                assert!(request_timeout_secs.is_none());
                assert!(cache_size_mb.is_none());
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn serve_advertise_addr() {
        let cmd = parse_args_from(&args("coordinode serve --advertise-addr node1:7080"));
        match cmd {
            Command::Serve { advertise_addr, .. } => {
                assert_eq!(advertise_addr.as_deref(), Some("node1:7080"));
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[cfg(feature = "rest-proxy")]
    #[test]
    fn serve_custom_rest_addr() {
        let cmd = parse_args_from(&args("coordinode serve --rest-addr 0.0.0.0:8081"));
        match cmd {
            Command::Serve { rest_addr, .. } => assert_eq!(rest_addr, "0.0.0.0:8081"),
            _ => panic!("expected Serve command"),
        }
    }

    #[cfg(feature = "rest-proxy")]
    #[test]
    fn serve_default_rest_addr() {
        let cmd = parse_args_from(&args("coordinode serve"));
        match cmd {
            Command::Serve { rest_addr, .. } => assert_eq!(rest_addr, "[::]:7081"),
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn version_command() {
        let cmd = parse_args_from(&args("coordinode version"));
        assert!(matches!(cmd, Command::Version));
    }

    #[test]
    fn admin_node_join_required_flags() {
        let cmd = parse_args_from(&args(
            "coordinode admin node join --node http://leader:7080 --id 3 --addr node3:7080",
        ));
        match cmd {
            Command::AdminNodeJoin {
                cluster_addr,
                node_id,
                node_addr,
                pre_seeded,
                follow,
            } => {
                assert_eq!(cluster_addr, "http://leader:7080");
                assert_eq!(node_id, 3);
                assert_eq!(node_addr, "node3:7080");
                assert!(!pre_seeded);
                assert!(!follow);
            }
            _ => panic!("expected AdminNodeJoin"),
        }
    }

    #[test]
    fn admin_node_join_with_flags() {
        let cmd = parse_args_from(&args(
            "coordinode admin node join --node http://n1:7080 --id 5 --addr n5:7080 --pre-seeded --follow",
        ));
        match cmd {
            Command::AdminNodeJoin {
                node_id,
                pre_seeded,
                follow,
                ..
            } => {
                assert_eq!(node_id, 5);
                assert!(pre_seeded);
                assert!(follow);
            }
            _ => panic!("expected AdminNodeJoin"),
        }
    }

    #[test]
    fn admin_node_decommission_required_flags() {
        let cmd = parse_args_from(&args(
            "coordinode admin node decommission --node http://leader:7080 --id 3",
        ));
        match cmd {
            Command::AdminNodeDecommission {
                cluster_addr,
                node_id,
                pruning,
                force,
                skip_confirmation,
            } => {
                assert_eq!(cluster_addr, "http://leader:7080");
                assert_eq!(node_id, 3);
                assert!(!pruning);
                assert!(!force);
                assert!(!skip_confirmation);
            }
            _ => panic!("expected AdminNodeDecommission"),
        }
    }

    #[test]
    fn admin_node_decommission_with_pruning() {
        let cmd = parse_args_from(&args(
            "coordinode admin node decommission --node http://n1:7080 --id 2 --pruning",
        ));
        match cmd {
            Command::AdminNodeDecommission {
                node_id,
                pruning,
                force,
                ..
            } => {
                assert_eq!(node_id, 2);
                assert!(pruning);
                assert!(!force);
            }
            _ => panic!("expected AdminNodeDecommission"),
        }
    }

    #[test]
    fn admin_node_decommission_force_requires_skip_confirmation() {
        // CLI parsing sets both flags independently — the enforcement is in the server.
        let cmd = parse_args_from(&args(
            "coordinode admin node decommission --node http://n1:7080 --id 4 --force --skip-confirmation",
        ));
        match cmd {
            Command::AdminNodeDecommission {
                node_id,
                force,
                skip_confirmation,
                ..
            } => {
                assert_eq!(node_id, 4);
                assert!(force);
                assert!(skip_confirmation);
            }
            _ => panic!("expected AdminNodeDecommission"),
        }
    }
}
