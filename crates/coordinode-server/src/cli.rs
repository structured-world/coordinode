//! CLI argument parsing for coordinode subcommands.
//!
//! Subcommands:
//! - `serve` (default) — start gRPC + ops servers
//! - `version` — print version
//! - `verify --deep` — verify storage integrity
//! - `backup` — export database to file
//! - `restore` — import database from file

use coordinode_embed::backup::BackupFormat;

use crate::config::CliOverrides;

/// Parsed CLI command.
pub enum Command {
    /// Start the database server (default). The resolved configuration is
    /// `ServerConfig::default()` overlaid by the `--config` YAML file (if any)
    /// then by these command-line `overrides` (CLI wins). Boxed because the
    /// override set is far larger than any other subcommand's payload.
    Serve {
        /// `--config <path>`: optional YAML config file. Absent = built-in
        /// defaults; present-but-unreadable / malformed = startup error.
        config_path: Option<String>,
        /// Command-line flag overrides, folded over the config file last so
        /// the command line beats the file.
        overrides: Box<CliOverrides>,
    },
    /// Print version and exit.
    Version,
    /// Verify storage integrity.
    Verify {
        /// Single-endpoint data directory (used when `config_path` is `None`).
        data_dir: String,
        /// Optional config file. When given, the storage topology (including a
        /// multi-endpoint layout) is resolved from it and `data_dir` is ignored.
        config_path: Option<String>,
        /// Deep verification (checksums on all pages).
        deep: bool,
    },
    /// Export database to a backup file.
    ///
    /// Takes a consistent MVCC snapshot at the start — ongoing writes
    /// are not blocked during backup.
    Backup {
        /// Single-endpoint data directory (used when `config_path` is `None`).
        data_dir: String,
        /// Optional config file. When given, the storage topology is resolved
        /// from it and `data_dir` is ignored.
        config_path: Option<String>,
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
        /// Single-endpoint data directory (used when `config_path` is `None`).
        data_dir: String,
        /// Optional config file. When given, the storage topology is resolved
        /// from it and `data_dir` is ignored.
        config_path: Option<String>,
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
        /// Single-endpoint data directory (used when `config_path` is `None`).
        data_dir: String,
        /// Optional config file. When given, the storage topology is resolved
        /// from it and `data_dir` is ignored.
        config_path: Option<String>,
        /// Output directory for the checkpoint (must not exist).
        output: String,
    },
    /// Compact the database offline: major-compact every partition and fold
    /// accumulated merge operands (adjacency, counters) into single values.
    ///
    /// Run after a bulk import to collapse the per-edge merge operands a
    /// super-node accumulates, so traversal reads cost O(1) in operand count
    /// instead of O(operands). The server must not be running against the same
    /// data directory.
    Compact {
        /// Single-endpoint data directory (used when `config_path` is `None`).
        data_dir: String,
        /// Optional config file. When given, the storage topology is resolved
        /// from it and `data_dir` is ignored.
        config_path: Option<String>,
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
            // Flags fill an all-`Option` override set: a flag left off the
            // command line stays `None` so the config-file / default value
            // stands. Defaults and cross-field validation (mode resolution,
            // node-id-requires-peers) move to the resolution step in `main`,
            // because the missing side may be supplied by the config file.
            let config_path = find_flag(args, "--config");
            // Validate the numeric format of --node-id at parse time (cheap,
            // value-independent); the >1-requires-peers rule is checked after
            // the config file is merged.
            let node_id: Option<u64> = match find_flag(args, "--node-id") {
                None => None,
                Some(s) => match s.parse() {
                    Ok(id) if id > 0 => Some(id),
                    _ => {
                        eprintln!("error: --node-id must be a positive integer, got '{s}'");
                        std::process::exit(1);
                    }
                },
            };
            let peers = find_flag(args, "--peers").map(|p| {
                p.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            });
            let overrides = CliOverrides {
                mode: find_flag(args, "--mode"),
                node_id,
                grpc_addr: find_flag(args, "--addr"),
                advertise_addr: find_flag(args, "--advertise-addr"),
                rest_addr: find_flag(args, "--rest-addr"),
                ops_addr: find_flag(args, "--ops-addr"),
                data_dir: find_flag(args, "--data"),
                peers,
                nofile: find_flag_num(args, "--nofile"),
                max_connections: find_flag_num(args, "--max-connections"),
                max_request_size_mb: find_flag_num(args, "--max-request-size-mb"),
                request_timeout_secs: find_flag_num(args, "--request-timeout-secs"),
                http2_keepalive_secs: find_flag_num(args, "--http2-keepalive-secs"),
                cache_size_mb: find_flag_num(args, "--cache-size-mb"),
                write_buffer_mb: find_flag_num(args, "--write-buffer-mb"),
                retention_window_secs: find_flag_num(args, "--retention-window-secs"),
                registry_heartbeat_ms: find_flag_num(args, "--registry-heartbeat-ms"),
                registry_eviction_ms: find_flag_num(args, "--registry-eviction-ms"),
                cdc_consumer_ttl_secs: find_flag_num(args, "--cdc-consumer-ttl-secs"),
                interactive_txn_idle_timeout_secs: find_flag_num(
                    args,
                    "--interactive-txn-idle-timeout-secs",
                ),
                interactive_txn_max_bytes: find_flag_num(args, "--interactive-txn-max-bytes"),
                wire_compression_level: find_flag_num(args, "--wire-compression-level"),
                tls_cert: find_flag(args, "--tls-cert"),
                tls_key: find_flag(args, "--tls-key"),
                tls_ca: find_flag(args, "--tls-ca"),
                tls_require_client_auth: args
                    .iter()
                    .any(|a| a == "--tls-require-client-auth")
                    .then_some(true),
                scrub_enabled: args.iter().any(|a| a == "--no-scrub").then_some(false),
                scrub_interval_secs: find_flag_num(args, "--scrub-interval-secs"),
                scrub_throttle_ms: find_flag_num(args, "--scrub-throttle-ms"),
                checkpoint_enabled: args.iter().any(|a| a == "--no-checkpoint").then_some(false),
                checkpoint_interval_secs: find_flag_num(args, "--checkpoint-interval-secs"),
                checkpoint_dir: find_flag(args, "--checkpoint-dir"),
                checkpoint_keep: find_flag_num(args, "--checkpoint-keep"),
            };
            Command::Serve {
                config_path,
                overrides: Box::new(overrides),
            }
        }
        "version" | "--version" | "-v" => Command::Version,
        "verify" => {
            let data_dir = find_flag(args, "--data").unwrap_or_else(|| "./data".to_string());
            let config_path = find_flag(args, "--config");
            let deep = args.iter().any(|a| a == "--deep");
            Command::Verify {
                data_dir,
                config_path,
                deep,
            }
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
            let config_path = find_flag(args, "--config");
            Command::Backup {
                data_dir,
                config_path,
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
            let config_path = find_flag(args, "--config");
            Command::Restore {
                data_dir,
                config_path,
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
            let config_path = find_flag(args, "--config");
            Command::Checkpoint {
                data_dir,
                config_path,
                output,
            }
        }
        "compact" => {
            let data_dir = find_flag(args, "--data").unwrap_or_else(|| "./data".to_string());
            let config_path = find_flag(args, "--config");
            Command::Compact {
                data_dir,
                config_path,
            }
        }
        "admin" => parse_admin_args(args),
        _ => {
            eprintln!(
                "coordinode v{}\n\n\
                 Usage:\n  \
                 coordinode serve [--mode full] [--node-id N] [--addr ADDR] [--advertise-addr ADDR]\n          \
                 [--rest-addr ADDR] [--ops-addr ADDR] [--data DIR] [--peers PEERS]\n          \
                 [--nofile N] [--max-connections N] [--max-request-size-mb N] [--request-timeout-secs N]\n          \
                 [--http2-keepalive-secs N] [--cache-size-mb N] [--write-buffer-mb N]\n          \
                 [--retention-window-secs N] [--registry-heartbeat-ms N] [--registry-eviction-ms N]\n          \
                 [--cdc-consumer-ttl-secs N]\n          \
                 [--wire-compression-level N]\n          \
                 [--tls-cert FILE --tls-key FILE] [--tls-ca FILE] [--tls-require-client-auth]\n          \
                 [--no-scrub] [--scrub-interval-secs N] [--scrub-throttle-ms N]\n          \
                 [--no-checkpoint] [--checkpoint-interval-secs N] [--checkpoint-dir DIR] [--checkpoint-keep N]\n  \
                 coordinode backup --output FILE [--data DIR | --config FILE] [--format json|cypher|binary|snapshot] [--namespace NS] [--since SEQNO]\n  \
                 coordinode restore --input FILE [--data DIR | --config FILE] [--format json|cypher|binary|snapshot|apoc-json|apoc-cypher|hetio-json] [--namespace NS] [--only-labels L1,L2] [--force]\n  \
                 coordinode checkpoint --output DIR [--data DIR | --config FILE]\n  \
                 coordinode compact [--data DIR | --config FILE]\n  \
                 coordinode verify [--data DIR | --config FILE] [--deep]\n  \
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
        config_path: None,
        overrides: Box::new(CliOverrides::default()),
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
mod tests;
