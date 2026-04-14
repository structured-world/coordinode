//! CLI argument parsing for coordinode subcommands.
//!
//! Subcommands:
//! - `serve` (default) — start gRPC + ops servers
//! - `version` — print version
//! - `verify --deep` — verify storage integrity
//! - `backup` — export database to file
//! - `restore` — import database from file

use coordinode_embed::backup::BackupFormat;

/// Parsed CLI command.
pub enum Command {
    /// Start the database server (default).
    Serve {
        /// gRPC listen address (default: [::]:7080).
        grpc_addr: String,
        /// Operational HTTP server address for /metrics and /health (default: [::]:7084).
        /// Pass port 0 to let the OS assign an ephemeral port (useful in tests).
        ops_addr: String,
        /// Data directory (default: ./data).
        data_dir: String,
        /// Peer addresses for cluster mode (comma-separated).
        /// When provided, enables Raft consensus with the given peers.
        /// Example: --peers "node2:7080,node3:7080"
        peers: Option<Vec<String>>,
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
            let grpc_addr = find_flag(args, "--addr").unwrap_or_else(|| "[::]:7080".to_string());
            let ops_addr = find_flag(args, "--ops-addr").unwrap_or_else(|| "[::]:7084".to_string());
            let data_dir = find_flag(args, "--data").unwrap_or_else(|| "./data".to_string());
            let peers = find_flag(args, "--peers").map(|p| {
                p.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            });
            Command::Serve {
                grpc_addr,
                ops_addr,
                data_dir,
                peers,
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
            Command::Backup {
                data_dir,
                output,
                format,
                namespace,
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
            Command::Restore {
                data_dir,
                input,
                format,
                namespace,
            }
        }
        "admin" => parse_admin_args(args),
        _ => {
            eprintln!(
                "coordinode v{}\n\n\
                 Usage:\n  \
                 coordinode serve [--addr ADDR] [--data DIR] [--peers PEERS]\n  \
                 coordinode backup --output FILE [--data DIR] [--format json|cypher|binary] [--namespace NS]\n  \
                 coordinode restore --input FILE [--data DIR] [--format json|cypher|binary] [--namespace NS]\n  \
                 coordinode verify [--data DIR] [--deep]\n  \
                 coordinode version\n  \
                 coordinode admin node join --node CLUSTER_ADDR --id NODE_ID --addr NODE_ADDR [--pre-seeded] [--follow]\n",
                env!("CARGO_PKG_VERSION")
            );
            std::process::exit(1);
        }
    }
}

/// Parse `coordinode admin <subcommand> ...` arguments.
///
/// Currently supports: `coordinode admin node join --node ADDR --id ID --addr ADDR`
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
        _ => {
            eprintln!(
                "coordinode admin: unknown subcommand '{object} {subcommand}'\n\n\
                 Admin commands:\n  \
                 coordinode admin node join --node CLUSTER_ADDR --id NODE_ID --addr NODE_ADDR [--pre-seeded] [--follow]\n"
            );
            std::process::exit(1);
        }
    }
}

fn default_serve() -> Command {
    Command::Serve {
        grpc_addr: "[::]:7080".to_string(),
        ops_addr: "[::]:7084".to_string(),
        data_dir: "./data".to_string(),
        peers: None,
    }
}

fn find_flag(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Parse --format flag (default: json).
fn parse_format(args: &[String]) -> BackupFormat {
    match find_flag(args, "--format").as_deref() {
        Some("json") | None => BackupFormat::Json,
        Some("cypher") => BackupFormat::Cypher,
        Some("binary") => BackupFormat::Binary,
        Some(other) => {
            eprintln!("error: unknown format '{other}'. Use: json, cypher, or binary");
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
}
