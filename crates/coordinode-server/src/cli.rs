//! CLI argument parsing for coordinode subcommands.
//!
//! Subcommands:
//! - `serve` (default) — start gRPC + ops servers
//! - `version` — print version
//! - `verify --deep` — verify storage integrity
//!
//! Future subcommands (when storage wiring is complete):
//! - `import <file>` — bulk import JSON/CSV
//! - `backup <path>` — snapshot backup

/// Parsed CLI command.
pub enum Command {
    /// Start the database server (default).
    Serve {
        /// gRPC listen address (default: [::]:7080).
        grpc_addr: String,
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
}

/// Parse command line arguments.
pub fn parse_args() -> Command {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        return default_serve();
    }

    match args[1].as_str() {
        "serve" => {
            let grpc_addr = find_flag(&args, "--addr").unwrap_or_else(|| "[::]:7080".to_string());
            let data_dir = find_flag(&args, "--data").unwrap_or_else(|| "./data".to_string());
            let peers = find_flag(&args, "--peers").map(|p| {
                p.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            });
            Command::Serve {
                grpc_addr,
                data_dir,
                peers,
            }
        }
        "version" | "--version" | "-v" => Command::Version,
        "verify" => {
            let data_dir = find_flag(&args, "--data").unwrap_or_else(|| "./data".to_string());
            let deep = args.iter().any(|a| a == "--deep");
            Command::Verify { data_dir, deep }
        }
        _ => {
            eprintln!(
                "coordinode v{}\n\nUsage:\n  coordinode serve [--addr ADDR] [--data DIR]\n  coordinode version\n  coordinode verify [--data DIR] [--deep]\n",
                env!("CARGO_PKG_VERSION")
            );
            std::process::exit(1);
        }
    }
}

fn default_serve() -> Command {
    Command::Serve {
        grpc_addr: "[::]:7080".to_string(),
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
