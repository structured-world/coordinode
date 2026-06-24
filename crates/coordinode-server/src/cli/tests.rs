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

/// Bare `serve` leaves every override `None` and no config path, so the
/// resolution step in `main` falls back to the built-in defaults.
#[test]
fn serve_bare_has_no_overrides_and_no_config() {
    let cmd = parse_args_from(&args("coordinode serve"));
    match cmd {
        Command::Serve {
            config_path,
            overrides,
        } => {
            assert!(config_path.is_none());
            assert!(overrides.mode.is_none());
            assert!(overrides.node_id.is_none());
            assert!(overrides.advertise_addr.is_none());
        }
        _ => panic!("expected Serve command"),
    }
}

#[test]
fn serve_config_path_parsed() {
    let cmd = parse_args_from(&args(
        "coordinode serve --config /etc/coordinode/coordinode.conf",
    ));
    match cmd {
        Command::Serve { config_path, .. } => {
            assert_eq!(
                config_path.as_deref(),
                Some("/etc/coordinode/coordinode.conf")
            );
        }
        _ => panic!("expected Serve command"),
    }
}

#[test]
fn serve_explicit_mode_full() {
    let cmd = parse_args_from(&args("coordinode serve --mode full"));
    match cmd {
        Command::Serve { overrides, .. } => {
            assert_eq!(overrides.mode.as_deref(), Some("full"))
        }
        _ => panic!("expected Serve command"),
    }
}

#[test]
fn serve_node_id() {
    let cmd = parse_args_from(&args(
        "coordinode serve --node-id 3 --peers node1:7080,node2:7080",
    ));
    match cmd {
        Command::Serve { overrides, .. } => {
            assert_eq!(overrides.node_id, Some(3));
            assert_eq!(overrides.peers.as_ref().map(|p| p.len()), Some(2));
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
        Command::Serve { overrides, .. } => {
            assert_eq!(overrides.nofile, Some(262144));
            assert_eq!(overrides.max_connections, Some(1024));
            assert_eq!(overrides.max_request_size_mb, Some(32));
            assert_eq!(overrides.request_timeout_secs, Some(30));
            assert_eq!(overrides.http2_keepalive_secs, Some(60));
            assert_eq!(overrides.cache_size_mb, Some(4096));
            assert_eq!(overrides.write_buffer_mb, Some(256));
        }
        _ => panic!("expected Serve command"),
    }
}

#[test]
fn serve_wire_compression_level_parsed_and_applied() {
    use crate::config::ServerConfig;
    let cmd = parse_args_from(&args("coordinode serve --wire-compression-level 22"));
    match cmd {
        Command::Serve { overrides, .. } => {
            assert_eq!(overrides.wire_compression_level, Some(22));
            let mut cfg = ServerConfig::default();
            assert_eq!(cfg.wire_compression_level, 3, "default wire level is 3");
            cfg.apply_overrides(&overrides);
            assert_eq!(cfg.wire_compression_level, 22, "CLI override applied");
        }
        _ => panic!("expected Serve command"),
    }
}

#[test]
fn serve_resource_flags_default_to_none() {
    // Unset → None so the resolution step keeps config-file / built-in
    // defaults (incl. the 16 MiB request-size cap).
    let cmd = parse_args_from(&args("coordinode serve"));
    match cmd {
        Command::Serve { overrides, .. } => {
            assert!(overrides.nofile.is_none());
            assert!(overrides.max_connections.is_none());
            assert!(overrides.max_request_size_mb.is_none());
            assert!(overrides.request_timeout_secs.is_none());
            assert!(overrides.cache_size_mb.is_none());
        }
        _ => panic!("expected Serve command"),
    }
}

#[test]
fn serve_registry_flags_parsed() {
    let cmd = parse_args_from(&args(
        "coordinode serve --retention-window-secs 3600 \
             --registry-heartbeat-ms 50 --registry-eviction-ms 500",
    ));
    match cmd {
        Command::Serve { overrides, .. } => {
            assert_eq!(overrides.retention_window_secs, Some(3600));
            assert_eq!(overrides.registry_heartbeat_ms, Some(50));
            assert_eq!(overrides.registry_eviction_ms, Some(500));
        }
        _ => panic!("expected Serve command"),
    }
}

#[test]
fn serve_registry_flags_default_to_none() {
    // Unset → None so the server keeps the registry's built-in defaults
    // (7-day retention window, 100 ms heartbeat, 1000 ms eviction sweep).
    let cmd = parse_args_from(&args("coordinode serve"));
    match cmd {
        Command::Serve { overrides, .. } => {
            assert!(overrides.retention_window_secs.is_none());
            assert!(overrides.registry_heartbeat_ms.is_none());
            assert!(overrides.registry_eviction_ms.is_none());
        }
        _ => panic!("expected Serve command"),
    }
}

#[test]
fn serve_interactive_txn_flags_parsed() {
    let cmd = parse_args_from(&args(
        "coordinode serve --interactive-txn-idle-timeout-secs 60 \
             --interactive-txn-max-bytes 1048576",
    ));
    match cmd {
        Command::Serve { overrides, .. } => {
            assert_eq!(overrides.interactive_txn_idle_timeout_secs, Some(60));
            assert_eq!(overrides.interactive_txn_max_bytes, Some(1_048_576));
        }
        _ => panic!("expected Serve command"),
    }
}

#[test]
fn serve_interactive_txn_flags_default_to_none() {
    // Unset → None so the server keeps the ADR-042 defaults (30s idle
    // timeout, 256 MiB buffered-write ceiling per interactive transaction).
    let cmd = parse_args_from(&args("coordinode serve"));
    match cmd {
        Command::Serve { overrides, .. } => {
            assert!(overrides.interactive_txn_idle_timeout_secs.is_none());
            assert!(overrides.interactive_txn_max_bytes.is_none());
        }
        _ => panic!("expected Serve command"),
    }
}

#[test]
fn serve_advertise_addr() {
    let cmd = parse_args_from(&args("coordinode serve --advertise-addr node1:7080"));
    match cmd {
        Command::Serve { overrides, .. } => {
            assert_eq!(overrides.advertise_addr.as_deref(), Some("node1:7080"));
        }
        _ => panic!("expected Serve command"),
    }
}

#[test]
fn serve_custom_rest_addr() {
    let cmd = parse_args_from(&args("coordinode serve --rest-addr 0.0.0.0:8081"));
    match cmd {
        Command::Serve { overrides, .. } => {
            assert_eq!(overrides.rest_addr.as_deref(), Some("0.0.0.0:8081"))
        }
        _ => panic!("expected Serve command"),
    }
}

#[test]
fn serve_default_rest_addr_is_none() {
    let cmd = parse_args_from(&args("coordinode serve"));
    match cmd {
        Command::Serve { overrides, .. } => assert!(overrides.rest_addr.is_none()),
        _ => panic!("expected Serve command"),
    }
}

/// End-to-end resolution exactly as `main` performs it: a YAML config file
/// on disk supplies some knobs, the command line overrides a subset, and
/// the rest fall through to the built-in defaults. Proves the three-layer
/// precedence (CLI > config file > default) through the real CLI parser.
#[test]
#[allow(clippy::unwrap_used)]
fn serve_config_file_plus_cli_resolves_with_correct_precedence() {
    use crate::config::ServerConfig;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("coordinode.conf");
    // File sets node_id, grpc_addr, and the idle timeout.
    std::fs::write(
        &path,
        "node_id: 5\n\
             grpc_addr: \"0.0.0.0:9999\"\n\
             interactive_txn_idle_timeout_secs: 90\n\
             peers:\n  - \"n1:7080\"\n  - \"n2:7080\"\n",
    )
    .unwrap();
    let path_str = path.to_str().unwrap();

    // CLI overrides node_id (beats the file) and sets the data dir; leaves
    // grpc_addr and the idle timeout to the file, and ops_addr to default.
    let argv = args(&format!(
        "coordinode serve --config {path_str} --node-id 7 --data /var/lib/coordinode"
    ));
    let (config_path, overrides) = match parse_args_from(&argv) {
        Command::Serve {
            config_path,
            overrides,
        } => (config_path, overrides),
        _ => panic!("expected Serve command"),
    };
    assert_eq!(config_path.as_deref(), Some(path_str));

    let mut cfg = ServerConfig::load(config_path.as_deref()).unwrap();
    cfg.apply_overrides(&overrides);

    // CLI wins.
    assert_eq!(cfg.node_id, 7, "CLI --node-id beats the config file");
    assert_eq!(cfg.data_dir, "/var/lib/coordinode", "CLI --data applied");
    // File wins over default.
    assert_eq!(
        cfg.grpc_addr, "0.0.0.0:9999",
        "config file grpc_addr stands"
    );
    assert_eq!(
        cfg.interactive_txn_idle_timeout_secs, 90,
        "config file idle timeout stands"
    );
    assert_eq!(cfg.peers, vec!["n1:7080", "n2:7080"], "config file peers");
    // Untouched by both → built-in default.
    assert_eq!(cfg.ops_addr, "[::]:7084", "default ops_addr");
    assert_eq!(cfg.max_request_size_mb, 16, "default request-size cap");
    assert_eq!(
        cfg.interactive_txn_max_bytes,
        256 * 1024 * 1024,
        "default interactive-txn byte ceiling"
    );
}

#[test]
fn version_command() {
    let cmd = parse_args_from(&args("coordinode version"));
    assert!(matches!(cmd, Command::Version));
}

#[test]
fn compact_uses_explicit_data_dir() {
    let cmd = parse_args_from(&args("coordinode compact --data /var/lib/coordinode"));
    match cmd {
        Command::Compact {
            data_dir,
            config_path,
        } => {
            assert_eq!(data_dir, "/var/lib/coordinode");
            assert!(config_path.is_none());
        }
        _ => panic!("expected Compact command"),
    }
}

#[test]
fn compact_defaults_data_dir() {
    let cmd = parse_args_from(&args("coordinode compact"));
    match cmd {
        Command::Compact {
            data_dir,
            config_path,
        } => {
            assert_eq!(data_dir, "./data");
            assert!(config_path.is_none());
        }
        _ => panic!("expected Compact command"),
    }
}

#[test]
fn compact_accepts_config_path() {
    let cmd = parse_args_from(&args(
        "coordinode compact --config /etc/coordinode/coordinode.conf",
    ));
    match cmd {
        Command::Compact { config_path, .. } => {
            assert_eq!(
                config_path.as_deref(),
                Some("/etc/coordinode/coordinode.conf")
            );
        }
        _ => panic!("expected Compact command"),
    }
}

#[test]
fn backup_accepts_config_path() {
    let cmd = parse_args_from(&args(
        "coordinode backup --output /tmp/b.bin --config /etc/coordinode/coordinode.conf",
    ));
    match cmd {
        Command::Backup { config_path, .. } => {
            assert_eq!(
                config_path.as_deref(),
                Some("/etc/coordinode/coordinode.conf")
            );
        }
        _ => panic!("expected Backup command"),
    }
}

#[test]
fn restore_accepts_config_path() {
    let cmd = parse_args_from(&args(
        "coordinode restore --input /tmp/b.bin --config /etc/coordinode/coordinode.conf",
    ));
    match cmd {
        Command::Restore { config_path, .. } => {
            assert_eq!(
                config_path.as_deref(),
                Some("/etc/coordinode/coordinode.conf")
            );
        }
        _ => panic!("expected Restore command"),
    }
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
