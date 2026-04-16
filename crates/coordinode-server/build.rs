/// Resolve a path to an absolute form that protoc can consume on all platforms.
///
/// `std::path::Path::canonicalize()` on Windows returns a UNC extended-length
/// path with a `\\?\` prefix (e.g. `\\?\D:\a\...`). protoc does not understand
/// this prefix and fails with "Invalid file name pattern". Strip it so protoc
/// receives a plain absolute Windows path (`D:\a\...`).
fn canonicalize_for_protoc(path: &std::path::Path) -> std::path::PathBuf {
    match path.canonicalize() {
        Ok(p) => {
            let s = p.to_string_lossy();
            if let Some(stripped) = s.strip_prefix(r"\\?\") {
                std::path::PathBuf::from(stripped)
            } else {
                p
            }
        }
        Err(_) => path.to_path_buf(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
    // Proto files are in the `proto/` submodule at the workspace root.
    // From this crate (crates/coordinode-server/) that's ../../proto.
    let proto_root = canonicalize_for_protoc(
        &std::path::Path::new(&manifest_dir).join("../../proto"),
    );

    let proto_root_str = proto_root.display().to_string();

    // Include paths: our proto root + system protobuf includes (for google/protobuf/*.proto).
    // On macOS: /opt/homebrew/include or /usr/local/include
    // On Linux/Docker: /usr/include
    let mut includes = vec![proto_root_str.clone()];
    for candidate in [
        "/usr/include",
        "/usr/local/include",
        "/opt/homebrew/include",
    ] {
        let p = std::path::Path::new(candidate).join("google/protobuf/descriptor.proto");
        if p.exists() {
            includes.push(candidate.to_string());
            break;
        }
    }

    // Compile admin/cluster.proto with both server and client stubs.
    // Server stub: ClusterServiceServer (registered in main.rs).
    // Client stub: ClusterServiceClient (used by `coordinode admin node join` CLI subcommand).
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[format!(
                "{proto_root_str}/coordinode/v1/admin/cluster.proto"
            )],
            &includes,
        )?;

    // Compile remaining service protos (server-only stubs).
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(
            &[
                format!("{proto_root_str}/coordinode/v1/graph/graph.proto"),
                format!("{proto_root_str}/coordinode/v1/graph/schema.proto"),
                format!("{proto_root_str}/coordinode/v1/graph/blob.proto"),
                format!("{proto_root_str}/coordinode/v1/query/cypher.proto"),
                format!("{proto_root_str}/coordinode/v1/query/vector.proto"),
                format!("{proto_root_str}/coordinode/v1/query/text.proto"),
                format!("{proto_root_str}/coordinode/v1/health/health.proto"),
                format!("{proto_root_str}/coordinode/v1/replication/cdc.proto"),
            ],
            &includes,
        )?;

    Ok(())
}
