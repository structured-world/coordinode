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
    let proto_root =
        canonicalize_for_protoc(&std::path::Path::new(&manifest_dir).join("../../proto"));

    let proto_root_str = proto_root.display().to_string();

    // Regenerate whenever any `.proto` under the submodule changes. Without
    // this, cargo never re-runs build.rs after the submodule is updated (the
    // proto tree lives outside the crate), so the OUT_DIR bindings silently go
    // stale: the service compiles against a wire contract that no longer
    // matches the one in the tree.
    println!("cargo:rerun-if-changed={proto_root_str}/coordinode");

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

    // Every service in ONE call, with both stubs.
    //
    // prost writes a file per proto PACKAGE, and these protos import each
    // other (a session runs Cypher, so session.proto pulls cypher.proto in).
    // Compiling them in separate calls therefore has a later call regenerate a
    // package an earlier one already wrote, silently dropping whatever the
    // earlier call had that this one does not reach. Splitting by which stubs
    // a service needs is the tempting arrangement and the broken one.
    //
    // Both stubs everywhere, then: the server stubs are what nodes serve, and
    // the client stubs are how they talk to each other. Cypher's client is the
    // load-bearing one, used to forward a write to whichever node leads, so a
    // client never has to know which that is. Unused clients cost compile
    // time and nothing else.
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[
                format!("{proto_root_str}/coordinode/v1/admin/cluster.proto"),
                format!("{proto_root_str}/coordinode/v1/query/cypher.proto"),
                format!("{proto_root_str}/coordinode/v1/query/vector.proto"),
                format!("{proto_root_str}/coordinode/v1/query/text.proto"),
                format!("{proto_root_str}/coordinode/v1/graph/graph.proto"),
                format!("{proto_root_str}/coordinode/v1/graph/schema.proto"),
                format!("{proto_root_str}/coordinode/v1/graph/blob.proto"),
                format!("{proto_root_str}/coordinode/v1/session/session.proto"),
                format!("{proto_root_str}/coordinode/v1/health/health.proto"),
                format!("{proto_root_str}/coordinode/v1/replication/cdc.proto"),
            ],
            &includes,
        )?;

    Ok(())
}
