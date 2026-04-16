/// Resolve a path to an absolute form that protoc can consume on all platforms.
///
/// `std::path::Path::canonicalize()` on Windows returns a UNC extended-length
/// path with a `\\?\` prefix (e.g. `\\?\D:\a\...`). protoc does not understand
/// this prefix and fails with "Invalid file name pattern". Strip it so protoc
/// receives a plain absolute Windows path (`D:\a\...`).
fn canonicalize_for_protoc(path: &std::path::Path) -> std::path::PathBuf {
    match path.canonicalize() {
        Ok(p) => {
            // `to_string_lossy` is safe here: proto paths are ASCII.
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
    // From this crate (crates/coordinode-raft/) that's ../../proto.
    let proto_root = canonicalize_for_protoc(
        &std::path::Path::new(&manifest_dir).join("../../proto"),
    );

    let proto_root_str = proto_root.display().to_string();
    let raft_proto = format!("{proto_root_str}/coordinode/v1/replication/raft.proto");

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&[raft_proto], &[proto_root_str])?;

    Ok(())
}
