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
    // From this crate (tests/integration/) that's ../../proto.
    let proto_root_path = std::path::Path::new(&manifest_dir).join("../../proto");
    // Regenerate whenever anything under the proto tree changes.
    println!("cargo:rerun-if-changed={}", proto_root_path.display());

    // The bindings are a build artifact, generated from the proto submodule
    // into OUT_DIR on every build and never committed. A checkout without the
    // submodule fails here, loudly, rather than compiling stale bindings.
    let sentinel = proto_root_path.join("coordinode/v1/query/cypher.proto");
    if !sentinel.exists() {
        return Err(format!(
            "proto submodule is not checked out (missing {}). \
             Run `git submodule update --init --recursive`.",
            sentinel.display()
        )
        .into());
    }

    let proto_root = canonicalize_for_protoc(&proto_root_path);
    let proto_root_str = proto_root.display().to_string();

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

    // Build clients only (no server side needed for integration tests).
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(true)
        .compile_protos(
            &[
                format!("{proto_root_str}/coordinode/v1/graph/schema.proto"),
                format!("{proto_root_str}/coordinode/v1/query/cypher.proto"),
                format!("{proto_root_str}/coordinode/v1/session/session.proto"),
                format!("{proto_root_str}/coordinode/v1/admin/cluster.proto"),
            ],
            &includes,
        )?;

    Ok(())
}
