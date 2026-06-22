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
    let proto_root_path = std::path::Path::new(&manifest_dir).join("../../proto");

    // The generated proto bindings are a pure build artifact: regenerated from
    // the proto submodule on every build, never committed. Require the submodule
    // so a fresh checkout fails loudly here instead of silently compiling against
    // stale, drifted bindings. A sentinel file only exists when it is checked out.
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

    // Regenerate whenever any `.proto` under the submodule changes. Without this,
    // cargo never re-runs build.rs after the submodule is updated (the proto tree
    // lives outside the crate), so the OUT_DIR bindings silently go stale.
    println!("cargo:rerun-if-changed={proto_root_str}/coordinode");

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

    // Client-only build: build_server(false), build_client(true)
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(true)
        .compile_protos(
            &[
                // Graph types referenced by query protos
                format!("{proto_root_str}/coordinode/v1/graph/graph.proto"),
                format!("{proto_root_str}/coordinode/v1/graph/schema.proto"),
                format!("{proto_root_str}/coordinode/v1/graph/blob.proto"),
                // Common types
                format!("{proto_root_str}/coordinode/v1/common/types.proto"),
                // Replication types (ReadPreference, ReadConcern) — imported by cypher.proto
                format!("{proto_root_str}/coordinode/v1/replication/consistency.proto"),
                // Query services
                format!("{proto_root_str}/coordinode/v1/query/cypher.proto"),
                format!("{proto_root_str}/coordinode/v1/query/vector.proto"),
                format!("{proto_root_str}/coordinode/v1/query/text.proto"),
            ],
            &includes,
        )?;

    Ok(())
}
