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

    // Guard against un-initialised proto submodule (release-plz temp worktrees,
    // shallow clones without --recurse-submodules, CI without submodule init, etc.).
    let out_dir = std::env::var("OUT_DIR")?;
    let fallback_dir = std::path::Path::new(&manifest_dir).join("proto_gen");
    println!("cargo:rerun-if-env-changed={UPDATE_PROTO_GEN}");

    let sentinel = proto_root_path.join("coordinode/v1/query/cypher.proto");
    if !sentinel.exists() {
        // Copy pre-generated files (committed in proto_gen/) to OUT_DIR so that
        // the `include!()` macros in proto.rs compile without a live proto submodule.
        // The `proto_gen_matches_the_proto_submodule` test keeps them current.
        for entry in std::fs::read_dir(&fallback_dir)? {
            let entry = entry?;
            let dest = std::path::Path::new(&out_dir).join(entry.file_name());
            std::fs::copy(entry.path(), dest)?;
        }
        return Ok(());
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

    // Refresh the committed fallback copy on request, so a proto change is
    // carried into proto_gen/ by the build itself rather than by hand.
    if std::env::var_os(UPDATE_PROTO_GEN).is_some() {
        for entry in std::fs::read_dir(&fallback_dir)? {
            std::fs::remove_file(entry?.path())?;
        }
        for entry in std::fs::read_dir(&out_dir)? {
            let entry = entry?;
            if entry.path().extension().is_some_and(|e| e == "rs") {
                std::fs::copy(entry.path(), fallback_dir.join(entry.file_name()))?;
            }
        }
    }

    Ok(())
}

/// Set to any value to copy the freshly generated bindings into `proto_gen/`.
const UPDATE_PROTO_GEN: &str = "COORDINODE_UPDATE_PROTO_GEN";
