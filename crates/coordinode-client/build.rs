fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
    let proto_root_path = std::path::Path::new(&manifest_dir).join("../../proto");

    // Guard against un-initialised proto submodule (release-plz temp worktrees,
    // shallow clones without --recurse-submodules, CI without submodule init, etc.).
    // Use a sentinel file that only exists when the submodule is checked out.
    let sentinel = proto_root_path.join("coordinode/v1/query/cypher.proto");
    if !sentinel.exists() {
        // Copy pre-generated files (committed in proto_gen/) to OUT_DIR so that
        // the `include!()` macros in lib.rs compile without a live proto submodule.
        // These files must be regenerated when proto changes (run `cargo build -p
        // coordinode-client` and copy target/debug/build/coordinode-client-*/out/
        // coordinode.v1.*.rs to crates/coordinode-client/proto_gen/).
        let out_dir = std::env::var("OUT_DIR")?;
        let fallback_dir = std::path::Path::new(&manifest_dir).join("proto_gen");
        for entry in std::fs::read_dir(&fallback_dir)? {
            let entry = entry?;
            let dest = std::path::Path::new(&out_dir).join(entry.file_name());
            std::fs::copy(entry.path(), dest)?;
        }
        return Ok(());
    }

    let proto_root = proto_root_path
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from("../../proto"));
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
