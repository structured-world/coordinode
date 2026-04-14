fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
    // Proto files are in the `proto/` submodule at the workspace root.
    // From this crate (crates/coordinode-client/) that's ../../proto.
    let proto_root = std::path::Path::new(&manifest_dir)
        .join("../../proto")
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
