fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
    // Proto files are in the `proto/` submodule at the workspace root.
    // From this crate (tests/integration/) that's ../../proto.
    let proto_root_path = std::path::Path::new(&manifest_dir).join("../../proto");

    let sentinel = proto_root_path.join("coordinode/v1/query/cypher.proto");
    if !sentinel.exists() {
        eprintln!(
            "cargo:warning=Proto submodule not initialised — integration tests will not compile"
        );
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

    // Build clients only (no server side needed for integration tests).
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(true)
        .compile_protos(
            &[
                format!("{proto_root_str}/coordinode/v1/graph/schema.proto"),
                format!("{proto_root_str}/coordinode/v1/query/cypher.proto"),
            ],
            &includes,
        )?;

    Ok(())
}
