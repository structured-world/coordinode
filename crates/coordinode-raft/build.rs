fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
    // Proto files are in the `proto/` submodule at the workspace root.
    // From this crate (crates/coordinode-raft/) that's ../../proto.
    let proto_root = std::path::Path::new(&manifest_dir)
        .join("../../proto")
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from("../../proto"));

    let proto_root_str = proto_root.display().to_string();
    let raft_proto = format!("{proto_root_str}/coordinode/v1/replication/raft.proto");

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&[raft_proto], &[proto_root_str])?;

    Ok(())
}
