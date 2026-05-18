//! CoordiNode S3-compatible gateway.
//!
//! Standalone binary exposing BlobStore via a subset of the S3 API.
//!
//! CE supports: GetObject, PutObject, DeleteObject, ListBuckets, ListObjectsV2.
//! EE adds: multipart upload, presigned URLs, bucket versioning, storage class mapping.
//!
//! Usage:
//!   coordinode-s3 [--addr ADDR] [--data DIR]

mod api;
mod mapping;

use std::net::SocketAddr;
use std::sync::Arc;

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, Tier};
use tracing::info;

fn main() {
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("failed to create runtime: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = rt.block_on(run()) {
        eprintln!("coordinode-s3 error: {e}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    // Simple arg parsing
    if args.iter().any(|a| a == "--version" || a == "version") {
        println!("coordinode-s3 v{}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let addr_str = find_flag(&args, "--addr").unwrap_or_else(|| "[::]:7081".to_string());
    let data_dir = find_flag(&args, "--data").unwrap_or_else(|| "./data".to_string());

    let addr: SocketAddr = addr_str.parse()?;

    let config = coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "default",
            &data_dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        ),
    ]);
    let engine = Arc::new(coordinode_storage::engine::core::StorageEngine::open(
        &config,
    )?);

    info!(
        addr = %addr,
        data_dir = %data_dir,
        "coordinode-s3 v{} starting",
        env!("CARGO_PKG_VERSION")
    );

    api::serve(addr, engine).await?;
    Ok(())
}

fn find_flag(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
