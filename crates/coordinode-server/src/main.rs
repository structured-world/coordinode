//! CoordiNode server binary.
//!
//! Usage:
//!   coordinode serve [--addr ADDR] [--data DIR]
//!   coordinode version
//!   coordinode verify [--data DIR] [--deep]
//!
//! Argument parsing lives in `coordinode_server::cli`, every behaviour in
//! `coordinode_server::run`. This binary is the CE entry point; an enterprise
//! distribution links the same library and extends it rather than forking it.

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    coordinode_server::run(coordinode_server::cli::parse_args()).await
}
