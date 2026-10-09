---
description: "Install and run CoordiNode as a single native binary without Docker, including the serve, backup, restore, verify, compact and checkpoint subcommands and their options."
---

# Binary Installation

Run CoordiNode directly on your machine — no Docker required.

## Download a Release Binary

Pre-built binaries for Linux and macOS are available on the [GitHub Releases](https://github.com/structured-world/coordinode/releases) page.

```bash
# Linux x86_64
curl -L https://github.com/structured-world/coordinode/releases/latest/download/coordinode-linux-x86_64.tar.gz \
  | tar -xz
sudo mv coordinode /usr/local/bin/

# macOS (Apple Silicon)
curl -L https://github.com/structured-world/coordinode/releases/latest/download/coordinode-macos-arm64.tar.gz \
  | tar -xz
sudo mv coordinode /usr/local/bin/
```

Verify:

```bash
coordinode --version
```

## Build from Source

Requirements: [rustup](https://rustup.rs/) with the toolchain pinned in
`rust-toolchain.toml` (currently Rust 1.98.1), and protoc 3.21+.

```bash
git clone --recurse-submodules https://github.com/structured-world/coordinode.git
cd coordinode
cargo build --release -p coordinode-server
```

The binary is at `target/release/coordinode`.

### Production optimization profile

Build from the workspace root with `--release`. The checked-in
`[profile.release]` applies to the server and its Rust dependencies:

| Setting | Value | Purpose |
| --- | --- | --- |
| `opt-level` | `3` | Optimize execution speed (O3). |
| `lto` | `"fat"` | Optimize across crate boundaries. |
| `codegen-units` | `1` | Compile each crate as one code-generation unit. |
| `panic` | `"abort"` | Abort on panic instead of unwinding. |
| `strip` | `true` | Strip symbols from the shipped binary. |

The command above includes the server's default features. For an embedded
application, set its own workspace release profile: Cargo does not inherit
the dependency repository's profile. `cargo build` without `--release`
selects the development profile and is not a production performance build.

O2 is an alternative to measure, not the current production default:

```bash
CARGO_PROFILE_RELEASE_OPT_LEVEL=2 cargo build --release -p coordinode-server
```

Compare the same workload, features, hardware, throughput and tail latency
before selecting an override. O3 does not guarantee a faster result than O2;
see the [Cargo profile reference](https://doc.rust-lang.org/cargo/reference/profiles.html).
Environment variables, Cargo configuration and `RUSTFLAGS` can override
build settings, so record them with benchmark results. Do not ship a binary
built with `target-cpu=native` to machines whose instruction sets differ;
portable builds preserve runtime CPU detection.

## Start the Server

```bash
coordinode serve --data /var/lib/coordinode
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--addr` | `[::]:7080` | gRPC listen address |
| `--ops-addr` | `[::]:7084` | Health + metrics endpoint |
| `--data` | `./data` | Data directory |
| `--peers` | (none) | Peer addresses for cluster mode |

See [Configuration](/guide/configuration) for the full flag and environment reference, the packaged config file, and the open-file-descriptor limit.

The binary exposes **gRPC only** on port 7080. For REST/JSON, run [structured-proxy](https://github.com/structured-world/structured-proxy) in front of it (see `docker-compose.yml` for a reference setup).

## Verify

```bash
curl http://localhost:7084/health
# → {"status":"serving"}
```

## Compact After a Bulk Import

After loading a large dataset, run an offline compaction to collapse the
accumulated merge operands (adjacency lists, counters) into single values.
This keeps traversal reads fast. The server must be stopped first.

```bash
coordinode compact --data /var/lib/coordinode
```

## systemd Service (Linux)

```ini
[Unit]
Description=CoordiNode graph database
After=network.target

[Service]
ExecStart=/usr/local/bin/coordinode serve --data /var/lib/coordinode
Restart=on-failure
User=coordinode

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now coordinode
```

## Next Step

See [Quick Start](../QUICKSTART) to seed data and run your first hybrid query.
