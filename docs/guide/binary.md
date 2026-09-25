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

Requirements: [Rust](https://rustup.rs/) 1.80+, protoc 3.21+

```bash
git clone https://github.com/structured-world/coordinode.git
cd coordinode
cargo build --release -p coordinode-server
```

The binary is at `target/release/coordinode`.

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

## Upgrade a Store Written by 0.6

A data directory written by 0.6.x does not open with this release. Each
partition now records which committed writes it physically holds, and 0.6
kept no such record, so recovery could not tell which journalled writes to
replay without risking a lost or doubled one. The server refuses the
directory with an error saying so and leaves it byte for byte as it was.

Move the data across with a dump, with the server stopped:

```bash
# With the 0.6 binary: dump the old directory.
coordinode backup --data /var/lib/coordinode --output coordinode-0.6.snap --format raft-snapshot

# With this release: restore into a new directory, then serve from it.
coordinode restore --data /var/lib/coordinode-new --input coordinode-0.6.snap --format raft-snapshot
coordinode serve --data /var/lib/coordinode-new
```

For a cluster, restore the dump into one node, start it, and add the other
nodes with empty data directories, as for a new cluster; they receive the
data from it.

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
