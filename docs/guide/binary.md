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
| `--http-addr` | `[::]:7081` | REST/HTTP listen address |
| `--ops-addr` | `[::]:7084` | Health + metrics endpoint |
| `--data` | `./data` | Data directory |

## Verify

```bash
curl http://localhost:7084/health
# → {"status":"serving"}
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
