# Docker Installation

The fastest way to run CoordiNode — no build tools required.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) 24+
- [Docker Compose](https://docs.docker.com/compose/install/) v2

## Start CoordiNode

```bash
docker compose up -d
```

CoordiNode starts in under 5 seconds. Verify it is healthy:

```bash
curl http://localhost:7084/health
```

Expected response:

```json
{"status": "serving"}
```

## Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| `7080` | gRPC | Native API — high-throughput clients |
| `7081` | HTTP/REST | JSON API via gRPC transcoding |
| `7084` | HTTP | `/health`, `/ready`, Prometheus `/metrics` |

## Data Persistence

By default, `docker-compose.yml` mounts a named volume `coordinode-data` for the data directory. Data survives container restarts.

To start fresh:

```bash
docker compose down -v   # removes the volume
docker compose up -d
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level: `error`, `warn`, `info`, `debug`, `trace` |
| `COORDINODE_DATA` | `/data` | Data directory inside the container |

Set them in `docker-compose.yml` or via `--env`:

```bash
RUST_LOG=debug docker compose up
```

## Run Seed Data (Optional)

Try the bundled quickstart example:

```bash
./examples/quickstart/seed.sh
```

This inserts a small knowledge graph (4 concept nodes + 4 document nodes with 384-dimensional embeddings) and verifies connectivity.

## Next Step

Run a hybrid query combining graph traversal + vector similarity:

```bash
curl -s -X POST http://localhost:7081/v1/query/cypher \
  -H "Content-Type: application/json" \
  -d @examples/quickstart/hybrid-query.json | python3 -m json.tool
```

See [Quick Start](../QUICKSTART) for the full walkthrough.
