---
description: "Run CoordiNode in Docker: the published image, port mapping for gRPC, REST and metrics, volume layout for persistent data, and a Compose file to start from."
---

# Docker Installation

The fastest way to run CoordiNode — no build tools required.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) 24+
- [Docker Compose](https://docs.docker.com/compose/install/) v2

## Get the Compose File

Clone the repository to get `docker-compose.yml` and the bundled examples:

```bash
git clone https://github.com/structured-world/coordinode.git
cd coordinode
```

The pre-built image is published at `ghcr.io/structured-world/coordinode`. If you prefer to use it directly without cloning, create a minimal `docker-compose.yml`:

```yaml
services:
  coordinode:
    image: ghcr.io/structured-world/coordinode:latest
    container_name: coordinode
    ports:
      - "7080:7080"
      - "7081:7081"
      - "7084:7084"
    volumes:
      - coordinode-data:/data
    environment:
      - COORDINODE_LOG_FORMAT=json
    restart: unless-stopped

volumes:
  coordinode-data:
    driver: local
```

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
{"status":"ok"}
```

## Health Check

The image declares a Docker `HEALTHCHECK` that runs the binary's own probe,
`/coordinode healthcheck`: it asks `/ready` on the ops port and exits 0 on
`200`. `/ready` answers `503` until the server accepts requests and again from
the moment it starts shutting down, so an orchestrator routes nothing to a node
that is starting or draining. The probe needs no shell or HTTP client in the
image. `docker compose ps` shows the container as `healthy` once it is ready.

Consensus that stops on a fatal error (a failed log write or sync) shuts the
server down with a non-zero exit status rather than leaving a container that
can no longer commit. Docker does not restart an `unhealthy` container, but it
does restart one that exits: give the service a restart policy
(`restart: unless-stopped`) and the node comes back and replays its durable
state, which holds every acknowledged write.

The check probes the ops address the server uses: the built-in default, or the
one set in a config file passed with `--config`, or `--ops-addr`, which wins
over both. A server that runs from a config file is checked with the same file:

```yaml
    command: ["serve", "--config", "/etc/coordinode/coordinode.conf"]
    healthcheck:
      test: ["CMD", "/coordinode", "healthcheck", "--config", "/etc/coordinode/coordinode.conf"]
```

A server started with a different `--ops-addr` on the command line needs the
same address in the check:

```yaml
    healthcheck:
      test: ["CMD", "/coordinode", "healthcheck", "--ops-addr", "127.0.0.1:9184"]
```

A port the server cannot bind (gRPC, REST or ops) stops it at startup, so the
check never gets its answer from another process holding the port.

`/health` is the liveness endpoint: it answers `200` for as long as the process
runs, ready or not.

## Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| `7080` | gRPC | Native API — high-throughput clients |
| `7081` | HTTP/REST | JSON API, transcoded to gRPC inside the same process |
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
