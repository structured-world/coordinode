# Configuration

CoordiNode is configured through command-line flags. The packaged Linux service
maps a small environment file to those flags, so on a package install you edit
one file and restart. This page is the full reference for every flag, the
environment variables the binary reads directly, the packaged config file, and
the operating-system limits that matter for a production deployment.

## How configuration flows

There is no separate config-file format parsed by the binary. Settings reach the
server in two ways:

1. **Command-line flags** to `coordinode serve` (the source of truth).
2. **A handful of environment variables** read directly by the process
   (logging only).

On a package install the systemd unit reads `/etc/coordinode/coordinode.conf`
(an `EnvironmentFile`) and expands those variables into the `ExecStart` flags.
So editing the conf file is equivalent to changing the flags:

```
/etc/coordinode/coordinode.conf   (KEY=value)
        |  EnvironmentFile=
        v
coordinode.service ExecStart:
  coordinode serve --addr ${COORDINODE_ADDR} --rest-addr ${COORDINODE_REST_ADDR} \
                   --ops-addr ${COORDINODE_OPS_ADDR} --data ${COORDINODE_DATA}
```

## `serve` flags

`coordinode serve` is the default subcommand (running `coordinode` with no
arguments is the same as `coordinode serve`).

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `full` | Operational mode. The CE binary supports only `full`. `compute` and `storage` require the Enterprise binary and are rejected with a clear message. |
| `--node-id` | `1` | Numeric node ID, unique within a cluster. Single-node deployments always use `1`. Any value above `1` requires `--peers`. |
| `--addr` | `[::]:7080` | gRPC listen address. Carries the native API and inter-node Raft RPCs. |
| `--advertise-addr` | same as `--addr` | Address other nodes use to reach this node. Set it when `--addr` binds `0.0.0.0` or `[::]` so peers learn the real host/IP. |
| `--rest-addr` | `[::]:7081` | REST/JSON listen address. Present only when built with the `rest-proxy` feature. |
| `--ops-addr` | `[::]:7084` | Operational HTTP address for `/metrics`, `/health`, `/ready`. Pass port `0` to let the OS assign an ephemeral port (handy in tests). |
| `--data` | `./data` | Data directory. Holds the storage engine state. |
| `--peers` | (none) | Comma-separated peer addresses, for example `node2:7080,node3:7080`. Presence enables Raft consensus. |
| `--nofile` | hard limit | Open-file-descriptor soft limit requested at startup. Unset raises the soft limit to the hard limit. Unix only. |
| `--max-connections` | (unbounded) | Maximum in-flight requests per client connection (gRPC concurrency limit). |
| `--max-request-size-mb` | `16` | Maximum decoded request message size, in MiB. Guards against unbounded-allocation requests. |
| `--request-timeout-secs` | (none) | Per-request server-side timeout, in seconds. |
| `--http2-keepalive-secs` | (none) | HTTP/2 keepalive ping interval, in seconds. Detects half-open connections behind a load balancer. |
| `--cache-size-mb` | engine default | Block cache size, in MiB. The read path serves hot blocks from this cache before touching disk. |
| `--write-buffer-mb` | engine default | Write buffer (memtable) size, in MiB. Larger buffers flush less often at the cost of memory. |
| `--retention-window-secs` | `604800` (7 days) | MVCC time-travel / `AS OF TIMESTAMP` horizon, in seconds. The GC watermark is held back to at least `now - this`, so history within the window stays queryable and CDC / backup consumers keep their checkpoint readable. |
| `--registry-heartbeat-ms` | `100` | Consumer-registry heartbeat coalescing window, in ms. Buffered consumer heartbeats flush as one Raft proposal per window; a larger window trades freshness for fewer proposals on busy shards. |
| `--registry-eviction-ms` | `1000` | Consumer-registry TTL-eviction sweep interval, in ms. How often expired registrations are swept and the retention floor is refreshed against the wall clock. |

Single-node start:

```bash
coordinode serve --data /var/lib/coordinode
```

Three-node cluster (run on each host, adjusting `--node-id`, `--addr`, and the
peer list):

```bash
coordinode serve --node-id 1 --addr 0.0.0.0:7080 \
  --advertise-addr node1:7080 --peers node2:7080,node3:7080 \
  --data /var/lib/coordinode
```

## Environment variables

The binary reads these directly, regardless of how it was launched:

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level filter. Standard env-filter syntax, for example `RUST_LOG=warn,coordinode_query=debug`. |
| `COORDINODE_LOG_FORMAT` | `text` | Log output format. `text` for human-readable, `json` for structured logs to ship to a collector. |

The `COORDINODE_ADDR`, `COORDINODE_REST_ADDR`, `COORDINODE_OPS_ADDR`, and
`COORDINODE_DATA` variables in the packaged conf file are **not** read by the
binary. The systemd unit expands them into `serve` flags. If you launch the
binary yourself (Docker, a script), pass the flags directly.

## Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 7080 | gRPC | Native API and inter-node Raft |
| 7081 | HTTP/REST | JSON transcoding (with the `rest-proxy` feature) |
| 7084 | HTTP | `/metrics`, `/health`, `/ready` |

Bind the ops address to loopback in production and scrape it through a reverse
proxy, or keep it internal to the host's monitoring network.

## Packaged config file (RPM / DEB)

The `.rpm` and `.deb` packages install `/etc/coordinode/coordinode.conf` and
mark it as a configuration file, so your edits survive package upgrades
(`%config(noreplace)` on RPM, a dpkg conffile on DEB). Default contents:

```bash
# gRPC listen address (native API + inter-node Raft).
COORDINODE_ADDR=0.0.0.0:7080

# REST/JSON listen address (transcoded from gRPC by the embedded proxy).
COORDINODE_REST_ADDR=0.0.0.0:7081

# Operational HTTP listen address: Prometheus /metrics, /health, /ready.
COORDINODE_OPS_ADDR=127.0.0.1:7084

# Data directory. Owned by the coordinode system user.
COORDINODE_DATA=/var/lib/coordinode/data

# Optional: structured JSON logs (uncomment to enable).
# COORDINODE_LOG_FORMAT=json

# Optional: log level filter (RUST_LOG syntax).
# RUST_LOG=info

# Optional: extra serve flags (resource / network / storage tuning), appended
# to the command line and word-split. See the flag table above.
# COORDINODE_EXTRA_ARGS=--nofile 262144 --cache-size-mb 4096 --max-connections 1024
```

The `COORDINODE_EXTRA_ARGS` line is how you set the resource, network, and
storage flags on a package install: list any `serve` flags there and they are
appended to the launch command. Apply changes:

```bash
sudo systemctl restart coordinode
```

## Operating-system limits

### Open file descriptors

The storage engine keeps many files open at once (one or more per on-disk table,
plus write-ahead logs and network sockets). A production node should run with a
high file-descriptor limit. The packaged systemd unit sets it:

```ini
[Service]
LimitNOFILE=65536
```

To raise it further, do not edit the shipped unit (an upgrade overwrites it).
Use a drop-in:

```bash
sudo systemctl edit coordinode
```

```ini
[Service]
LimitNOFILE=262144
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart coordinode
```

The server also raises its own soft limit at startup. With no `--nofile` flag it
lifts the soft limit to the hard limit; pass `--nofile N` to request a specific
value (clamped to the hard limit). This works regardless of how the process is
launched, so a Docker or manually-started node gets the same treatment as the
systemd service. The hard limit is still a ceiling the process cannot exceed on
its own:

- **Docker / Podman:** raise the hard limit with `--ulimit nofile=65536:65536`
  on `docker run`, or the `ulimits:` block in `docker-compose.yml`. The server
  then raises its soft limit to that hard limit automatically.
- **Shell / supervisor:** the hard limit comes from `/etc/security/limits.conf`
  (or the supervisor config); the server raises the soft limit toward it.

If the limit is too low you will see "too many open files" errors under load or
during compaction.

### Memory

The engine sizes its caches from available memory and does not require a fixed
heap setting. For predictable behaviour under memory pressure, run one node per
host or pin the process with `MemoryMax=` in a systemd drop-in.

## Network limits

These guard the gRPC server against resource exhaustion and detect dead peers.
All are off (unbounded / disabled) unless set.

- `--max-connections N` caps the number of in-flight requests a single client
  connection may have outstanding. On a stream-multiplexed transport this is the
  practical equivalent of a connection cap.
- `--max-request-size-mb N` rejects any request whose decoded body exceeds the
  cap (default 16 MiB), so a malformed or hostile request cannot force an
  unbounded allocation.
- `--request-timeout-secs N` aborts a request that runs longer than the limit.
- `--http2-keepalive-secs N` sends keepalive pings so the server reclaims
  connections that died silently (common behind a load balancer or NAT).

## Storage tuning

The storage engine picks safe defaults; override them only with a measured
reason.

- `--cache-size-mb N` sets the block cache. A larger cache keeps more hot blocks
  in memory and cuts read latency, at the cost of resident memory. Size it to
  the working set, not the whole dataset.
- `--write-buffer-mb N` sets the in-memory write buffer (memtable). Larger
  buffers flush to disk less often, trading memory for fewer, larger flushes.

## Retention and time-travel

MVCC keeps superseded versions so `AS OF TIMESTAMP` queries can read the past
and CDC / backup consumers can resume from a checkpoint. The garbage collector
reclaims versions older than the retention window; the window is therefore the
horizon for both time-travel reads and lagging-consumer recovery.

- `--retention-window-secs N` sets that horizon, in seconds (default seven
  days). Shorter windows reclaim space sooner but shrink the time-travel range
  and the grace period a slow consumer has before its checkpoint is collected.
  A registered consumer lagging beyond the window holds the floor back for
  itself rather than losing data silently.
- `--registry-heartbeat-ms N` and `--registry-eviction-ms N` tune the
  consumer-retention registry's background service: how often buffered consumer
  heartbeats are flushed as a coalesced proposal, and how often expired
  registrations are swept. The defaults (100 ms / 1000 ms) suit most
  deployments; raise the heartbeat window on shards with many consumers to cut
  proposal volume.

## Other subcommands

The same binary runs maintenance and cluster operations. Each shares the
`--data` flag pointing at the database directory.

| Command | Required flags | Notable options |
|---------|----------------|-----------------|
| `coordinode backup` | `--output FILE` | `--format json\|cypher\|binary\|snapshot`, `--namespace NS`, `--since SEQNO` (incremental snapshot) |
| `coordinode restore` | `--input FILE` | `--format ...` (plus import-only `apoc-json`, `apoc-cypher`, `hetio-json`), `--namespace NS`, `--only-labels L1,L2`, `--force` |
| `coordinode checkpoint` | `--output DIR` | Hard-linked physical checkpoint; restore by pointing `serve --data` at it |
| `coordinode verify` | (none) | `--deep` for full-page checksum verification |
| `coordinode version` | (none) | Print version and exit |
| `coordinode admin node join` | `--node ADDR --id ID --addr ADDR` | `--pre-seeded`, `--follow` |
| `coordinode admin node decommission` | `--node ADDR --id ID` | `--pruning`, `--force`, `--skip-confirmation` |

Run `coordinode` with no recognized subcommand to print the full usage summary.
