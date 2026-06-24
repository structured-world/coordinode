# Configuration

CoordiNode resolves its configuration from a single in-code gate that layers
three sources, lowest to highest precedence:

1. **Built-in defaults** — every setting has a safe default.
2. **The YAML config file** — `coordinode serve --config <path>`, if given.
3. **Command-line flags** — every setting also has a `serve` flag.

So the command line overrides the config file, which overrides the defaults.
Each setting appears once in each layer, and every config-file key has a
matching CLI flag of the same meaning. This page is the full reference for every
setting, the environment variables the binary reads directly, the packaged
config file, and the operating-system limits that matter in production.

## How configuration flows

Pass a config file and override individual values on the command line:

```bash
# File supplies the baseline; --node-id on the CLI wins over the file's value.
coordinode serve --config /etc/coordinode/coordinode.conf --node-id 2
```

A missing `--config` runs on built-in defaults plus any flags. A `--config` path
that cannot be read or contains an unknown key / malformed value is a startup
error — the server fails loud rather than silently falling back.

On a package install the systemd unit points `--config` at the shipped file:

```
/etc/coordinode/coordinode.conf   (YAML)
        |  --config
        v
coordinode.service ExecStart:
  coordinode serve --config /etc/coordinode/coordinode.conf
```

To override one value without editing the file, add a CLI flag through a systemd
drop-in (`systemctl edit coordinode`); the command line beats the file.

## `serve` flags

`coordinode serve` is the default subcommand (running `coordinode` with no
arguments is the same as `coordinode serve`).

Every flag below (except `--config` itself) has a YAML config-file key with the
same name in `snake_case` (for example `--max-request-size-mb` is
`max_request_size_mb`, `--addr` is `grpc_addr`, `--data` is `data_dir`). The
defaults in the table are the built-in defaults that apply when neither the
config file nor a flag sets the value.

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (none) | Path to a YAML config file. Absent = built-in defaults. A present-but-unreadable or malformed file is a startup error. Has no config-file key (it names the file). |
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
| `--interactive-txn-idle-timeout-secs` | `30` | Idle timeout for an interactive transaction (a `BeginTransaction` left open without `CommitTransaction`/`RollbackTransaction`), in seconds. An open transaction pins an MVCC snapshot and buffers writes in memory; one idle this long is auto-rolled-back to release them. |
| `--interactive-txn-max-bytes` | `268435456` (256 MiB) | Max buffered (uncommitted) bytes per interactive transaction. A transaction whose accumulated writes exceed this is aborted, capping the leader memory a client can hold without committing. |
| `--wire-compression-level` | `3` | Inter-node gRPC transport zstd compression level for wire traffic (C-zstd numbering: `1`..=`22` trade speed for ratio). The default `3` is zstd's standard speed/ratio default and gives roughly a 9x reduction on Raft batches; raise it on a bandwidth-constrained link (for example a geo replica) where bandwidth costs more than CPU. Independent of the on-disk storage codec. |
| `--tls-cert` / `--tls-key` | (none) | PEM paths to the node's TLS certificate and private key. Set both to serve inter-node + client gRPC over TLS (pure-Rust crypto, no C FFI); unset = plaintext (single-host dev). |
| `--tls-ca` | (none) | PEM path to the CA that verifies peer certificates — trusted by clients to verify the server, and (with `--tls-require-client-auth`) by the server to verify connecting nodes. |
| `--tls-require-client-auth` | `false` | Require and verify a client certificate (mutual TLS) on incoming connections. Needs `--tls-ca`. |
| `--no-scrub` | (scrub on) | Disable the background integrity scrub. Each node otherwise periodically verifies every on-disk block's checksum on its own local storage. |
| `--scrub-interval-secs` | `604800` | Seconds between background scrub cycles (default 7 days). |
| `--scrub-throttle-ms` | `50` | Pause between SST scans during a scrub so it yields I/O to production; `0` runs at full speed. |

### TLS trust and self-signed certificates

There is no "skip verification" switch: whenever TLS is on, the peer certificate
is always verified against `--tls-ca`, which is the trust root. For a cluster,
issue every node certificate from one internal CA and point `--tls-ca` at that
CA on every node (`--tls-require-client-auth` then gives node-to-node mutual
TLS). For a single self-signed certificate, point `--tls-ca` at the certificate
itself: a self-signed cert is its own trust anchor. If you want no certificate
management at all, leave TLS off (the default) and serve plaintext on a trusted
network, rather than running TLS without verification.

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

Logging is controlled through the environment (the standard Rust ecosystem
knobs), independent of the config file. The binary reads these directly,
regardless of how it was launched:

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level filter. Standard env-filter syntax, for example `RUST_LOG=warn,coordinode_query=debug`. |
| `COORDINODE_LOG_FORMAT` | `text` | Log output format. `text` for human-readable, `json` for structured logs to ship to a collector. |

On a package install set these via a systemd drop-in (`systemctl edit
coordinode`):

```ini
[Service]
Environment=RUST_LOG=warn,coordinode_query=debug
Environment=COORDINODE_LOG_FORMAT=json
```

## Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 7080 | gRPC | Native API and inter-node Raft |
| 7081 | HTTP/REST | JSON transcoding (with the `rest-proxy` feature) |
| 7084 | HTTP | `/metrics`, `/health`, `/ready` |

Bind the ops address to loopback in production and scrape it through a reverse
proxy, or keep it internal to the host's monitoring network.

## Packaged config file (RPM / DEB)

The `.rpm` and `.deb` packages install `/etc/coordinode/coordinode.conf` (the
YAML config file the unit passes to `--config`) and mark it as a configuration
file, so your edits survive package upgrades (`%config(noreplace)` on RPM, a
dpkg conffile on DEB). It ships with every key documented; uncommented keys set
the default value, commented keys show it. Excerpt:

```yaml
# Operational mode (CE supports only "full").
mode: full

# Numeric node id. Single-node = 1; a value above 1 needs a non-empty peers list.
node_id: 1

# gRPC listen address (native API + inter-node Raft).
grpc_addr: "0.0.0.0:7080"

# REST/JSON listen address (transcoded from gRPC by the embedded proxy).
rest_addr: "0.0.0.0:7081"

# Operational HTTP listen address: Prometheus /metrics, /health, /ready.
ops_addr: "127.0.0.1:7084"

# Data directory. Owned by the coordinode system user.
data_dir: /var/lib/coordinode/data

# Storage topology. Empty = a single endpoint at data_dir. Declare endpoints to
# spread storage across disks (see "Storage topology" above for every key):
# storage:
#   endpoints:
#     - { id: nvme-hot, path: /mnt/nvme0, media: nvme, durability: durable, tier: hot }
#     - { id: hdd-cold, path: /mnt/hdd0,  media: hdd,  durability: degraded, tier: cold }
storage:
  endpoints: []

# Cluster peers (empty = standalone). For HA list the other members:
# peers:
#   - "node2.internal:7080"
#   - "node3.internal:7080"
peers: []

# Resource / network / storage tuning (commented keys show the default):
# nofile: 262144
# max_connections: 1024
max_request_size_mb: 16
# cache_size_mb: 4096
# write_buffer_mb: 256

# Interactive-transaction limits.
interactive_txn_idle_timeout_secs: 30
interactive_txn_max_bytes: 268435456

# Inter-node gRPC transport zstd level for wire traffic.
wire_compression_level: 3

# Inter-node + client gRPC TLS (PEM paths). Unset = plaintext.
# tls_cert: /etc/coordinode/tls/node.crt
# tls_key: /etc/coordinode/tls/node.key
# tls_ca: /etc/coordinode/tls/ca.crt
# tls_require_client_auth: false

# Background integrity scrub (per-node, verifies on-disk block checksums).
scrub_enabled: true
# scrub_interval_secs: 604800
# scrub_throttle_ms: 50
```

Set any tunable directly in this file. To override one value without editing the
file, add the matching CLI flag through a drop-in (`systemctl edit coordinode`)
since the command line beats the file. Apply changes:

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

## Storage topology

By default a node uses a single storage endpoint: a durable HDD warm-tier
endpoint rooted at `data_dir`. That covers the common single-disk deployment
without any extra configuration; `--data` (or the `data_dir` config key) is all
you set.

To spread storage across several disks — an NVMe cache in front of an SSD tier
in front of an HDD JBOD, for example — declare each mount point as an *endpoint*
under `storage.endpoints` in the config file. A topology is a list of structured
records, so it lives in the YAML config file only; there is no command-line flag
for the list. The per-LSM-level placement (hot levels on fast media, cold levels
on slow), cascade eviction between endpoints, and WAL / oplog routing are all
driven from this list.

Each endpoint has these keys:

| Key | Required | Values | Meaning |
|-----|----------|--------|---------|
| `id` | yes | unique string | Endpoint name, used in metrics, logs, and placement rules. |
| `path` | yes | directory | Exclusive mount-point directory. Two endpoints must not share a path. |
| `media` | yes | `hdd` `ssd` `nvme` `ram` | Physical media kind (metadata; does not set durability). |
| `durability` | yes | `durable` `degraded` `volatile` | `durable` = hardware-redundant (RAID); `degraded` = single drive; `volatile` = lost on restart. Drives the per-block ECC `auto` rule and cluster redundancy invariants. |
| `tier` | yes | `memory` `hot_cache` `hot` `warm` `cold` | Placement preference (fastest first). Hot LSM levels prefer hotter tiers. |
| `capacity_bytes` | no | integer | Physical capacity. `0` or omitted = untracked. |
| `hard_limit_bytes` | no | integer | The placement engine never writes past this. `0` or omitted = no limit. Must be `<= capacity_bytes` when both are set. |
| `page_ecc` | no | `auto` `force_on` `force_off` | Per-block Reed-Solomon ECC policy. `auto` (default) turns ECC **on** for `degraded` endpoints and **off** for `durable` / `volatile`. See the note below. |
| `hard_limit_strategy` | no | `reject` `cascade_evict` | What to do when a write would exceed `hard_limit_bytes`. `reject` (default) fails the write; `cascade_evict` demotes data to a cooler endpoint and retries. |

Example: NVMe hot tier plus an HDD cold tier on a single drive (so ECC is forced
on to catch media read errors):

```yaml
data_dir: /var/lib/coordinode/data
storage:
  endpoints:
    - id: nvme-hot
      path: /mnt/nvme0/coordinode
      media: nvme
      durability: durable
      tier: hot
    - id: hdd-cold
      path: /mnt/hdd0/coordinode
      media: hdd
      durability: degraded
      tier: cold
      page_ecc: force_on
      capacity_bytes: 16000000000000
```

Precedence: when `storage.endpoints` is set it overrides `data_dir` for storage
placement (`data_dir` still anchors the consensus log and CDC). Passing `--data`
on the command line wins over the file — it names a single directory, so it
collapses any file topology back to one endpoint at that path. A typo in an
endpoint key is a startup error, like any other unknown config key.

**Page ECC is a build-time feature.** The Reed-Solomon page-ECC codec is compiled
in only when the binary is built with `--features page_ecc` (off by default). On a
binary built without it, `page_ecc: force_on` (or a `degraded` endpoint under the
`auto` rule) is accepted but has no on-disk effect, and the server logs a warning
at startup. Build with the feature to make the policy take effect.

**The io_uring storage backend is a build-time feature (Linux only).** Building
with `--features io-uring` opens the storage engine on a single shared io_uring
ring instead of the default `StdFs` (synchronous `pread`/`pwrite`) backend, for
higher-throughput I/O on Linux 5.6+. It is off by default and is a no-op on
non-Linux targets. If the running kernel lacks io_uring, the engine logs a
warning and falls back to `StdFs` at startup, so a binary built with the feature
is always safe to run. There is no runtime config knob; the backend is selected
at build time.

### Maintenance commands on a multi-endpoint node

The offline maintenance commands (`backup`, `restore`, `verify`, `checkpoint`,
`compact`) must open the database with the same topology the server uses, or
they would look for data at the wrong path. Pass them the same config file:

```bash
coordinode compact --config /etc/coordinode/coordinode.conf
coordinode verify  --config /etc/coordinode/coordinode.conf --deep
coordinode backup  --config /etc/coordinode/coordinode.conf --output dump.bin
```

Without `--config` they operate on a single endpoint at `--data` (default
`./data`) — correct only for single-disk nodes. With `--config`, `--data` is
ignored and the topology comes from the file.

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

The same binary runs maintenance and cluster operations. Each opens the database
at a directory given by `--data` (default `./data`); every command that opens
the storage engine (`backup`, `restore`, `verify`, `checkpoint`, `compact`) also
accepts `--config FILE` to open a multi-endpoint node at its configured paths
(see [Storage topology](#storage-topology)). With `--config`, `--data` is
ignored.

| Command | Required flags | Notable options |
|---------|----------------|-----------------|
| `coordinode backup` | `--output FILE` | `--config FILE` (multi-endpoint); `--format json\|cypher\|binary\|snapshot`, `--namespace NS`, `--since SEQNO` (incremental snapshot) |
| `coordinode restore` | `--input FILE` | `--config FILE` (multi-endpoint); `--format ...` (plus import-only `apoc-json`, `apoc-cypher`, `hetio-json`), `--namespace NS`, `--only-labels L1,L2`, `--force` |
| `coordinode checkpoint` | `--output DIR` | `--config FILE` (multi-endpoint); hard-linked physical checkpoint; restore by pointing `serve --data` at it |
| `coordinode compact` | (none) | `--config FILE` (multi-endpoint); offline major-compaction folding merge operands |
| `coordinode verify` | (none) | `--config FILE` (multi-endpoint); `--deep` for full-page checksum verification |
| `coordinode version` | (none) | Print version and exit |
| `coordinode admin node join` | `--node ADDR --id ID --addr ADDR` | `--pre-seeded`, `--follow` |
| `coordinode admin node decommission` | `--node ADDR --id ID` | `--pruning`, `--force`, `--skip-confirmation` |

Run `coordinode` with no recognized subcommand to print the full usage summary.
