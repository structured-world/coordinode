---
description: "Every CoordiNode setting: command-line flags, the YAML configuration file and the values that can be changed on a running database, each with its default and whether it needs a restart."
---

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

The CLI carries only bootstrap-critical settings (bind addresses, identity, data
dir, peers, the TLS identity, the startup fd limit); every fine tunable is a
config-file key with no flag (see "Config-file-only settings" below). Each flag
below (except `--config` itself) has a YAML config-file key with the same name in
`snake_case` (for example `--ops-addr` is `ops_addr`, `--addr` is `grpc_addr`,
`--data` is `data_dir`). The defaults in the table are the built-in defaults that
apply when neither the config file nor a flag sets the value.

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (none) | Path to a YAML config file. Absent = built-in defaults. A present-but-unreadable or malformed file is a startup error. Has no config-file key (it names the file). |
| `--mode` | `full` | Operational mode. The CE binary supports only `full`. `compute` and `storage` require the Enterprise binary and are rejected with a clear message. |
| `--node-id` | `1` | Numeric node ID, unique within a cluster. Single-node deployments always use `1`. Any value above `1` requires `--peers`. |
| `--addr` | `[::]:7080` | gRPC listen address. Carries the native API and inter-node Raft RPCs. A port already in use fails the start before storage is opened; so does a busy `--rest-addr` or `--ops-addr`. |
| `--advertise-addr` | same as `--addr` | Address other nodes use to reach this node. Set it when `--addr` binds `0.0.0.0` or `[::]` so peers learn the real host/IP. |
| `--rest-addr` | `[::]:7081` | REST/JSON listen address. Present only when built with the `rest-proxy` feature. A port already in use fails the start. Pass port `0` for an ephemeral port. |
| `--ops-addr` | `[::]:7084` | Operational HTTP address for `/metrics`, `/health` (200 while the process runs), `/ready` (200 while the gRPC server serves; 503 while starting and from the shutdown signal on) and `POST /reload` (reads a cluster member's node identity files again, see [Node identity](#node-identity)). Consensus that stops on a fatal error, such as a failed log write or sync, shuts the process down with a non-zero exit status: after a failed sync the unsynced pages may already be gone, so the node does not carry on; its supervisor restarts it and recovery replays the durable state, which holds every acknowledged write. Run the server under a supervisor that restarts it (`restart: unless-stopped` in Compose, `Restart=on-failure` in systemd). A port already in use fails the start, so a health check on this port can never be answered by another process. Pass port `0` to let the OS assign an ephemeral port (handy in tests). |
| `--pg-addr` | (disabled) | PostgreSQL wire-protocol listen address (for example `127.0.0.1:7085`). Unset = the Postgres frontend is off. Serves SQL over the Postgres Simple Query protocol with trust authentication (no password), so bind it to a trusted interface only. |
| `--data` | `./data` | Data directory. Holds the storage engine state. |
| `--peers` | (none) | Comma-separated peer addresses, for example `node2:7080,node3:7080`. Presence enables Raft consensus, and then the node needs a `node_identity` (see [Node identity](#node-identity)). |
| `--nofile` | hard limit | Open-file-descriptor soft limit requested at startup. Unset raises the soft limit to the hard limit. Unix only. |
| `--tls-cert` / `--tls-key` | (none) | PEM paths to a standalone server's TLS certificate and private key. Set both to serve client gRPC over TLS (pure-Rust crypto, no C FFI); unset = plaintext (single-host dev). Refused with `--peers`: a cluster member's port presents its `node_identity` certificate to clients too. |
| `--tls-ca` | (none) | PEM path to the CA that issues client certificates. The server verifies a client certificate against it; with `--peers`, a client may connect presenting none unless `--tls-require-client-auth` is set. Nodes are never verified against it: they prove their node identity. |
| `--tls-require-client-auth` | `false` | Require and verify a client certificate (mutual TLS) on incoming client connections, and refuse plain text. Needs `--tls-ca`. |

### Config-file-only settings (no CLI flag)

Fine tunables that are not needed to start the process have **no CLI flag** — the
command line is reserved for bootstrap-critical settings (its length is OS-bounded
by `ARG_MAX`). These keys live in the YAML config file only. Defaults apply when
the key is unset.

| Config key | Default | Mutability | Description |
|------------|---------|------------|-------------|
| `max_connections` | (unbounded) | restart | Maximum in-flight requests per client connection (gRPC concurrency limit). |
| `max_request_size_mb` | `16` | restart | Maximum decoded request message size, in MiB. Guards against unbounded-allocation requests. |
| `request_timeout_secs` | (none) | restart | Per-request server-side timeout, in seconds. |
| `http2_keepalive_secs` | (none) | restart | HTTP/2 keepalive ping interval, in seconds. Detects half-open connections behind a load balancer. |
| `cache_size_mb` | `64` | restart | Block cache size, in MiB. The read path serves hot blocks from this cache before touching disk. Size it to the blocks reads touch often: a working set larger than the cache is read and decompressed again on every access. |
| `write_buffer_mb` | `64` | restart | Write buffer (memtable) size, in MiB. Larger buffers flush less often at the cost of memory. |
| `retention_window_secs` | `300` (5 minutes) | restart | MVCC time-travel / `AS OF TIMESTAMP` horizon, in seconds, enforced by the storage engine itself (embedded databases honour it too via `StorageConfig`). The GC watermark is held back to at least `now - this`, so history within the window stays queryable; a live snapshot pin can hold it back further, never less. A read older than the horizon is refused with `OUT_OF_RANGE` / `OUTSIDE_RETENTION`. Storage held by the window scales with the updates inside it: every version of a key written within the window is kept, plus the newest one below it. |
| `max_invariant_claims` | `100000` | restart | Ceiling on the invariant claims held by all in-flight write attempts on this node, together. A claim is what a mutation states its result depends on (an endpoint it needs to keep existing, a bound it decided, a pair it observed); a commit writing an edge type that declares a relationship cardinality constraint also holds one claim per pair of that type it changes, so a single transaction changing more pairs than this ceiling is refused. The guard holds one set per attempt until the attempt ends, and refuses a claim beyond this ceiling rather than letting the table grow with load. A refused attempt is answered with `ABORTED` / `INVARIANT_REFUSED` and retries cheaply. Raise it only for workloads that legitimately hold many conditions at once (large scans inside write transactions); reaching it otherwise points at attempts that never end. |
| `max_commits_in_flight` | `10000` | restart | Ceiling on the commits admitted and not yet applied on this node. A commit registers the keys it will write before it validates them and holds that registration until its writes are local state, so the table holds one entry per commit in flight: bounded by write concurrency, not by data size. At the ceiling a commit is refused as retryable backpressure (`RESOURCE_EXHAUSTED` / `WRITE_BACKPRESSURE`) rather than the table growing without limit. Reaching it means commits are not finishing (a stalled replication wait, a member that stopped acknowledging), not that the node is merely busy. |
| `snapshot_wait_ms` | `5` | live (`StorageEngine::set_snapshot_wait_ms`) | How long a read waits for commits that are still landing before it is answered from a view that stops behind them. A snapshot must not cover a commit that has not applied, or the reader sees neither the write nor any sign of it; waiting keeps the view fresh for the microseconds a commit needs to apply, stepping behind is instant but hands back a view older than the reader's own last write, which then conflicts with itself. Raise it to favour freshness under heavy write load, lower it to favour read latency. Zero is the step-behind-only behaviour, measured at 59% false conflicts between writers that shared no keys. |
| `min_free_bytes` | `1073741824` (1 GiB) | live (`SpaceGuard::set_reserve`) | Free disk space below which every write is refused before it reaches the disk, on any filesystem holding a durable data directory. The engine reads free space from the filesystem (everything on it counts, not only the database's files) at most every 250 ms. Below the reserve the node keeps serving reads; writes are answered `RESOURCE_EXHAUSTED` / `STORAGE_FULL` (PostgreSQL `53100 disk_full`); a follower takes no new log entries (heartbeats still pass, so it keeps following its leader) and catches up once space is freed; snapshots and periodic checkpoints wait. The node's own recovery records (index use guards, its consensus vote and log bounds) are still written down to half the reserve; the lower half is kept for the flushes that retire them from their journal. A write that reached a full disk would instead fail inside an fsync of the consensus log, which stops the node until it restarts. `coordinode_storage_free_bytes` and `coordinode_storage_writes_paused` report the state. |
| `resume_free_bytes` | `2147483648` (2 GiB) | live (`SpaceGuard::set_reserve`) | Free disk space every durable data directory needs before refused writes are admitted again; taken as at least `min_free_bytes`. The gap keeps a write that frees a little space from flipping writes on and off. |
| `membership_change_timeout_secs` | `30` | live (`RaftNode::set_membership_settle_timeout`) | How long a membership change (`admin node join`, the promotion that follows it, `admin node decommission`) waits for the previous change to commit before it is refused. Status shows a new member as soon as its change is proposed, before the change commits, and the cluster takes one change at a time, so a command issued the moment status shows the previous result waits here instead of failing. A change commits in one replication round, so running out of this means the cluster has lost the quorum to commit; the command is refused naming the change still in progress. |
| `join_readiness_lag_entries` | `1000` | live (`RaftNode::set_join_readiness_lag`) | How many log entries a joining member may still lack when it is promoted from learner to voter. Small enough that the rest replicates within a few heartbeats once it votes; `0` promotes only a member that holds everything. A member that has not answered at all is never promoted, whatever this is set to. |
| `join_timeout_secs` | `1800` | live, for joins started afterwards (`RaftNode::set_join_timeout`) | How long `admin node join` may take to catch the new member up before the join fails and reports it. Zero is refused. |
| `raft_snapshot_entries` | `10000` | restart | Entries applied since the last Raft snapshot that trigger the next. A snapshot lets the log before it be dropped: taking one flushes every partition's memtables so the applied entries are on disk, and copies, links and serializes nothing. A member far behind gets the store itself, its files pinned in place and streamed while it is sent, so a busy node snapshots every `entries / write rate` seconds: raise it to flush less often, at the cost of a longer log to keep and to replay after a restart. Zero is refused. |
| `raft_snapshot_log_bytes` | `268435456` (256 MiB) | restart | Bytes the Raft log's segments grow by since the last snapshot that trigger the next, so a few large entries compact the log as a count of small ones would. Sized every second. Zero is refused. |
| `raft_snapshot_min_interval_secs` | `60` | restart | Shortest time between two Raft snapshots the entry count or the log size asks for, however fast the log grows. Time alone never takes a snapshot: a node whose log stays below both thresholds takes none. Zero is refused. |
| `routed_commands_max` | `512` | restart (`MemberOptions::routing` in embedded mode) | Commands other members route to this one (writes and transactions their clients send, which run here as the leader) that run at once, from all of them together. A command past it is refused before it runs, and the member that routed it returns its client an error saying the command did not run, so it is safe to retry. The default matches the number of threads routed commands run on. Zero is refused. |
| `routed_session_commands_max` | `256` | restart (`MemberOptions::routing` in embedded mode) | Commands one member routes here that run at once, so a single busy member leaves room for the others. Refused past it as above. It also bounds the replies a member may leave unread: past `routed_session_commands_max × (routed_frames_ahead + 3)` the member's session is ended with `RESOURCE_EXHAUSTED`, and what still ran for it is stopped, as when the session is lost. Zero is refused. |
| `routed_frames_ahead` | `4` | restart (`MemberOptions::routing` in embedded mode) | Batches of a routed statement's rows sent ahead of the member that routed it taking them. A member that reads slowly holds the statement back rather than its rows piling up here; raise it for large results across links with a long round trip. Zero is refused. |
| `routed_row_batch_bytes` | `1048576` (1 MiB) | restart (`MemberOptions::routing` in embedded mode) | Encoded size a batch of a routed statement's rows grows to before it is sent; a single larger row is sent in a batch of its own. With `routed_frames_ahead` it bounds what one routed answer holds in memory. Zero is refused. |
| `election_profile` | `rack` | restart (`MemberOptions::timing` in embedded mode) | Election and heartbeat timing of the group. `rack`, for servers in one rack or data centre, sends a heartbeat every 150 ms and starts an election after 300 to 600 ms without one, so a failover takes under a second. `personal_fleet`, for members on laptops, phones and home links that sleep, roam and lose packets for seconds, sends one every 2 s and waits 20 to 40 s, so a short outage does not depose the leader. Give every member of a group the same profile. A member cut off from its leader asks the others whether they would vote before it raises the term (pre-vote), so its return never deposes a leader the rest still follow. |
| `election_heartbeat_ms` | the profile's | restart | Replaces the profile's heartbeat interval, in ms. A start whose timing would let a healthy group keep electing (a heartbeat not well below the shortest election timeout, or an empty election window) is refused. Zero is refused. |
| `election_timeout_min_ms` | the profile's | restart | Replaces the profile's shortest election timeout, in ms. Zero is refused. |
| `election_timeout_max_ms` | the profile's | restart | Replaces the profile's longest election timeout, in ms. Zero is refused. |
| `election_preference` | `ordinary` | restart (`MemberOptions::preference` in embedded mode) | When this member stands for election. `ordinary` campaigns once its election timeout expires; `last` waits past every ordinary member's timeout (two to three times the longest), so it leads only when no ordinary member can, and once it leads it hands leadership to an ordinary member as soon as one has caught up. For a phone or a member behind a slow link. |
| `planner_stats_ttl_secs` | `60` | restart (`Database::set_stats_ttl` in embedded mode) | How long the query planner reuses its storage statistics (node counts per label, edge fan-out) before reading them again. Shorter keeps estimates closer to fresh writes at the cost of a counter read and a bounded adjacency sample on each refresh. The statistics steer the choice of plan, never its result. When they cannot be read because a counter or an adjacency list is damaged, the failure is logged as an error naming the key and remembered for the same time, and queries plan with defaults until the next refresh. |
| `index_feed_capacity` | `16384` | restart | Committed writes each text and vector index worker holds until it has folded them into its index. Searches never wait for a worker: the nodes those writes touched are read from the store and evaluated exactly beside the index. Past this many the queue drops, the worker rebuilds its indexes from the store, and until it has, a search reads the whole label from the store (correct, slower). Raise it for write bursts larger than the workers keep up with; each held write costs its written keys in memory. |
| `slow_query_ms` | `100` | restart | A Cypher statement that runs at least this long, in milliseconds, is logged at info level under the target `coordinode::slow_query` with its kind (read, write, failed), duration, rows returned and text (the first 2 KiB); parameter values are not logged. `0` logs every statement. Silence it with the log filter `coordinode::slow_query=off`. |
| `vector_build_wait_ms` | `30000` | restart (`Database::set_vector_build_wait` in embedded mode; a client changes it for its own Session with `SET vector_build_wait`) | How long a query waits for a vector index still being built, under the `block` online-during-build policy, before it is refused. It is the default for queries that name no bound: a query's own `/*+ vector_build_wait('5s') */` hint overrides it, so a query that can wait longer, or should not wait at all, says so itself. `0` refuses a building index at once. |
| `query_memory_limit_mb` | `256` | restart (`Database::set_query_memory_limit` live in embedded mode; a client changes it for its own Session with `SET query_memory_limit_mb`, and a unary request for itself with its `query_memory_limit_mb` field) | Memory limit of each statement, in MiB: what its operators, index access paths and buffered results may hold at once. Memory is reserved before it is used, so a statement that would need more fails with `RESOURCE_EXHAUSTED`, reason `EXCEEDS_MEMORY_BUDGET` (metadata `requested_bytes`, `used_bytes`, `limit_bytes`), returns no partial result and writes nothing; it is never answered by a slower path that holds less. The administrative ceiling is 4096: a value of `0` or past it is refused at start, and a session or request naming one is refused with `INVALID_ARGUMENT`. A statement also stops at the gRPC deadline of its call (`DEADLINE_EXCEEDED`) and when cancelled (`CANCELLED`). |
| `vector_retired_bytes_budget` | `268435456` (256 MiB) | restart (`Database::set_vector_retired_bytes_budget` live in embedded mode) | Bytes of replaced neighbour lists each vector index lets wait for reclamation. Inserts, updates and removals publish a new list in place of an old one while searches run; a search keeps every list replaced after it began until it finishes, so a slow search holds that memory. Past the budget, a write holds off until running searches release it, and then goes ahead: writes are slowed, never dropped. Per `{label, property}` the server exports the held memory (`coordinode_vector_index_retired_bytes`, `coordinode_vector_index_retired_lists`), the age of the oldest running operation (`coordinode_vector_index_oldest_operation_seconds`), the writes held off and for how long (`coordinode_vector_index_admission_waits_total`, `coordinode_vector_index_admission_wait_microseconds_total`), publications recomputed after losing to a concurrent writer (`coordinode_vector_index_lost_cas_total`), and slot reuse (`coordinode_vector_index_retired_nodes`, `coordinode_vector_index_free_slots`). |
| `index_build_max_running` | `2` | live (`Database::set_index_build_config`; the server applies it at startup) | Index builds this node fills at once. Builds run on the engine, not on the connection of the statement that created the index; the others wait for a seat with their builds accepted, and inspection shows them waiting. Raising it seats waiting builds at once; lowering it lets running builds finish and seats no more until fewer run. Zero is refused. |
| `index_build_older_transactions_wait_secs` | `60` | live, for builds that start afterwards (`Database::set_index_build_config`) | How long an index build waits for the transactions opened before its index existed to end. Such a transaction decided what to maintain without the new index, so the build reads no data until it ends. A build that outwaits this fails: a new index is withdrawn, a rebuilt one kept failed, and the failure names the open transactions. |
| `index_build_unique_read_limit` | `100000` | live (`Database::set_index_build_config`; the server applies it at startup) | While a unique index is being built, the stored nodes its build has not reached yet have no entries, so a missing entry does not prove a value free. A write that takes a value of such an index has its commit read those nodes, as they stand when the commit is decided, and refuses the write when another node holds the value. This bounds the stored node rows one such commit reads. Past it the write is refused as unresolved, never as a duplicate: the value may be free, and the same write succeeds once the build is done. A larger limit admits more writes during a build at the cost of a longer commit. The same proof, over every node of the label, is made for a unique index whose entries were found disagreeing with their records, until a check verifies or rebuilds it. |
| `index_check_interval_secs` | `86400` | live (`Database::set_index_build_config`; the server applies it at startup) | How often every B-tree index is checked against the records it is derived from, even when no read or write found it wrong. A check reads every record of the label for the entries its values give it and every entry for a record holding its value, repairs each disagreement only while the record and the entry are still as it read them, and marks the index verified once a complete pass finds nothing. A missing entry gives no sign on any read (the lookup just does not find the record), so only a check finds it. A disagreement a read or a write meets starts a check at once, whatever this says; meanwhile lookups in that index answer from the records and its unique values are proved free from them. The member that takes writes runs the checks. `0` checks only on reports and requests (`Database::check_index`). |
| `index_check_repair_batch` | `512` | live (`Database::set_index_build_config`; the server applies it at startup) | The most disagreements one repair commit of an index check writes. Zero is refused. |
| `index_check_memory_mb` | `64` | live (`Database::set_index_build_config`; the server applies it at startup) | Memory, in MiB, one pass of index checks holds the expected entries in. A pass reads every record of the store once for all the indexes being checked, at one snapshot, and compares each index's entries with what the records expect, both sorted, so it never looks an entry or a record up one by one. Past this memory it writes sorted runs to unnamed scratch files under `index-check/` in the data directory, which vanish with the process. A check whose scratch files cannot be written fails with that cause and concludes nothing: it is never verified by a partial comparison, and an index found wrong keeps answering from the records. A check interrupted by a restart reads everything again. Zero is refused. |
| `index_check_max_repairs` | `10000` | live (`Database::set_index_build_config`; the server applies it at startup) | The most disagreements one check repairs entry by entry. Past it, or when a third pass still finds disagreements, the damage is taken as wide and the index is rebuilt into a fresh generation (`REINDEX`), which writers maintain at once and lookups use once it is built. Two records holding one value of a unique index are reported and left as they are: no entry can represent them, and no check changes a record. |
| `index_build_statement_wait_secs` | `60` | live (`Database::set_index_build_config`; the server applies it at startup) | How long a statement or call creating an index or a constraint waits for the index's build, or for the proof of a presence or type constraint or of a label definition over the stored nodes, when it names no bound of its own (a gRPC `CreateConstraint` names one in `wait`; a schema package is awaited separately through `AwaitSchema`). Past it, the statement returns with the work still running: the index `BUILDING`, the constraint `VALIDATING` and enforced on writes, and the `operation`. The work goes on without the statement and is inspected, awaited and cancelled through that operation (`db.indexBuild`, `db.cancelIndexBuild`; gRPC `GetIndexBuild`, `CancelIndexBuild`). `0` returns once the build is admitted. |
| `node_shard` | `0` | restart | The shard whose node rows this engine holds. Node keys carry the shard ahead of the id, and the invariant guard is the one place inside the engine that resolves a node from its id alone, so it needs this to find the row. It must match the shard the statements above run against; the embedded database sets it from its own handle. |
| `registry_heartbeat_ms` | `1000` | restart | Consumer-registry heartbeat coalescing window, in ms. The first heartbeat buffered opens a window, and the window's heartbeats flush as one Raft proposal; a larger window trades freshness for fewer proposals on busy shards. Nothing buffered, nothing runs. |
| `registry_eviction_ms` | `1000` | restart | Shortest gap between two consumer-registry sweeps, in ms. A sweep ends every `BOUNDED` registration whose bound is crossed (liveness, progress lag, retained bytes) and refreshes the retention floor. Progress lag is the age of the oldest committed entry the consumer has not acknowledged; log entries that carry no commit (the entry a leader appends to close a burst of commits, the first entry of its term, a membership change) are not owed, and a consumer with nothing else ahead of it is not lagging. A scrape of `/metrics` shows each live consumer's `registry_consumer_lag_seqno`, `registry_consumer_lag_age_seconds`, `registry_consumer_heartbeat_age_seconds` and `registry_consumer_retained_bytes`. It runs when the earliest bound can be crossed, while a `BOUNDED` consumer is behind, and when a registration changes (on this member or replicated from another); `STRICT` registrations are never ended by it. |
| `cdc_heartbeat_interval_ms` | `10000` | restart | How often a CDC change stream that is waiting (caught up, or on a slow reader) heartbeats its registration, in ms. Delivery does not wait on it: a caught-up stream wakes as soon as this node applies another entry. A subscription that asks for a `BOUNDED` liveness timeout no longer than this is refused, since its own stream could not keep it alive. `0` is refused. |
| `cdc_batch_size` | `256` | restart | Most entries a CDC change stream reads and sends per poll. Larger batches cut per-poll overhead for a reader far behind; smaller ones keep each poll short. `0` is refused. |
| `cdc_buffer_bytes` | `67108864` (64 MiB) | restart | Memory for log entries shared by every CDC change stream of the shard. The log is read once for all streams; an entry stays in this buffer until every stream reading from it has passed it. A stream that falls behind more than this reads the gap from the log itself and rejoins the buffer. Retention of the log on disk is unaffected: it still follows the consumers' acknowledgements. |
| `default_read_concern` | `local` | restart | Read concern of a statement that names none and whose session set none: `local`, `majority`, `linearizable` or `snapshot`. A statement's own setting wins, then its session's, then this. A causal read (one that names an index to wait for) needs `majority` or `snapshot`, so with `local` here every causal read must name its level. |
| `default_read_preference` | `primary` | restart | Which member serves a read that names no preference and whose session set none: `primary`, `primary_preferred`, `secondary`, `secondary_preferred` or `nearest`. A `primary` read that reaches a follower is forwarded to the leader. |
| `default_write_concern` | `{ w: majority, journal: journal, timeout_ms: 0 }` | restart | When a write that names no concern, and whose session set none, is acknowledged. `w` is `majority` or a member count, the leader included (`0` answers at once); `journal` is the state each of those members holds the write's log entry in: `journal` (fsynced), `cache` (handed to the operating system, synced later) or `memory` (in the process, synced later). `cache` and `memory` are honoured for `w: 1` only; any other combination with them fails the start. `timeout_ms` bounds the wait for `w` (`0` = none); a write that times out is not rolled back and may still commit. A causal session refuses writes under anything weaker than `w: majority, journal: journal`, including this default. |
| `interactive_txn_idle_timeout_secs` | `30` | restart | Idle timeout for an interactive transaction (one started and left open without commit/rollback), in seconds. An open transaction pins an MVCC snapshot and buffers writes; one idle this long ends with its timeout outcome. |
| `interactive_txn_max_bytes` | `268435456` (256 MiB) | restart | Max buffered (uncommitted) bytes per interactive transaction. A transaction whose accumulated writes exceed this is aborted, capping leader memory a client can hold without committing. |
| `interactive_txn_timeout_outcome` | `rollback` | restart (a session sets its own with `Configure.timeout_outcome`; embedded: `Database::set_interactive_timeout_outcome` live, `TransactionOptions::timeout_outcome` per transaction) | What an interactive transaction left idle past its timeout becomes, when its session set nothing: `rollback` discards it, `commit` commits what it holds. A transaction that commits on timeout must name its operation reference when it begins, so its client can learn the outcome (see [transactions](./transactions#multi-statement-transactions)); a statement error aborts a transaction for good, so a failed one is never committed. |
| `autocommit_retry_attempts` | `6` | restart | Runs, in all, of a statement executed without `BEGIN` whose commit is refused for contention, before the conflict is returned. A refused run applied nothing, so the server runs it again from a fresh snapshot. At least `1`; `1` returns the first refusal. Statements of an explicit transaction, ones pinned to a named timestamp and cursor pages always run once. |
| `autocommit_retry_backoff_ms` | `2` | restart | Wait before the second run of a refused auto-commit statement, in ms; each later run waits twice as long as the one before. With the defaults the runs wait about 60 ms in all. |
| `operation_admission_window` | `65536` | restart | How many client operations an issuer incarnation enrolled from now on may hold above its retired prefix (see [safe retries](./transactions#retrying-a-write-safely)). A bound write numbered past the window is refused, and runs nothing, until the client acknowledges or retires outcomes below it. It bounds the operation records one client can make the database keep, about 100 bytes each. An incarnation keeps the window it was enrolled with. Zero is refused. Embedded: `Database::set_operation_admission_window`. |
| `peer_connect_timeout_ms` | `5000` | restart | How long a member's dial to a peer may take, the TCP connection and the TLS handshake together, in ms. A dial that runs out fails, and the call that needed it fails with it; the next call dials again. Raise it on distant or lossy links where a handshake takes longer. Zero is refused. |
| `peer_call_timeout_ms` | `30000` | restart | How long one call from a member to a peer (a consensus message, a version handshake, a segment piece, a client's BEGIN, COMMIT or ROLLBACK passed to the member that runs it) may take, in ms, before it fails. A client's statement passed on streams its answer back, so the bound covers only the wait for it to start; the client's own deadline bounds the rest. Zero is refused. |
| `peer_handshake_timeout_ms` | `10000` | restart | How long a connection accepted on a member's gRPC port has to complete its TLS handshake, in ms. A connection that opens one and stalls is dropped at this bound, freeing what it held. Zero is refused. |
| `wire_compression_level` | `3` | restart | Inter-node gRPC transport zstd compression level (C-zstd `1`..=`22`). Default `3` is zstd's standard default (~9x reduction on Raft batches); raise on a bandwidth-constrained link. Independent of the on-disk codec. |
| `scrub_enabled` | `false` | restart | Whether the background integrity scrub runs (each node verifies its own on-disk block checksums). A pass reads every block, so it is off unless enabled. |
| `scrub_interval_secs` | `604800` (7 days) | restart | Seconds between background scrub cycles. |
| `scrub_throttle_ms` | `50` | restart | Pause between SST scans during a scrub so it yields I/O to production; `0` runs at full speed. |
| `maintenance_mb_per_sec` | unset (unthrottled) | live (`StorageEngine::set_maintenance_rate`, bytes per second) | MiB per second the node's background maintenance reads and writes at most: every compaction of every partition and `STORAGE COLUMNAR` table, and the scrub, draw from this one budget, so together they cannot saturate the disk the foreground needs. A budget too low for the write rate lets compaction fall behind until write backpressure engages. Unset or `0` leaves maintenance unthrottled. |
| `maintenance_backoff_ms` | unset (off) | live (`StorageEngine::set_maintenance_backoff`) | Smoothed read latency, in ms, above which the maintenance budget backs off, down to 5% of its rate, until the device answers faster again; it climbs back once the latency is 20% under this. Applies only while `maintenance_mb_per_sec` sets a rate. Unset or `0` keeps the budget at its rate. |
| `compaction_strategy` | `leveled` | restart | How every partition's and `STORAGE COLUMNAR` table's compactions are chosen. `leveled` keeps each level a fixed size ratio over the one above. `planned` plans merges for query throughput over a round of compactions, from the share of point reads and scans each tree has served, and sizes them to the maintenance budget (`maintenance_mb_per_sec`), so set a budget with it. |
| `write_slowdown_mb_per_sec` | unset (off) | live (`StorageEngine::set_write_slowdown_rate`, bytes per second) | MiB per second client writes are held to while a partition is past its slowdown threshold (`backpressure`): each commit waits its turn at this rate, so writes slow down well before the stop threshold refuses them. Unset or `0` lets writes run at full speed until the stop. |
| `vector_locality_depth` | unset (off) | restart | How far the vector truth tier's values may interleave across blob files before a compaction rewrites them in key order, so a scan over the vectors (an index build, a quantizer calibration) reads runs of adjacent values. A relocation costs at most half again what the compaction writes anyway. Unset or `0` never relocates for locality. |
| `columnar_encoded_from_level` | `3` | restart | The first LSM level whose `STORAGE COLUMNAR` column pages are encoded with a light codec chosen per page; the levels above keep plain pages, which a read serves without decoding. `7` or more keeps every level plain. |
| `checkpoint_enabled` | `false` | restart | Whether periodic local checkpoints are taken (the base for WAL-replay repair when no healthy replica is available). A checkpoint hard-links the store's tables, so taking one copies nothing, but it keeps every table compaction replaces after it: on data that is rewritten, each retained checkpoint grows toward a second copy of the store. Off by default; enable it where that space buys repair without a replica (a standalone node), and size `checkpoint_keep` to the disk. |
| `checkpoint_interval_secs` | `3600` (1 hour) | restart | Seconds between periodic checkpoints. A rebuild from the latest checkpoint replays the Raft log from the oldest entry the checkpoint lacks, so the log is kept from there: a longer interval keeps more log on disk. |
| `checkpoint_dir` | `<data_dir>/checkpoints` | restart | Directory checkpoints are written under. |
| `checkpoint_keep` | `3` | restart | Number of recent checkpoints to retain; older ones are pruned. |
| `trigger_max_cascade_depth` | `10` | live (setParameter) | Async AFTER COMMIT cascade-depth cap. A self- or mutually-triggering chain deeper than this is dead-lettered as a cascade overflow rather than looping. Per-trigger `CASCADE_LIMIT n` overrides it. |
| `trigger_default_retry_attempts` | `3` | live (setParameter) | Default total execution attempts for an AFTER COMMIT trigger that declares no `ON ERROR` clause, before its event is dead-lettered into `trigger_failures`. Per-trigger `ON ERROR RETRY n` overrides it. |
| `trigger_default_backoff_ms` | `1000` | live (setParameter) | Default base retry backoff in ms for AFTER COMMIT triggers with no `ON ERROR` clause; per-attempt wait is `backoff * 2^attempt`. Per-trigger `WITH BACKOFF ms` overrides it. |
| `trigger_dispatch_interval_ms` | `1000` | restart | Shortest gap between two passes of the leader's AFTER COMMIT dispatch worker, in ms. The worker wakes on each replicated write and at the earliest scheduled retry; with nothing queued it does not wake at all. |
| `extensions` | empty | restart | Settings belonging to a distribution built on top of this server; see below. |

The three `live` knobs are runtime-tunable on a running database by an owner via
the privileged setParameter surface, without a restart; the config file sets
their startup value.

### Settings for a downstream distribution

`coordinode-server` can be linked as a library and extended, which is how the
Enterprise build is assembled. Anything an extension needs to be told goes
under `extensions`, keyed by the extension's own name:

```yaml
extensions:
  pitr:
    enabled: true
    target: /var/lib/coordinode/pitr
    interval_secs: 3600
```

The base server never reads this table: it passes each key through to whatever
registered under that name, which parses its own shape. That is also why keys
inside `extensions` escape the strict unknown-key check applied to the rest of
the file, so a typo there surfaces from the extension rather than at startup.
Running the stock build, leave the table out.

### TLS trust and self-signed certificates

There is no "skip verification" switch: a client certificate is always verified
against `--tls-ca`, which is the trust root. For a single self-signed server
certificate, clients point their trust at the certificate itself: a self-signed
cert is its own trust anchor. If you want no certificate management on a
standalone server, leave client TLS off (the default) and serve plaintext on a
trusted network, rather than running TLS without verification.

### Node identity

Members of a cluster authenticate each other on every connection, in both
directions, with node identities: X.509-SVIDs, certificates carrying exactly one
SPIFFE ID (`spiffe://trust-domain/path`) as a URI name. A member is bound to its
SPIFFE ID when it joins, and its peers admit it under that identity alone, only
as the member it is bound to and only in the roles it holds. An address says
where to find a member, never who it is. There is no plain-text peer channel and
no setting that turns verification off.

A node with `--peers` needs a `node_identity` section (config file only):

```yaml
node_identity:
  # The node's certificate chain, leaf first, and its key (PEM).
  cert: /etc/coordinode/identity/node.crt
  key: /etc/coordinode/identity/node.key
  # Per SPIFFE trust domain, the CA certificates (PEM) that issue its
  # identities. A peer is trusted only under its own domain's bundle.
  trust_bundles:
    prod.example: /etc/coordinode/identity/bundle.pem
  # Joining nodes only: the deployment they join, as the first member's
  # cluster status (or its startup log line "deployment") reports it, and
  # the members they learn the membership from, each with its identity.
  # deployment: 9f0c2e7a5b1d4c3e8a6f0b2d4e6a8c0e
  # enroll_from:
  #   - addr: node1:7080
  #     identity: spiffe://prod.example/db/node1
```

The SPIFFE path is the operator's choice; nothing about its layout is assumed.
Give each member its own identity: two members proving one identity cannot be
told apart, and the second is refused. The first member of a cluster leaves
`deployment` unset; it records a fresh deployment identity when its group forms
(a standalone server records one on its first start). A node configured with a
deployment its store does not belong to refuses to start and names both.

The gRPC port serves clients and nodes alike and presents the node certificate
to both; give the certificate the DNS names clients verify the server by, as
X.509-SVIDs may carry. A connection is a node's only when its certificate proves
a node identity; every inter-node service (consensus, version handshake, segment
transfer) refuses any other connection before reading what it sent, while the
client services accept clients as `--tls-ca` and `--tls-require-client-auth`
say: in plain text unless client certificates are required. A client
certificate never counts as a node identity, even one from the same CA. An
application may present an X.509-SVID of its own; it is a client as long as
its trust domain is not one of `trust_bundles`. A certificate naming a SPIFFE ID
in one of those trust domains is judged as a node identity only, and refused if
that domain's bundle does not verify it.

Renewed certificates and changed trust bundles take effect without a restart:
write the new files in place and send the process `SIGHUP` (Unix) or `POST
/reload` to its ops port (every platform). The files are read and validated as
a whole first; a set that does not validate (a key that does not match, a
certificate no bundle trusts, another identity than the node has) is refused,
logged, and the last valid set stays in force. Connections made under trust that
was withdrawn, or with a certificate that expired, stop carrying traffic. A new
identity is a new member, not a renewal: a node that left the cluster joins
again under a new identity, its old one stays refused.

Single-node start:

```bash
coordinode serve --data /var/lib/coordinode
```

Three-node cluster (run on each host, adjusting `--node-id`, `--addr`, the peer
list and the identity files in the config file):

```bash
coordinode serve --config /etc/coordinode/coordinode.conf --node-id 1 \
  --addr 0.0.0.0:7080 --advertise-addr node1:7080 \
  --peers node2:7080,node3:7080 --data /var/lib/coordinode
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
| 7084 | HTTP | `/metrics`, `/health`, `/ready`, `/version`, `POST /reload` |
| 7085 | PostgreSQL wire | SQL over the Postgres protocol (opt-in via `--pg-addr`; trust auth) |

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

# PostgreSQL wire-protocol listen address. Unset = off. Trust auth only, so
# bind it to a trusted interface (loopback / internal network).
# pg_addr: "127.0.0.1:7085"

# Data directory. Owned by the coordinode system user.
data_dir: /var/lib/coordinode/data

# Storage topology. Empty = a single endpoint at data_dir. Declare endpoints to
# spread storage across disks (see "Storage topology" above for every key):
# storage:
#   endpoints:
#     - { id: nvme-hot, path: /mnt/nvme0, media: nvme, durability: durable, tier: hot }
#     - { id: hdd-cold, path: /mnt/hdd0,  media: hdd,  durability: degraded, tier: cold }
#   oplog:                  # see "Oplog" above
#     sync_method: full
storage:
  endpoints: []

# Cluster peers (empty = standalone). For HA list the other members:
# peers:
#   - "node2.internal:7080"
#   - "node3.internal:7080"
peers: []
# How long a membership change waits for the previous one to commit.
# membership_change_timeout_secs: 30
# Entries a joining member may lack when it is promoted, and how long a join may take.
# join_readiness_lag_entries: 1000
# join_timeout_secs: 1800
# When a Raft snapshot is taken: entries or log growth (bytes), no sooner
# than the minimum interval (s) after the last.
# raft_snapshot_entries: 10000
# raft_snapshot_log_bytes: 268435456
# raft_snapshot_min_interval_secs: 60
# Commands other members route here: how many run at once (all, one member),
# and how a routed statement's rows go back (batch bytes, batches ahead).
# routed_commands_max: 512
# routed_session_commands_max: 256
# routed_frames_ahead: 4
# routed_row_batch_bytes: 1048576
# How long the planner reuses its storage statistics.
# planner_stats_ttl_secs: 60
# Log statements running at least this long (ms); 0 logs every statement.
# slow_query_ms: 100
# Committed writes a text/vector index worker holds before it rebuilds.
# index_feed_capacity: 16384
# How long a query waits for a vector index still being built (ms).
# vector_build_wait_ms: 30000
# Memory limit of each statement (MiB), at most 4096.
# query_memory_limit_mb: 256
# Replaced-list memory a vector index holds for running searches before
# writers hold off (bytes).
# vector_retired_bytes_budget: 268435456
# Index builds filled at once, and how long one waits for older
# transactions to end (s).
# index_build_max_running: 2
# index_build_older_transactions_wait_secs: 60
# Stored rows one commit reads to prove a value free in a unique index
# still being built.
# index_build_unique_read_limit: 100000
# How long a statement creating an index waits for its build (s).
# index_build_statement_wait_secs: 60
# How often every B-tree index is checked against its records (s; 0 = only
# on reports and requests), repairs per commit, memory a check pass holds
# before it spills to scratch files (MiB), and repairs a check makes before
# it rebuilds instead.
# index_check_interval_secs: 86400
# index_check_repair_batch: 512
# index_check_memory_mb: 64
# index_check_max_repairs: 10000

# Resource / network / storage tuning (commented keys show the default):
# nofile: 262144
# max_connections: 1024
max_request_size_mb: 16
# cache_size_mb: 64
# write_buffer_mb: 64

# Interactive-transaction limits.
interactive_txn_idle_timeout_secs: 30
interactive_txn_max_bytes: 268435456
# What an idle transaction becomes at its timeout: rollback or commit.
interactive_txn_timeout_outcome: rollback

# Re-execution of an auto-commit statement refused at commit.
autocommit_retry_attempts: 6
autocommit_retry_backoff_ms: 2

# Client operations an issuer incarnation may hold above its retired prefix.
operation_admission_window: 65536

# Inter-node gRPC transport zstd level for wire traffic.
wire_compression_level: 3

# Client gRPC TLS of a standalone server (PEM paths). Unset = plaintext.
# tls_cert: /etc/coordinode/tls/server.crt
# tls_key: /etc/coordinode/tls/server.key
# tls_ca: /etc/coordinode/tls/client-ca.crt
# tls_require_client_auth: false

# A cluster member's node identity (required with peers).
# node_identity:
#   cert: /etc/coordinode/identity/node.crt
#   key: /etc/coordinode/identity/node.key
#   trust_bundles:
#     prod.example: /etc/coordinode/identity/bundle.pem
# Bounds of the inter-node channel (ms): dial, call, accepted handshake.
# peer_connect_timeout_ms: 5000
# peer_call_timeout_ms: 30000
# peer_handshake_timeout_ms: 10000

# Background integrity scrub (per-node, verifies on-disk block checksums).
scrub_enabled: false
# scrub_interval_secs: 604800
# scrub_throttle_ms: 50

# One budget for compaction and the scrub (MiB/s, unset = unthrottled), and
# the read latency (ms) above which it backs off.
# maintenance_mb_per_sec: 200
# maintenance_backoff_ms: 20
# How compactions are chosen: leveled (default) or planned.
# compaction_strategy: leveled
# Client writes held to this rate (MiB/s) past the slowdown threshold.
# write_slowdown_mb_per_sec: 64
# Rewrite interleaved vector blob files in key order past this depth.
# vector_locality_depth: 4
# First level whose columnar pages are encoded per page.
# columnar_encoded_from_level: 3

# Periodic local checkpoints (base for WAL-replay repair); off by default.
# checkpoint_enabled: false
# checkpoint_interval_secs: 3600
# checkpoint_dir: /var/lib/coordinode/checkpoints
# checkpoint_keep: 3
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
They are config-file keys (no CLI flag); all are off (unbounded / disabled)
unless set.

- `max_connections` caps the number of in-flight requests a single client
  connection may have outstanding. On a stream-multiplexed transport this is the
  practical equivalent of a connection cap.
- `max_request_size_mb` rejects any request whose decoded body exceeds the cap
  (default 16 MiB), so a malformed or hostile request cannot force an unbounded
  allocation.
- `request_timeout_secs` aborts a request that runs longer than the limit.
- `http2_keepalive_secs` sends keepalive pings so the server reclaims
  connections that died silently (common behind a load balancer or NAT).

## Storage tuning

The storage engine picks safe defaults; override these config-file keys only
with a measured reason.

- `cache_size_mb` sets the block cache. A larger cache keeps more hot blocks in
  memory and cuts read latency, at the cost of resident memory. Size it to the
  working set, not the whole dataset.
- `write_buffer_mb` sets the in-memory write buffer (memtable). Larger buffers
  flush to disk less often, trading memory for fewer, larger flushes.

### Write backpressure

Sustained write bursts can outrun background compaction and flushes. The
engine tracks that on three axes: per partition tree, the L0 run count (a tall
L0 inflates read amplification) and the pending-compaction byte debt; and for
the whole engine, the memtable bytes not yet flushed to tables. It turns them
into a verdict against six thresholds under `storage.backpressure`:

```yaml
storage:
  backpressure:
    l0_slowdown: 40               # default
    l0_stop: 200                  # default
    bytes_slowdown: 68719476736   # 64 GiB, default
    bytes_stop: 274877906944      # 256 GiB, default
    memtable_slowdown: 536870912  # 512 MiB, default
    memtable_stop: 2147483648     # 2 GiB, default
```

The L0 stop level is a deliberately high wall: a DDL burst legitimately
creates dozens of tiny L0 tables between two compaction passes, and the
byte-debt axis carries the real ceiling. The memtable levels bound the memory
writes can take while the flushes fall behind them (a slow disk, a flush that
fails and is retried); the unflushed bytes are exported as
`coordinode_storage_unflushed_memtable_bytes`.

Crossing a *slowdown* level raises that partition's compaction priority to
urgent and is visible in metrics (`coordinode_storage_backpressure_tier`,
`coordinode_storage_compaction_debt_bytes`); writes still proceed at full
rate. Crossing a *stop* level makes commits carrying writes fail with a
retryable `WRITE_BACKPRESSURE` error (gRPC `RESOURCE_EXHAUSTED` with
`RetryInfo`) until compaction or the flushes drain below the threshold; official
drivers back off and retry on it automatically. The server never sleeps
inside the write path on either tier, read-only traffic is unaffected, and
replicated (Raft) entry application is never gated. Changing the thresholds
requires a restart.

### Oplog

The oplog is the Raft log on a cluster member and the retained journal on a
standalone node. A write is acknowledged once its oplog append is durable.
On a cluster member the append is flushed by a thread of its own: consensus
goes on replicating while the disk flushes, and one flush makes durable every
append made while the previous one ran, so under load a flush is shared by a
group of writes instead of paid by each. The settings sit under
`storage.oplog`; every key is optional and changing one requires a restart:

```yaml
storage:
  oplog:
    sync_method: full            # default
    segment_max_bytes: 67108864  # 64 MiB, default
    segment_max_entries: 50000   # default
    retention_secs: 604800       # 7 days, default
    local_journal_compact_entries: 4096  # default, see below
```

`sync_method` chooses how an append is made durable:

| Value | Call | Survives a power loss |
|-------|------|-----------------------|
| `full` (default) | `F_FULLFSYNC` on macOS, `fdatasync` on Linux, `FlushFileBuffers` on Windows: the drive flushes its own write cache. The same calls MongoDB's WiredTiger journal makes. | yes |
| `fsync` | `fsync(2)`. On macOS the data reaches the drive but may stay in its volatile cache, which makes an append much cheaper there; on Linux and Windows it is the same as `full`. | on Linux and Windows; not guaranteed on macOS |
| `open_datasync` | The segment is opened with `O_DSYNC` (write-through on Windows) and no separate flush follows. | where the OS honours the flag down to the medium; not on macOS |
| `barrier` | `F_BARRIERFSYNC` on macOS: the appends reach the medium in order, at a fraction of `full`'s cost, but the drive may still hold the last of them in its cache. For a developer machine. Elsewhere the same as `full`. | on Linux and Windows; on macOS a power loss can take the last acknowledged writes, never one without the writes before it |

The same setting governs the storage tables: an oplog entry is dropped only
once the table it went into is durable, so tables are kept at least as
durable as the oplog. Under `full`, `open_datasync` and `barrier` a table's
own syncs only keep its writes in order (a barrier on macOS), and every
point where the store promises its tables durable (before the oplog behind
them is dropped, a Raft vote, a checkpoint) flushes the device caches under
each endpoint once; `fsync` syncs tables the plain way and never flushes the
device.

Segments rotate at whichever size limit is reached first. In the journal of
an embedded database, `retention_secs` is a floor, not a promise to delete: a
segment is kept while a checkpoint still needs it or any table still lacks one
of its entries. The Raft log of a server (standalone or cluster member) drops
entries once they are in a snapshot and every table holds them;
`retention_secs` does not extend it, and a change stream does not hold it. A
change stream resuming at an entry the log has dropped is refused with
`RETENTION_LOST` rather than resumed past the gap. Zero sizes and unknown keys
are refused when the file is read.

A server node (standalone or cluster member) also keeps a small journal of
its own in `local-journal/`, in the storage endpoint that holds the Raft
log (the data directory unless endpoints are configured): the records that
belong to this member and are not replicated, such as its vote, its log
bounds and the guards of the indexes it serves. It uses the same
`sync_method` and segment limits, syncs only where a record is about to be
relied on, and is replayed at the next start. Every
`local_journal_compact_entries` records (under `storage.oplog`, default
`4096`, zero refused, restart to change) it is compacted: the tables holding
its records are flushed and the segments they cover are dropped, so a lower
value keeps the journal shorter at the cost of more table flushes. It is part of the node's
state: back it up and restore it together with the rest of the data
directory, never on its own. When the directory is later opened as an
embedded database, its records move into the database's journal and the
directory is removed.

### Volatile writes

A write under the `MEMORY` or `CACHE` journal ([durability](./transactions#durability))
is logged, applied and visible like any other; only its answer does not wait
for the oplog append to be synced. Two keys set how the journal of a
standalone node or an embedded database syncs such entries, under
`storage.oplog` with the other oplog settings (restart to change):

```yaml
storage:
  oplog:
    volatile_flush_interval_ms: 100      # default
    volatile_flush_bytes: 1048576        # default, 1 MiB
```

| Key | Default | Description |
|-----|---------|-------------|
| `volatile_flush_interval_ms` | `100` | The interval at which the journal syncs entries written under `MEMORY` or `CACHE`: normally the window of writes a crash can lose. A schedule, not a promise: a slow or failing disk can stretch it. A journaled write syncs every entry before it at once, and so does every flush of a storage table, which never holds a write the journal could still lose. `0` syncs as soon as the journal's flusher wakes. The same default as MongoDB's `journalCommitIntervalMs`. |
| `volatile_flush_bytes` | `1048576` | Unsynced bytes past which the journal syncs without waiting out the interval. Past sixteen times this, a volatile write syncs the journal itself before it is answered, so the unsynced tail stays bounded however fast writes come. `0` syncs every volatile write. The same default as PostgreSQL's `wal_writer_flush_after`. |

A read under the `local` read concern sees a volatile write as soon as it
is answered. A read under `majority`, `snapshot` or `linearizable` sees
only writes a crash cannot take back, on a standalone node too: it reads
the latest state below the oldest entry not yet synced. It does not wait
for a sync and does not cause one. A read at a named timestamp that covers
an unsynced or not yet applied write is refused with `UNAVAILABLE` and
reason `READ_NOT_READY` rather than kept waiting; the metadata carries the
latest timestamp readable now.

A crash loses the unsynced tail of the journal as a whole, in log order:
never part of a write, and never a write whose successor survived. A
`MEMORY` entry stays in the process until its sync, so a killed process
loses it; a `CACHE` entry is handed to the operating system first, so only
a power loss does. A failed sync leaves the journal refusing every later
write until the store is reopened: after a failed `fsync` the operating
system may have dropped the unwritten pages, and a later sync could report
success without them.

On a cluster member the Raft log syncs every batch as it arrives (see the
oplog above); there `MEMORY` and `CACHE` only answer the writer at the
leader's append, before that sync, and the write commits once a majority
holds it synced, as every write does.

Embedded, the setting is the `StorageConfig` field of the same name.

### Compression

The storage tables compress their data blocks per LSM level: the levels above
`cold_level_threshold` with the `hot` codec, that level and the ones below it
with the `cold` codec. The settings sit under `storage.compression`; every key
is optional and changing one requires a restart:

```yaml
storage:
  compression:
    hot: { codec: lz4 }               # default
    cold: { codec: zstd, level: 3 }   # default
    cold_level_threshold: 4           # default
    partitions:                       # none by default
      idx: { codec: lz4 }
```

| Key | Values | Meaning |
|-----|--------|---------|
| `hot`, `cold` | `{ codec: none \| lz4 \| zstd, level: N }` | `level` only for `zstd`: `1`-`22`, `0` for the library default, negative for the fast levels; omitted means `3`. |
| `cold_level_threshold` | `0`-`7` | First level that takes the `cold` codec: `0` puts every level on `cold`, `7` every level on `hot`. |
| `partitions` | partition name to codec | One codec at every level of that partition instead of the hot/cold split. Names: `node`, `adj`, `edgeprop`, `blob`, `blobref`, `schema`, `idx`, `raft`, `counter`, `vec`, `registry`. |

For the smallest data on disk set both `hot` and `cold` to
`{ codec: zstd, level: 22 }`; writes and compactions then spend far more CPU
per byte, and reads decode zstd on every level.

Each table records the codecs it was written with, so a database opens under
any setting: a change applies to tables written from then on, and reaches
existing data as compaction rewrites it. The settings cover the tables' data
blocks; a table's own index blocks keep the storage engine's default, and the
oplog is not compressed (for the inter-node wire, see `wire_compression_level`).
An unknown codec, a level zstd cannot store, a threshold past `7`, an unknown
partition and an unknown key are refused when the file is read.

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

A server node's store keeps its group's history. `backup`, `verify`,
`checkpoint` and `compact` open it as the node itself does and read or
maintain it in place. `restore` writes, and an offline write into one member
would never reach the rest of the group, so `restore` into a group member's
store is refused, naming the group, and the store is left as it was. Restore
into an empty directory instead and start the node on it: it forms its group
from that data, and the other members join it empty.

## Retention and time-travel

MVCC keeps superseded versions so `AS OF TIMESTAMP` queries can read the past
and CDC / backup consumers can resume from a checkpoint. The garbage collector
reclaims versions older than the retention window; the window is therefore the
horizon for both time-travel reads and lagging-consumer recovery.

- `retention_window_secs` sets that horizon, in seconds (default five minutes).
  The storage engine enforces it directly, so it applies identically to a
  server node and to an embedded database (`StorageConfig::retention_window_secs`,
  runtime-tunable with `Database::set_retention_window`). Shorter windows
  reclaim space sooner but shrink the time-travel range and the grace period a
  slow consumer has before its checkpoint is collected. A registered consumer
  lagging beyond the window holds the floor back for itself rather than losing
  data silently. Reads older than the horizon are refused (`OUT_OF_RANGE`,
  reason `OUTSIDE_RETENTION`, metadata `oldest_readable_ts`) instead of being
  answered from partially collected history. A window of `0` turns time travel
  off; statements and open transactions reading the current state are
  unaffected, because each holds its own snapshot for as long as it runs.
- The window covers data only. The deployment's own state (the consumer
  registry and the node's consensus state) is read as it stands now and is
  rewritten on every heartbeat, so compaction keeps only its live version
  plus whatever a running snapshot still reads.
- The window is paid for in storage, per key: compaction keeps every version
  of a key written inside the window plus the newest one below it, and folds
  the rest. Budget disk for the data size plus one stored version per update
  that lands inside the window. Inserts pay nothing extra whatever their key
  order, since a key written once has a single version; a working set
  rewritten N times inside the window holds N versions of it until the
  horizon passes them. Versions the horizon has passed leave the disk when a
  compaction next rewrites the tables holding them, not at the moment the
  horizon moves, so a partition under steady updates sits above that figure
  by whatever its deeper levels have not merged yet. The price is part of
  `coordinode_storage_live_bytes`
  (per partition, the current tree's footprint, in-window versions included).
  `coordinode_storage_retained_history_bytes` reports what is on disk beside
  it: tables a compaction has replaced and that are still waiting to be
  unlinked. It rises by up to a compaction's input size right after an
  install and drains to zero by itself; a figure that stays high is worth
  investigating. Both refresh every five seconds (the capacity-scan
  cadence) on every deployment, whether or not an endpoint sets a capacity
  limit. On the same cadence `coordinode_graph_nodes_total{label}` reports
  the stored nodes per label, read from counters that change in the same
  transaction as the nodes (a temporal node counts once per stored
  version).
- `registry_heartbeat_ms` and `registry_eviction_ms` tune the
  consumer-retention registry's background service: the window over which
  buffered consumer heartbeats are coalesced into one proposal (opened by the
  first heartbeat buffered), and the shortest gap between two sweeps of expired
  registrations (a sweep runs when the earliest registration can expire and
  when a registration changes). With no heartbeat buffered and nothing about to
  expire the service does not wake. The defaults (1000 ms / 1000 ms) suit most
  deployments; raise the heartbeat window on shards with many consumers to cut
  proposal volume.

These are config-file keys (no CLI flag).

## Other subcommands

The same binary runs maintenance and cluster operations. Each opens the database
at a directory given by `--data` (default `./data`); every command that opens
the storage engine (`backup`, `restore`, `verify`, `checkpoint`, `compact`) also
accepts `--config FILE` to open a multi-endpoint node at its configured paths
(see [Storage topology](#storage-topology)). With `--config`, `--data` is
ignored. `healthcheck` opens nothing: it asks a running server.

| Command | Required flags | Notable options |
|---------|----------------|-----------------|
| `coordinode backup` | `--output FILE` | `--config FILE` (multi-endpoint); `--format json\|cypher\|binary\|snapshot`, `--namespace NS`; always the whole database |
| `coordinode restore` | `--input FILE` | `--config FILE` (multi-endpoint); `--format ...` (plus import-only `apoc-json`, `apoc-cypher`, `hetio-json`), `--namespace NS`, `--only-labels L1,L2`, `--force` |
| `coordinode checkpoint` | `--output DIR` | `--config FILE` (multi-endpoint); hard-linked physical checkpoint gathered into one directory, whatever the endpoint layout; restore by pointing `serve --data` at it |
| `coordinode compact` | (none) | `--config FILE` (multi-endpoint); offline major-compaction folding merge operands |
| `coordinode verify` | (none) | `--config FILE` (multi-endpoint); `--deep` for full-page checksum verification |
| `coordinode version` | (none) | Print version and exit |
| `coordinode healthcheck` | (none) | Exit 0 when a running server answers `/ready` with 200, 1 otherwise. Probes the server's `ops_addr`: the built-in default (`[::]:7084`), else the one in `--config FILE`, else `--ops-addr HOST:PORT`, which overrides both. A wildcard address (`[::]`, `0.0.0.0`) is probed on this host's loopback, IPv6 and IPv4. `--timeout-ms N` (default 2000) bounds the connect and the answer. Opens no storage; meant for container health checks |
| `coordinode admin node join` | `--node ADDR --id ID --addr ADDR --identity SPIFFE_ID` | `--pre-seeded`, `--follow` |
| `coordinode admin node decommission` | `--node ADDR --id ID` | `--pruning`, `--force`, `--skip-confirmation` |

Run `coordinode` with no recognized subcommand to print the full usage summary.

### Restoring a logical dump

A `json`, `cypher` or `binary` restore keeps every node's identifier exactly as
it was in the source, and every instance of a discriminated or temporal edge
type and every version of a temporal node under its own discriminator or
`valid_from`. Imports that bring their own numbering (`apoc-json`,
`apoc-cypher`, `hetio-json`) keep it too.

- **Restore into a database that has not issued those identifiers.** The
  restore reads the whole input first and refuses it, writing nothing, when any
  of its identifiers is already taken in the target: held by a node, or handed
  out earlier, including to a node that was deleted since. The error names the
  first identifiers found. A database that never created a node never refuses.
  `--force` relaxes
  only the compatibility checks of a binary dump (format version, schema), never
  this one.
- **Nodes created afterwards get fresh identifiers.** The target's identifier
  allocator is moved past every restored identifier before the first write.
- **An interrupted restore finishes when rerun with the same input.** A restore
  records the digest of its input before it writes. After a crash or a failed
  write, running it again with the same file completes it. Once the interrupted
  restore may have written records, a different input is refused until then.
- **A `json` or `binary` dump brings its schema.** Node types, edge types and
  indexes are declared in the target before the data. A target that already
  declares one of them differently is refused before anything is written. A
  `cypher` dump is portable OpenCypher and holds nodes and edges only: declare
  its types and indexes in the target (`CREATE NODE TYPE ...`,
  `CREATE EDGE TYPE ...`, `CREATE INDEX ...`) before restoring it.
- **Indexes cover the restored data when the restore returns.** Every B-tree,
  vector and full-text index is built from the loaded nodes as the last step of
  the restore. Restored data that breaks a unique index the target declared is
  reported as an error, and that index is marked failed, so lookups stop using
  it, until the data is fixed and the index is created again.

## Growing and shrinking a cluster

A single machine that already holds data becomes a replicated cluster in
place, without export or import.

1. Stop the standalone server and start it again as the first member, with its
   [node identity](#node-identity) and naming the members to come:
   `coordinode serve --config ... --node-id 1 --advertise-addr node1:7080
   --peers node2:7080,node3:7080 --data /var/lib/coordinode`. What it holds
   becomes the group's starting state, and it logs its deployment.
2. Start each new machine as a member with an **empty** data directory and a
   node identity naming that deployment and the first member to enroll from:
   `coordinode serve --config ... --node-id 2 --peers node1:7080,node3:7080
   --data ...`.
3. Add the members from the first one, one at a time, each with the identity
   it proves: `coordinode admin node join --node node1:7080 --id 2 --addr
   node2:7080 --identity spiffe://prod.example/db/node2`, then the same for
   node 3. A member joins as a learner, receives the group's state, and is
   promoted to a voter once it has caught up. A member that has not answered
   yet is never promoted, and the cluster status shows it as `DOWN` until it
   does.

Member addresses are given as `host:port`. Members dial them over mutually
authenticated TLS, expecting the identity the member is bound to.

Only the member a group is formed around brings data into it. A machine that
still holds data of its own is refused when it starts as a joiner, and the
refusal says so and names the fix: start it with an empty data directory.
Its data is not merged; anything it holds that the cluster needs is written
into the cluster through its members, like any other write.

To take a member out, run `coordinode admin node decommission --node
node1:7080 --id 3`. Decommissioning the current leader hands leadership to
another voter first. A CE cluster keeps at least two voters: a decommission
that would leave one is refused with `FAILED_PRECONDITION` and changes
nothing.

The cluster takes one membership change at a time, and status shows a new
member as soon as its change is proposed, before it commits. A join or
decommission issued while the previous change is still committing waits for
it, up to `membership_change_timeout_secs` (default 30), and is refused
naming the change in progress when that runs out. When no leader is known,
because the cluster has lost the quorum to elect one, the command is refused
saying so; nothing is changed.
