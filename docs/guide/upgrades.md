# Upgrading a Group

Members of one replica set never run mixed versions side by side. A group
moves to a new version by majority: you update members one at a time, the
side that holds a majority keeps writing, and a member that does not run the
version its group runs is read-only until it is updated.

## What a version is

Members are matched on a **version pair**:

| Part | Set by | Changes when |
|------|--------|--------------|
| Engine format version | The CoordiNode release | A release changes what members send each other or what the data directory holds |
| Host format epoch | The application embedding CoordiNode | The application changes the format of what it stores. A server always runs epoch `0` |

Releases that share an engine format version are identical on the wire and on
disk and mix freely in one group: nothing moves. Two members whose pairs
differ exchange only a version handshake, nothing else: no replication, no
votes, no routed writes.

## Updating a replica set

1. **Stage** the new release on every member before touching any of them.
2. **Check** that every member is reachable and caught up, and that no
   membership change is in progress.
3. **Update the members that are not the leader, one at a time, back to
   back.** Each one comes up read-only until a majority runs the new version.
   The update that completes the new majority starts the group's single write
   pause: writes resume once that member leads at the new version, which it
   records as its first entry.
4. **Update the old leader last.** It has been read-only since the majority
   moved; it catches up and matches.
5. **Verify** that every member reports the new version, matched.

Reads are served by every member throughout. Writes pause once per move, for
the time it takes the completing member to restart, win an election and bring
enough members up to date. A move can be abandoned only before the new
version is recorded: remove the updated members, wipe them, and add them back
at the old version. A data directory that a newer release has opened is never
opened by an older one again.

## Read-only members

A member that does not run its group's version refuses writes:

- gRPC: `FAILED_PRECONDITION` with reason `MEMBER_READ_ONLY`. The error
  metadata carries `member_engine_format`, `member_host_epoch`,
  `group_engine_format`, `group_host_epoch`, `behind` (`true` when the group
  moved past this member), `as_of_ts` (the commit timestamp its reads are as
  of), and `leader_id` / `leader_addr` when the leader is known. Retry the
  write at the leader.
- PostgreSQL wire protocol: SQLSTATE `25006` (`read_only_sql_transaction`).

It does not forward writes to the leader: forwarding is part of the protocol
it no longer shares with the group.

## Data directory format

A data directory records the engine format that wrote it in an
`ENGINE_FORMAT` file. A release opens a directory of its own format and
migrates one written by the format just before it, once, on open. A directory
two or more formats behind, or written by a newer release, is refused by name
and left untouched: take it through the intermediate release first, or remove
the member from its group and add it back empty.

## Watching a move

`GET /version` on the operational port (`7084` by default) returns the
member's version report as JSON:

```json
{
  "node_id": 2,
  "pair": { "engine": 2, "host_epoch": 0 },
  "group_pair": { "pair": { "engine": 2, "host_epoch": 0 }, "seq": 2 },
  "read_only": null,
  "voters": [
    { "node_id": 1, "pair": { "engine": 1, "host_epoch": 0 } },
    { "node_id": 2, "pair": { "engine": 2, "host_epoch": 0 } },
    { "node_id": 3, "pair": { "engine": 2, "host_epoch": 0 } }
  ],
  "majority_pair": { "engine": 2, "host_epoch": 0 },
  "pause_ms": null
}
```

`read_only`, when set, gives the reason, `behind`, `as_of` and the leader.
`pause_ms` counts the time no version has been able to write.

Prometheus metrics on `/metrics`:

| Metric | Meaning |
|--------|---------|
| `coordinode_version_engine_format`, `coordinode_version_host_epoch` | The pair this member runs |
| `coordinode_version_group_engine_format`, `coordinode_version_group_host_epoch` | The pair its group last recorded |
| `coordinode_version_read_only` | `1` while this member does not run its group's version |
| `coordinode_version_pause_seconds` | Length of the current write pause, `0` while the group writes |
| `coordinode_version_refused_writes_total` | Writes refused as read-only |
| `coordinode_version_refused_calls_total` | Inter-node calls refused from another version or group |
