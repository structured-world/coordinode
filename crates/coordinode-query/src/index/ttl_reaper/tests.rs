use super::*;
// Tests plant fixtures directly (adjacency posting lists, edge-type
// markers, node records) — raw partition + posting access is legitimate
// setup the typed stores can't express.
use coordinode_core::graph::edge::{PostingList, encode_adj_key_forward, encode_adj_key_reverse};
use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::NodeId;
use coordinode_core::schema::definition::PropertyDef;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::partition::Partition;

fn test_engine(dir: &std::path::Path) -> StorageEngine {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    StorageEngine::open(&config).expect("open engine")
}

fn persist_schema(engine: &StorageEngine, schema: &LabelSchema) {
    // Use the typed LocalSchemaStore so both the body and the
    // current-revision pointer are written atomically — matches
    // what `discover_ttl_targets` (via SchemaStore::list_labels)
    // reads back through the pointer indirection.
    use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
    LocalSchemaStore::new(engine)
        .save_label(schema)
        .expect("persist schema");
}

fn insert_node(
    engine: &StorageEngine,
    shard_id: u16,
    node_id: u64,
    label: &str,
    timestamp_us: i64,
    interner: &mut FieldInterner,
) {
    let mut record = NodeRecord::new(label);
    let ts_field = interner.intern("created_at");
    record.set(ts_field, Value::Timestamp(timestamp_us));
    seed_node_record(engine, shard_id, NodeId::from_raw(node_id), &record);
}

/// Commit a built node record in its own MVCC transaction.
fn seed_node_record(engine: &StorageEngine, shard_id: u16, node_id: NodeId, record: &NodeRecord) {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    use coordinode_storage::engine::transaction::{CommitContext, Transaction};
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let mut txn = Transaction::begin(engine, Some(&oracle), read_ts);
    LocalNodeStore
        .put(&mut txn, shard_id, node_id, record)
        .expect("put node");
    let wc = WriteConcern::majority();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).expect("commit node");
}

/// Read a node at the latest committed snapshot via an MVCC transaction.
fn read_node(engine: &StorageEngine, shard_id: u16, node_id: NodeId) -> Option<NodeRecord> {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    use coordinode_storage::engine::transaction::Transaction;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let txn = Transaction::begin(engine, Some(&oracle), read_ts);
    LocalNodeStore
        .get(&txn, shard_id, node_id)
        .expect("get node")
}

fn node_exists(engine: &StorageEngine, shard_id: u16, node_id: u64) -> bool {
    read_node(engine, shard_id, NodeId::from_raw(node_id)).is_some()
}

fn make_ttl_schema(label: &str, duration_secs: u64, scope: TtlScope) -> LabelSchema {
    let mut schema = LabelSchema::new_node_id(label);
    schema.add_property(PropertyDef::new("content", PropertyType::String));
    schema.add_property(PropertyDef::new("created_at", PropertyType::Timestamp));
    schema.add_property(PropertyDef::computed(
        "_ttl",
        ComputedSpec::Ttl {
            duration_secs,
            anchor_field: "created_at".into(),
            scope,
            target_field: None,
        },
    ));
    schema
}

fn now_us() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64
}

// ── reap_computed_ttl_committed: the transactional pass ──────────

/// Commit through the plain commit path, as the database's pipeline does on
/// one member.
fn commit_now(txn: &mut Transaction<'_>) -> Result<(), CommitError> {
    let wc = coordinode_core::txn::write_concern::WriteConcern::majority();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).map(|_| ())
}

/// Commit `record` for `node` in a transaction of `oracle`.
fn put_with(engine: &StorageEngine, oracle: &TimestampOracle, node: u64, record: &NodeRecord) {
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let mut txn = Transaction::begin(engine, Some(oracle), oracle.next());
    LocalNodeStore
        .put(&mut txn, 1, NodeId::from_raw(node), record)
        .expect("put node");
    commit_now(&mut txn).expect("commit node");
}

fn session(interner: &mut FieldInterner, created_at_us: i64) -> NodeRecord {
    let mut record = NodeRecord::new("Session");
    record.set(
        interner.intern("created_at"),
        Value::Timestamp(created_at_us),
    );
    record
}

/// The pass deletes what expired and keeps what did not, committing through
/// the path it is given.
#[test]
fn a_committed_pass_deletes_what_expired() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let mut interner = FieldInterner::new();
    persist_schema(&engine, &make_ttl_schema("Session", 3600, TtlScope::Node));
    let now = now_us();
    put_with(
        &engine,
        &oracle,
        1,
        &session(&mut interner, now - 2 * 3600 * 1_000_000),
    );
    put_with(&engine, &oracle, 2, &session(&mut interner, now));

    let result = reap_computed_ttl_committed(&engine, 1, 1000, &interner, &oracle, &mut commit_now);

    assert_eq!(result.nodes_deleted, 1, "{:?}", result.errors);
    assert!(!node_exists(&engine, 1, 1), "the expired node is gone");
    assert!(node_exists(&engine, 1, 2), "the fresh node stays");
}

/// A pass reads the shard once for all its targets, and decodes in full
/// only the records of a label with a TTL: nodes of other labels, large ones
/// included, cost a read of their labels and no more, however many there
/// are and however many targets the pass serves.
#[test]
fn a_pass_reads_the_shard_once_and_decodes_only_ttl_labels() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let mut interner = FieldInterner::new();
    persist_schema(&engine, &make_ttl_schema("Session", 3600, TtlScope::Node));
    persist_schema(&engine, &make_ttl_schema("Token", 3600, TtlScope::Node));
    let now = now_us();
    let old = now - 2 * 3600 * 1_000_000;
    put_with(&engine, &oracle, 1, &session(&mut interner, old));
    put_with(&engine, &oracle, 2, &session(&mut interner, now));
    let mut token = NodeRecord::new("Token");
    token.set(interner.intern("created_at"), Value::Timestamp(old));
    put_with(&engine, &oracle, 3, &token);
    let blob = interner.intern("state");
    for node in 10..60 {
        let mut other = NodeRecord::new("State");
        other.set(blob, Value::Blob(vec![7; 64 * 1024]));
        put_with(&engine, &oracle, node, &other);
    }

    let result = reap_computed_ttl_committed(&engine, 1, 1000, &interner, &oracle, &mut commit_now);

    assert_eq!(result.nodes_deleted, 2, "{:?}", result.errors);
    assert_eq!(
        result.records_scanned, 53,
        "one read of the shard for both targets"
    );
    assert_eq!(result.records_decoded, 3, "only the TTL labels are decoded");
    assert!(!node_exists(&engine, 1, 1) && !node_exists(&engine, 1, 3));
    assert!(node_exists(&engine, 1, 2) && node_exists(&engine, 1, 10));
}

/// No label declares a TTL: the pass reads nothing of the shard.
#[test]
fn a_pass_without_targets_reads_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let mut interner = FieldInterner::new();
    put_with(&engine, &oracle, 1, &session(&mut interner, 0));

    let result = reap_computed_ttl_committed(&engine, 1, 1000, &interner, &oracle, &mut commit_now);

    assert_eq!(result.records_scanned, 0);
    assert!(node_exists(&engine, 1, 1));
}

/// A renewal committed after the pass read the record and before its page
/// commits is a later version than the one the expiry was decided on: the
/// page is refused, read again, and the renewed record is no longer expired.
#[test]
fn a_renewal_before_the_page_commits_keeps_the_record() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let mut interner = FieldInterner::new();
    persist_schema(&engine, &make_ttl_schema("Session", 3600, TtlScope::Node));
    let now = now_us();
    put_with(
        &engine,
        &oracle,
        1,
        &session(&mut interner, now - 2 * 3600 * 1_000_000),
    );
    let renewed = session(&mut interner, now);

    let mut renewed_once = false;
    let mut commit = |txn: &mut Transaction<'_>| {
        if !renewed_once {
            renewed_once = true;
            put_with(&engine, &oracle, 1, &renewed);
        }
        commit_now(txn)
    };
    let result = reap_computed_ttl_committed(&engine, 1, 1000, &interner, &oracle, &mut commit);

    assert_eq!(result.nodes_deleted, 0, "a renewed record was reaped");
    assert!(node_exists(&engine, 1, 1), "the renewed node stays");
}

/// The same for a field-scoped TTL, where the removal is a merge operand
/// that write-set validation does not compare: only the version condition
/// keeps a renewed field from being removed.
#[test]
fn a_renewal_before_the_page_commits_keeps_the_field() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let mut interner = FieldInterner::new();
    persist_schema(&engine, &make_ttl_schema("Session", 3600, TtlScope::Field));
    let now = now_us();
    put_with(
        &engine,
        &oracle,
        1,
        &session(&mut interner, now - 2 * 3600 * 1_000_000),
    );
    let renewed = session(&mut interner, now);

    let mut renewed_once = false;
    let mut commit = |txn: &mut Transaction<'_>| {
        if !renewed_once {
            renewed_once = true;
            put_with(&engine, &oracle, 1, &renewed);
        }
        commit_now(txn)
    };
    let result = reap_computed_ttl_committed(&engine, 1, 1000, &interner, &oracle, &mut commit);

    assert_eq!(result.fields_removed, 0, "a renewed field was removed");
    let field = interner.lookup("created_at").expect("interned");
    let kept = read_node(&engine, 1, NodeId::from_raw(1)).expect("node");
    assert_eq!(kept.props.get(&field), Some(&Value::Timestamp(now)));
}

/// An edge attached to an expired node after the pass read it is one the
/// deletion never removed: the page is refused rather than leaving that edge
/// pointing at a node that is gone, and the pass read again deletes the node
/// with it.
#[test]
fn an_edge_attached_before_the_page_commits_is_removed_with_the_node() {
    use coordinode_modality::{EdgeStore as _, LocalEdgeStore};

    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let mut interner = FieldInterner::new();
    persist_schema(&engine, &make_ttl_schema("Session", 3600, TtlScope::Node));
    let now = now_us();
    put_with(
        &engine,
        &oracle,
        1,
        &session(&mut interner, now - 2 * 3600 * 1_000_000),
    );
    put_with(&engine, &oracle, 2, &NodeRecord::new("User"));

    let mut attached = false;
    let mut commit = |txn: &mut Transaction<'_>| {
        if !attached {
            attached = true;
            // An edge of a type that did not exist when the pass began, so
            // the pass cannot have known to look for it.
            use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
            let mut link = Transaction::begin(&engine, Some(&oracle), oracle.next());
            LocalSchemaStore::new(&engine)
                .register_edge_type_marker(&mut link, "OWNS")
                .expect("register the type");
            LocalEdgeStore
                .put_edge(
                    &mut link,
                    "OWNS",
                    NodeId::from_raw(2),
                    NodeId::from_raw(1),
                    None,
                )
                .expect("attach");
            commit_now(&mut link).expect("the attachment commits");
        }
        commit_now(txn)
    };
    let result = reap_computed_ttl_committed(&engine, 1, 1000, &interner, &oracle, &mut commit);

    assert_eq!(result.nodes_deleted, 1, "{:?}", result.errors);
    assert!(!node_exists(&engine, 1, 1));
    let left = engine
        .get(
            Partition::Adj,
            &encode_adj_key_forward("OWNS", NodeId::from_raw(2)),
        )
        .expect("read")
        .map(|bytes| PostingList::from_bytes(&bytes).expect("posting").len())
        .unwrap_or(0);
    assert_eq!(left, 0, "an edge points at the reaped node");
}

// ── discover_ttl_targets ─────────────────────────────────────────

#[test]
fn discover_finds_ttl_properties() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let schema = make_ttl_schema("Session", 3600, TtlScope::Node);
    persist_schema(&engine, &schema);

    let targets = discover_ttl_targets(&engine, None).expect("discover");
    assert_eq!(targets.len(), 1);
    assert_eq!(targets[0].label, "Session");
    assert_eq!(targets[0].duration_secs, 3600);
    assert_eq!(targets[0].scope, TtlScope::Node);
}

#[test]
fn discover_ignores_non_ttl_labels() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    // Schema without COMPUTED TTL.
    let mut schema = LabelSchema::new_node_id("User");
    schema.add_property(PropertyDef::new("name", PropertyType::String));
    persist_schema(&engine, &schema);

    let targets = discover_ttl_targets(&engine, None).expect("discover");
    assert!(targets.is_empty());
}

// ── reap_computed_ttl: scope Node ────────────────────────────────

#[test]
fn reap_deletes_expired_node() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let schema = make_ttl_schema("Session", 3600, TtlScope::Node);
    persist_schema(&engine, &schema);

    let now = now_us();

    // Node 1: created 2 hours ago (expired, TTL = 1h).
    insert_node(
        &engine,
        1,
        1,
        "Session",
        now - 2 * 3600 * 1_000_000,
        &mut interner,
    );
    // Node 2: created 30 min ago (NOT expired).
    insert_node(
        &engine,
        1,
        2,
        "Session",
        now - 30 * 60 * 1_000_000,
        &mut interner,
    );

    let result = reap_computed_ttl(&engine, 1, 1000);
    assert_eq!(result.nodes_deleted, 1);
    assert_eq!(result.nodes_checked, 2);

    assert!(
        !node_exists(&engine, 1, 1),
        "expired node should be deleted"
    );
    assert!(node_exists(&engine, 1, 2), "fresh node should remain");
}

#[test]
fn reap_respects_batch_size_limit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let schema = make_ttl_schema("Session", 3600, TtlScope::Node);
    persist_schema(&engine, &schema);

    let old_ts = now_us() - 2 * 3600 * 1_000_000;

    // Create 5 expired nodes.
    for i in 1..=5 {
        insert_node(&engine, 1, i, "Session", old_ts, &mut interner);
    }

    // Batch size = 2 → only 2 deleted.
    let result = reap_computed_ttl(&engine, 1, 2);
    assert_eq!(result.nodes_deleted, 2);

    // Run again → 2 more.
    let result2 = reap_computed_ttl(&engine, 1, 2);
    assert_eq!(result2.nodes_deleted, 2);

    // Run again → last 1.
    let result3 = reap_computed_ttl(&engine, 1, 2);
    assert_eq!(result3.nodes_deleted, 1);
}

// ── reap_computed_ttl: scope Field ───────────────────────────────

#[test]
fn reap_removes_field_on_expiry() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let schema = make_ttl_schema("CacheEntry", 60, TtlScope::Field);
    persist_schema(&engine, &schema);

    let old_ts = now_us() - 120 * 1_000_000; // 2 min ago, TTL = 60s
    insert_node(&engine, 1, 10, "CacheEntry", old_ts, &mut interner);

    let result = reap_computed_ttl(&engine, 1, 1000);
    assert_eq!(result.fields_removed, 1);

    // Node should still exist but without the timestamp field.
    assert!(
        node_exists(&engine, 1, 10),
        "node should survive field removal"
    );

    let record = read_node(&engine, 1, NodeId::from_raw(10)).unwrap();
    // The timestamp field should be removed.
    let has_timestamp = record
        .props
        .values()
        .any(|v| matches!(v, Value::Timestamp(_)));
    assert!(
        !has_timestamp,
        "timestamp field should be removed after TTL expiry"
    );
}

// ── reap_computed_ttl: scope Subtree ─────────────────────────────

/// When `target_field` is specified, Subtree scope must delete the target
/// DOCUMENT field, NOT the anchor TIMESTAMP field that triggered expiry.
///
/// Regression test: Subtree must not behave like Field (deleting
/// anchor_field regardless of target_field).
#[test]
fn reap_subtree_with_target_field_deletes_target_not_anchor() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Schema: anchor = created_at (TIMESTAMP), target = profile_data (String).
    let mut schema = LabelSchema::new_node_id("Profile");
    schema.add_property(PropertyDef::new("created_at", PropertyType::Timestamp));
    schema.add_property(PropertyDef::new("profile_data", PropertyType::String));
    schema.add_property(PropertyDef::computed(
        "_ttl",
        ComputedSpec::Ttl {
            duration_secs: 60,
            anchor_field: "created_at".into(),
            scope: TtlScope::Subtree,
            target_field: Some("profile_data".into()),
        },
    ));
    persist_schema(&engine, &schema);

    // Insert node with expired anchor (created 2 minutes ago) + profile_data content.
    let old_ts = now_us() - 120 * 1_000_000;
    let mut record = NodeRecord::new("Profile");
    let ts_field = interner.intern("created_at");
    let pd_field = interner.intern("profile_data");
    record.set(ts_field, Value::Timestamp(old_ts));
    record.set(pd_field, Value::String("sensitive content".into()));
    seed_node_record(&engine, 1, NodeId::from_raw(30), &record);

    // Use the same interner so the reaper can resolve target_field_id = pd_field.
    // Without an interner, the reaper has no way to map "profile_data" → u32 field_id
    // (props is keyed by u32, not by name).
    let result = reap_computed_ttl_with_interner(&engine, 1, 1000, &interner);
    assert_eq!(
        result.subtrees_removed, 1,
        "subtree removal should be counted"
    );
    assert!(
        node_exists(&engine, 1, 30),
        "node must survive subtree removal"
    );

    // Reload node and verify: profile_data deleted, created_at preserved.
    let updated = read_node(&engine, 1, NodeId::from_raw(30)).expect("node exists");
    assert!(
        !updated.props.contains_key(&pd_field),
        "profile_data must be removed by subtree TTL"
    );
    assert!(
        updated.props.contains_key(&ts_field),
        "created_at (anchor) must NOT be removed — only the target field is deleted"
    );
}

/// When `target_field` is specified but the field name is NOT in the interner
/// (e.g., schema added after database open, no nodes with that field yet),
/// the reaper must skip the deletion AND surface an error in `result.errors`.
///
/// This is NOT a silent no-op — operators must be able to detect the condition.
#[test]
fn reap_subtree_unresolved_target_field_records_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    // Empty interner — "payload" is not interned.
    let interner = FieldInterner::new();

    // Schema with target_field = "payload", but "payload" is not in the interner.
    let mut schema = LabelSchema::new_node_id("Cache");
    schema.add_property(PropertyDef::new("cached_at", PropertyType::Timestamp));
    schema.add_property(PropertyDef::new("payload", PropertyType::String));
    schema.add_property(PropertyDef::computed(
        "_ttl",
        ComputedSpec::Ttl {
            duration_secs: 60,
            anchor_field: "cached_at".into(),
            scope: TtlScope::Subtree,
            target_field: Some("payload".into()),
        },
    ));
    persist_schema(&engine, &schema);

    // Insert expired node using a separate interner (simulates data written after startup).
    let mut write_interner = FieldInterner::new();
    let old_ts = now_us() - 120 * 1_000_000;
    let ts_field = write_interner.intern("cached_at");
    let payload_field = write_interner.intern("payload");
    let mut record = NodeRecord::new("Cache");
    record.set(ts_field, Value::Timestamp(old_ts));
    record.set(payload_field, Value::String("stale data".into()));
    seed_node_record(&engine, 1, NodeId::from_raw(99), &record);

    // Reap with the EMPTY interner — target_field_id will be None.
    let result = reap_computed_ttl_with_interner(&engine, 1, 1000, &interner);

    // No deletion should happen (safe no-op), but an error must be recorded.
    assert_eq!(
        result.subtrees_removed, 0,
        "no deletion when target unresolved"
    );
    assert!(
        !result.errors.is_empty(),
        "must record error for unresolved target_field"
    );
    assert!(
        result.errors[0].contains("payload"),
        "error must mention the unresolved field name, got: {:?}",
        result.errors[0]
    );

    // The node must be untouched.
    let updated = read_node(&engine, 1, NodeId::from_raw(99)).expect("node exists");
    assert!(
        updated.props.contains_key(&payload_field),
        "payload must NOT be removed when target_field_id is unresolved"
    );
}

/// When `target_field` is specified and the first reap deletes it, the node
/// stays alive (anchor_field preserved by design).  On the second pass,
/// `resolve_anchor()` still finds the expired anchor — but the target field
/// is already absent, so NO mutation should be submitted and `subtrees_removed`
/// must NOT be incremented again.
///
/// Without the `record.props.contains_key` guard this would emit a no-op
/// merge mutation and increment the counter on every subsequent pass.
#[test]
fn reap_subtree_second_pass_is_idempotent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let mut schema = LabelSchema::new_node_id("Profile");
    schema.add_property(PropertyDef::new("created_at", PropertyType::Timestamp));
    schema.add_property(PropertyDef::new("bio", PropertyType::String));
    schema.add_property(PropertyDef::computed(
        "_ttl",
        ComputedSpec::Ttl {
            duration_secs: 60,
            anchor_field: "created_at".into(),
            scope: TtlScope::Subtree,
            target_field: Some("bio".into()),
        },
    ));
    persist_schema(&engine, &schema);

    let old_ts = now_us() - 120 * 1_000_000;
    let ts_field = interner.intern("created_at");
    let bio_field = interner.intern("bio");
    let mut record = NodeRecord::new("Profile");
    record.set(ts_field, Value::Timestamp(old_ts));
    record.set(bio_field, Value::String("hello".into()));
    seed_node_record(&engine, 1, NodeId::from_raw(77), &record);

    // First reap: bio must be removed, node survives, subtrees_removed = 1.
    let r1 = reap_computed_ttl_with_interner(&engine, 1, 1000, &interner);
    assert_eq!(r1.subtrees_removed, 1, "first pass must remove bio");
    assert!(node_exists(&engine, 1, 77), "node must survive subtree TTL");

    // Verify bio is gone.
    let after_r1 = read_node(&engine, 1, NodeId::from_raw(77)).expect("node exists");
    assert!(
        !after_r1.props.contains_key(&bio_field),
        "bio removed after first pass"
    );
    assert!(after_r1.props.contains_key(&ts_field), "anchor preserved");

    // Second reap: bio already absent — subtrees_removed must be 0 (idempotent).
    let r2 = reap_computed_ttl_with_interner(&engine, 1, 1000, &interner);
    assert_eq!(
        r2.subtrees_removed, 0,
        "second pass must not count already-absent target as removed"
    );
    assert!(r2.errors.is_empty(), "no errors on idempotent second pass");
}

#[test]
fn reap_removes_subtree_on_expiry() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let schema = make_ttl_schema("TempDoc", 60, TtlScope::Subtree);
    persist_schema(&engine, &schema);

    let old_ts = now_us() - 120 * 1_000_000;
    insert_node(&engine, 1, 20, "TempDoc", old_ts, &mut interner);

    let result = reap_computed_ttl(&engine, 1, 1000);
    assert_eq!(result.subtrees_removed, 1);
    assert!(
        node_exists(&engine, 1, 20),
        "node should survive subtree removal"
    );
}

// ── edge cleanup on Node scope ───────────────────────────────────

#[test]
fn reap_node_scope_cleans_edges() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let schema = make_ttl_schema("Session", 3600, TtlScope::Node);
    persist_schema(&engine, &schema);

    // Register edge type.
    let et_key = coordinode_core::schema::definition::encode_edge_type_schema_key("OWNS", 1);
    engine
        .put(Partition::Schema, &et_key, &[])
        .expect("put edge type");

    let old_ts = now_us() - 2 * 3600 * 1_000_000;

    // Node 1 (expired) and node 100 (not TTL-managed, just a peer).
    insert_node(&engine, 1, 1, "Session", old_ts, &mut interner);

    let mut peer_record = NodeRecord::new("User");
    let name_field = interner.intern("name");
    peer_record.set(name_field, Value::String("alice".into()));
    seed_node_record(&engine, 1, NodeId::from_raw(100), &peer_record);

    // Create edge: node 1 -[OWNS]-> node 100
    let fwd_key = encode_adj_key_forward("OWNS", NodeId::from_raw(1));
    let rev_key = encode_adj_key_reverse("OWNS", NodeId::from_raw(100));
    let fwd_plist = PostingList::from_sorted(vec![100]);
    let rev_plist = PostingList::from_sorted(vec![1]);
    engine
        .put(Partition::Adj, &fwd_key, &fwd_plist.to_bytes().unwrap())
        .unwrap();
    engine
        .put(Partition::Adj, &rev_key, &rev_plist.to_bytes().unwrap())
        .unwrap();

    let result = reap_computed_ttl(&engine, 1, 1000);
    assert_eq!(result.nodes_deleted, 1);

    // Forward adj key for deleted node should be gone.
    assert!(engine.get(Partition::Adj, &fwd_key).unwrap().is_none());

    // Peer node should still exist.
    assert!(node_exists(&engine, 1, 100));
}

// ── no TTL schemas → no-op ───────────────────────────────────────

#[test]
fn reap_no_schemas_is_noop() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let result = reap_computed_ttl(&engine, 1, 1000);
    assert_eq!(result.labels_scanned, 0);
    assert_eq!(result.nodes_checked, 0);
    assert_eq!(result.total_deletions(), 0);
}

// ── regression: multi-Timestamp field anchor resolution ────────

/// BUG: find_anchor_timestamp without interner picks FIRST Timestamp,
/// which may be `updated_at` (fresh) instead of `created_at` (expired).
/// Uses find_anchor_timestamp_with_interner to verify correct resolution.
#[test]
fn find_anchor_without_interner_may_pick_wrong_field() {
    let mut interner = FieldInterner::new();
    let mut record = NodeRecord::new("Session");

    // Intern 5 fields to make HashMap order unpredictable.
    let _f1 = interner.intern("field_a");
    let _f2 = interner.intern("field_b");
    let created_field = interner.intern("created_at");
    let _f3 = interner.intern("field_c");
    let updated_field = interner.intern("updated_at");

    let old_ts = 1000i64; // "expired" timestamp
    let new_ts = 9_999_999_999i64; // "fresh" timestamp

    record.set(created_field, Value::Timestamp(old_ts));
    record.set(updated_field, Value::Timestamp(new_ts));

    // Interner-aware: always finds the correct field.
    let correct = find_anchor_timestamp_with_interner(&record, "created_at", &interner);
    assert_eq!(correct, Some(old_ts), "interner-aware must find created_at");

    // Without interner: finds SOME Timestamp — may or may not be correct.
    let heuristic = find_anchor_timestamp(&record, "created_at");
    assert!(heuristic.is_some(), "should find at least one timestamp");
    // The heuristic might return old_ts or new_ts depending on HashMap order.
    // This is the bug — it's nondeterministic.

    // Verify interner-aware is always correct for both fields.
    let updated = find_anchor_timestamp_with_interner(&record, "updated_at", &interner);
    assert_eq!(updated, Some(new_ts));
}

/// Regression test: reaper with interner resolves correct anchor for
/// multi-Timestamp nodes. Node with created_at=expired + updated_at=fresh
/// MUST be deleted (anchor is created_at).
#[test]
fn reap_multi_timestamp_uses_interner_for_correct_anchor() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let schema = make_ttl_schema("Session", 3600, TtlScope::Node);
    persist_schema(&engine, &schema);

    let now = now_us();
    let two_hours_ago = now - 2 * 3600 * 1_000_000;

    // Node with TWO Timestamp fields.
    let mut record = NodeRecord::new("Session");
    let created_field = interner.intern("created_at");
    let updated_field = interner.intern("updated_at");
    record.set(created_field, Value::Timestamp(two_hours_ago));
    record.set(updated_field, Value::Timestamp(now));

    seed_node_record(&engine, 1, NodeId::from_raw(42), &record);

    // Use interner-aware reaper function directly.
    let result = reap_computed_ttl_with_interner(&engine, 1, 1000, &interner);

    assert_eq!(
        result.nodes_deleted, 1,
        "node with expired created_at should be deleted even when updated_at is fresh"
    );
    assert!(
        !node_exists(&engine, 1, 42),
        "expired node should not exist after reap"
    );
}

/// Claim `key` of `table` for `node_id` in its own committed transaction.
fn seed_table_key(engine: &StorageEngine, table: &str, key: &[Value], node_id: NodeId) {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_storage::engine::transaction::{CommitContext, Transaction};
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let mut txn = Transaction::begin(engine, Some(&oracle), oracle.next());
    LocalTableKeyStore
        .claim(&mut txn, table, key, node_id)
        .expect("claim");
    let wc = WriteConcern::majority();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).expect("commit key");
}

fn key_holder(engine: &StorageEngine, table: &str, key: &[Value]) -> Option<NodeId> {
    LocalTableKeyStore
        .committed_holder(engine, table, key)
        .expect("holder")
}

/// An expired row of a keyed table frees its key with it, so the key can be
/// inserted again; a row that has not expired keeps its key.
#[test]
fn reaping_a_table_row_frees_its_key() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let mut schema = make_ttl_schema("Session", 3600, TtlScope::Node);
    schema.make_table(vec!["content".into()]);
    persist_schema(&engine, &schema);

    let now = now_us();
    let content = interner.intern("content");
    let created = interner.intern("created_at");
    for (id, key, age_us) in [
        (1u64, "old", 2 * 3600 * 1_000_000),
        (2, "new", 60 * 1_000_000),
    ] {
        let mut record = NodeRecord::new("Session");
        record.set(content, Value::String(key.into()));
        record.set(created, Value::Timestamp(now - age_us));
        seed_node_record(&engine, 1, NodeId::from_raw(id), &record);
        seed_table_key(
            &engine,
            "Session",
            &[Value::String(key.into())],
            NodeId::from_raw(id),
        );
    }

    let result = reap_computed_ttl_with_interner(&engine, 1, 1000, &interner);
    assert_eq!(result.nodes_deleted, 1, "{:?}", result.errors);
    assert_eq!(
        key_holder(&engine, "Session", &[Value::String("old".into())]),
        None,
        "the reaped row's key is free"
    );
    assert_eq!(
        key_holder(&engine, "Session", &[Value::String("new".into())]),
        Some(NodeId::from_raw(2)),
        "the live row keeps its key"
    );
}

/// Publish `descriptor` through the catalog in a direct-mode transaction.
fn publish_index(
    engine: &StorageEngine,
    descriptor: super::super::IndexDescriptor,
) -> super::super::IndexDefinition {
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    use coordinode_storage::engine::transaction::Transaction;
    LocalIndexStore::new(engine)
        .publish_definition_txn(
            &mut Transaction::new(engine, None, Timestamp::ZERO, None),
            descriptor,
        )
        .expect("define index")
}

/// Seed the unique `index` on `Session.content` holding `value` for `node`.
fn seed_unique_content(
    engine: &StorageEngine,
    index: &super::super::IndexDefinition,
    value: &str,
    node: NodeId,
) {
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    use coordinode_storage::engine::transaction::Transaction;
    let mut txn = Transaction::new(engine, None, Timestamp::ZERO, None);
    let no_fields = |_: &str| None;
    LocalIndexStore::new(engine)
        .stage_membership(
            &mut txn,
            index,
            &no_fields,
            coordinode_core::index::derive::EntryOwner::node(node.as_raw()),
            None,
            Some(&[Value::String(value.into())]),
        )
        .expect("seed entry");
}

fn content_holder(
    engine: &StorageEngine,
    index: &super::super::IndexDefinition,
    value: &str,
) -> Option<NodeId> {
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    LocalIndexStore::new(engine)
        .committed_conflict(index, &[Value::String(value.into())], NodeId::from_raw(0))
        .expect("holder")
}

/// Seed `record` for `node` with its entry in the `created_at` `index`.
fn seed_with_anchor_entry(
    engine: &StorageEngine,
    index: &super::super::IndexDefinition,
    node: u64,
    record: &NodeRecord,
    created_at_us: i64,
) {
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    use coordinode_storage::engine::transaction::Transaction;
    seed_node_record(engine, 1, NodeId::from_raw(node), record);
    let mut txn = Transaction::new(engine, None, Timestamp::ZERO, None);
    let no_fields = |_: &str| None;
    LocalIndexStore::new(engine)
        .stage_membership(
            &mut txn,
            index,
            &no_fields,
            coordinode_core::index::derive::EntryOwner::node(node),
            None,
            Some(&[Value::Timestamp(created_at_us)]),
        )
        .expect("seed entry");
}

/// With a ready index on the anchor, a pass finds its candidates in the
/// index, below the expiry bound, and reads the shard not at all: the fresh
/// nodes of the label and the nodes of other labels are never read, and only
/// the expired candidates are decoded.
#[test]
fn a_ready_anchor_index_supplies_the_candidates() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    persist_schema(&engine, &make_ttl_schema("Session", 3600, TtlScope::Node));
    let index = publish_index(
        &engine,
        super::super::IndexDescriptor::btree("s_created", "Session", "created_at"),
    );
    let now = now_us();
    let old = now - 2 * 3600 * 1_000_000;
    for node in 1..=3u64 {
        seed_with_anchor_entry(&engine, &index, node, &session(&mut interner, old), old);
    }
    for node in 4..=40u64 {
        seed_with_anchor_entry(&engine, &index, node, &session(&mut interner, now), now);
    }
    let blob = interner.intern("state");
    for node in 100..120u64 {
        let mut other = NodeRecord::new("State");
        other.set(blob, Value::Blob(vec![7; 16 * 1024]));
        seed_node_record(&engine, 1, NodeId::from_raw(node), &other);
    }

    let result = reap_computed_ttl_with_interner(&engine, 1, 1000, &interner);

    assert_eq!(result.nodes_deleted, 3, "{:?}", result.errors);
    assert_eq!(result.records_scanned, 0, "the shard is not read");
    assert_eq!(result.records_decoded, 3, "only the candidates are read");
    for node in 1..=3 {
        assert!(!node_exists(&engine, 1, node));
    }
    assert!(node_exists(&engine, 1, 4) && node_exists(&engine, 1, 100));
}

/// An index still being built is not evidence that a node absent from it
/// has not expired: the pass reads the shard for that target.
#[test]
fn an_anchor_index_being_built_is_not_trusted() {
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    use coordinode_storage::engine::transaction::Transaction;

    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    persist_schema(&engine, &make_ttl_schema("Session", 3600, TtlScope::Node));
    let mut index = publish_index(
        &engine,
        super::super::IndexDescriptor::btree("s_created", "Session", "created_at"),
    );
    index.state = super::super::IndexState::Building {
        written: 0,
        estimated_total: 2,
    };
    LocalIndexStore::new(&engine)
        .put_definition_txn(
            &mut Transaction::new(&engine, None, Timestamp::ZERO, None),
            &index,
        )
        .expect("mark building");
    let old = now_us() - 2 * 3600 * 1_000_000;
    // Expired, and not yet in the index.
    seed_node_record(
        &engine,
        1,
        NodeId::from_raw(1),
        &session(&mut interner, old),
    );

    let result = reap_computed_ttl_with_interner(&engine, 1, 1000, &interner);

    assert_eq!(result.nodes_deleted, 1, "{:?}", result.errors);
    assert_eq!(result.records_scanned, 1, "the shard is read");
    assert!(!node_exists(&engine, 1, 1));
}

/// A reaped node's B-tree entries go with it: its unique value is free for
/// a new node, the live node keeps its own.
#[test]
fn reaping_a_node_frees_its_unique_index_value() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    persist_schema(&engine, &make_ttl_schema("Session", 3600, TtlScope::Node));
    let index = publish_index(
        &engine,
        super::super::IndexDescriptor::btree("s_content", "Session", "content").unique(),
    );

    let now = now_us();
    let content = interner.intern("content");
    let created = interner.intern("created_at");
    for (id, value, age_us) in [
        (1u64, "old", 2 * 3600 * 1_000_000),
        (2, "new", 60 * 1_000_000),
    ] {
        let mut record = NodeRecord::new("Session");
        record.set(content, Value::String(value.into()));
        record.set(created, Value::Timestamp(now - age_us));
        seed_node_record(&engine, 1, NodeId::from_raw(id), &record);
        seed_unique_content(&engine, &index, value, NodeId::from_raw(id));
    }

    let result = reap_computed_ttl_with_interner(&engine, 1, 1000, &interner);
    assert_eq!(result.nodes_deleted, 1, "{:?}", result.errors);
    assert_eq!(
        content_holder(&engine, &index, "old"),
        None,
        "the reaped value is free"
    );
    assert_eq!(
        content_holder(&engine, &index, "new"),
        Some(NodeId::from_raw(2))
    );
}

/// Expiring a property an index reads would leave the entry of its old
/// value behind, so the row keeps the property and the reaper says why.
#[test]
fn an_indexed_property_is_not_expired_by_field() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    persist_schema(&engine, &make_ttl_schema("Session", 3600, TtlScope::Field));
    let index = publish_index(
        &engine,
        super::super::IndexDescriptor::btree("s_created", "Session", "created_at"),
    );
    // A ready index holds an entry for every node of its label.
    let old = now_us() - 2 * 3600 * 1_000_000;
    seed_with_anchor_entry(&engine, &index, 1, &session(&mut interner, old), old);

    let result = reap_computed_ttl_with_interner(&engine, 1, 1000, &interner);
    assert_eq!(result.fields_removed, 0);
    assert!(
        result.errors.iter().any(|e| e.contains("indexed property")),
        "{:?}",
        result.errors
    );
}

/// Without the field dictionary the reaper cannot name a row's key, so it
/// keeps the rows of a keyed table rather than leave their keys held.
#[test]
fn a_keyed_table_is_not_reaped_without_its_key_columns() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let mut schema = make_ttl_schema("Session", 3600, TtlScope::Node);
    schema.make_table(vec!["content".into()]);
    persist_schema(&engine, &schema);
    insert_node(
        &engine,
        1,
        1,
        "Session",
        now_us() - 2 * 3600 * 1_000_000,
        &mut interner,
    );

    let result = reap_computed_ttl(&engine, 1, 1000);
    assert_eq!(result.nodes_deleted, 0);
    assert!(!result.errors.is_empty(), "the kept row is reported");
    assert!(node_exists(&engine, 1, 1));
}

// ── interner-aware anchor lookup ─────────────────────────────────

#[test]
fn find_anchor_with_interner_resolves_correct_field() {
    let mut interner = FieldInterner::new();
    let mut record = NodeRecord::new("Session");

    let created_field = interner.intern("created_at");
    let updated_field = interner.intern("updated_at");

    record.set(created_field, Value::Timestamp(1000));
    record.set(updated_field, Value::Timestamp(2000));

    // Should find created_at (1000), not updated_at (2000).
    let ts = find_anchor_timestamp_with_interner(&record, "created_at", &interner);
    assert_eq!(ts, Some(1000));

    let ts2 = find_anchor_timestamp_with_interner(&record, "updated_at", &interner);
    assert_eq!(ts2, Some(2000));

    // Non-existent field → None.
    let ts3 = find_anchor_timestamp_with_interner(&record, "missing", &interner);
    assert_eq!(ts3, None);
}
