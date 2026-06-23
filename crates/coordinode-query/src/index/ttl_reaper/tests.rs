use super::*;
// Tests plant fixtures directly (adjacency posting lists, edge-type
// markers, node records) — raw partition + posting access is legitimate
// setup the typed stores can't express.
use coordinode_core::graph::edge::PostingList;
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
    let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
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
    let txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
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
/// Regression test for G068: previously Subtree behaved identically to Field
/// (always deleted anchor_field regardless of target_field).
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
