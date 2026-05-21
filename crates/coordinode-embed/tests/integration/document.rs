//! Integration tests: DOCUMENT property type storage roundtrip.
//!
//! Tests that Document values (rmpv::Value) survive full storage cycle:
//! NodeRecord → MessagePack → CoordiNode storage → read back → deserialize → verify.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_core::graph::node::{NodeId, NodeRecord, PropertyValue};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;

/// Logic-test fixture (memory backing). Returns `EngineFixture` —
/// caller uses `&fx.engine`. Document-property tests verify CRUD +
/// dot-notation projection semantics; no persistence requirement.
fn open_engine() -> coordinode_test_fixtures::EngineFixture {
    coordinode_test_fixtures::engine_for_logic()
}

/// Store a NodeRecord with Document property in storage, read back, verify exact match.
#[test]
fn document_property_storage_roundtrip() {
    let fx = open_engine();
    let engine = &fx.engine;

    // Build a node with a nested Document property
    let doc = rmpv::Value::Map(vec![
        (
            rmpv::Value::String("network".into()),
            rmpv::Value::Map(vec![
                (
                    rmpv::Value::String("ssid".into()),
                    rmpv::Value::String("home-wifi".into()),
                ),
                (
                    rmpv::Value::String("channel".into()),
                    rmpv::Value::Integer(6.into()),
                ),
            ]),
        ),
        (
            rmpv::Value::String("tags".into()),
            rmpv::Value::Array(vec![
                rmpv::Value::String("production".into()),
                rmpv::Value::Integer(42.into()),
                rmpv::Value::Boolean(true),
            ]),
        ),
    ]);

    let mut rec = NodeRecord::new("Device");
    rec.set(1, PropertyValue::String("sensor-001".into()));
    rec.set(2, PropertyValue::Document(doc.clone()));

    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    LocalNodeStore::new(engine)
        .put(0, NodeId::from_raw(1), &rec)
        .expect("put to storage");

    // Read back
    let restored = LocalNodeStore::new(engine)
        .get(0, NodeId::from_raw(1))
        .expect("get from storage")
        .expect("key should exist");

    assert_eq!(restored.primary_label(), "Device");
    assert_eq!(
        restored.get(1),
        Some(&PropertyValue::String("sensor-001".into()))
    );
    assert_eq!(restored.get(2), Some(&PropertyValue::Document(doc)));
}

/// 5-level nested Document survives storage roundtrip.
#[test]
fn document_deep_nesting_storage_roundtrip() {
    let fx = open_engine();
    let engine = &fx.engine;

    let level5 = rmpv::Value::Map(vec![(
        rmpv::Value::String("value".into()),
        rmpv::Value::F64(9.81),
    )]);
    let level4 = rmpv::Value::Map(vec![(rmpv::Value::String("d".into()), level5)]);
    let level3 = rmpv::Value::Map(vec![(rmpv::Value::String("c".into()), level4)]);
    let level2 = rmpv::Value::Map(vec![(rmpv::Value::String("b".into()), level3)]);
    let level1 = rmpv::Value::Map(vec![(rmpv::Value::String("a".into()), level2)]);

    let mut rec = NodeRecord::new("Config");
    rec.set(1, PropertyValue::Document(level1.clone()));

    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    LocalNodeStore::new(engine)
        .put(0, NodeId::from_raw(2), &rec)
        .expect("put to storage");

    let restored = LocalNodeStore::new(engine)
        .get(0, NodeId::from_raw(2))
        .expect("get from storage")
        .expect("key should exist");

    assert_eq!(restored.get(1), Some(&PropertyValue::Document(level1)));
}

/// Document with heterogeneous array (mixed types) — no homogeneity constraint.
#[test]
fn document_heterogeneous_array_storage() {
    let fx = open_engine();
    let engine = &fx.engine;

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("mixed".into()),
        rmpv::Value::Array(vec![
            rmpv::Value::Integer(1.into()),
            rmpv::Value::String("two".into()),
            rmpv::Value::Boolean(false),
            rmpv::Value::Nil,
            rmpv::Value::F64(4.0),
        ]),
    )]);

    let mut rec = NodeRecord::new("Test");
    rec.set(1, PropertyValue::Document(doc.clone()));

    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    LocalNodeStore::new(engine)
        .put(0, NodeId::from_raw(3), &rec)
        .expect("put to storage");

    let restored = LocalNodeStore::new(engine)
        .get(0, NodeId::from_raw(3))
        .expect("get from storage")
        .expect("key should exist");

    assert_eq!(restored.get(1), Some(&PropertyValue::Document(doc)));
}

/// Node with both regular properties and Document coexist.
#[test]
fn document_mixed_with_regular_properties() {
    let fx = open_engine();
    let engine = &fx.engine;

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("setting".into()),
        rmpv::Value::Boolean(true),
    )]);

    let mut rec = NodeRecord::new("User");
    rec.set(1, PropertyValue::String("Alice".into()));
    rec.set(2, PropertyValue::Int(30));
    rec.set(3, PropertyValue::Float(1.75));
    rec.set(4, PropertyValue::Bool(true));
    rec.set(5, PropertyValue::Vector(vec![0.1, 0.2, 0.3]));
    rec.set(6, PropertyValue::Document(doc.clone()));

    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    LocalNodeStore::new(engine)
        .put(0, NodeId::from_raw(4), &rec)
        .expect("put to storage");

    let restored = LocalNodeStore::new(engine)
        .get(0, NodeId::from_raw(4))
        .expect("get from storage")
        .expect("key should exist");

    assert_eq!(
        restored.get(1),
        Some(&PropertyValue::String("Alice".into()))
    );
    assert_eq!(restored.get(2), Some(&PropertyValue::Int(30)));
    assert_eq!(restored.get(3), Some(&PropertyValue::Float(1.75)));
    assert_eq!(restored.get(4), Some(&PropertyValue::Bool(true)));
    assert_eq!(
        restored.get(5),
        Some(&PropertyValue::Vector(vec![0.1, 0.2, 0.3]))
    );
    assert_eq!(restored.get(6), Some(&PropertyValue::Document(doc)));
}

/// Document persists across close/reopen (crash safety).
#[test]
fn document_persists_across_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("important".into()),
        rmpv::Value::String("data".into()),
    )]);

    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let id = NodeId::from_raw(5);

    // Write + close
    {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = StorageEngine::open(&config).expect("open");
        let mut rec = NodeRecord::new("Persistent");
        rec.set(1, PropertyValue::Document(doc.clone()));
        LocalNodeStore::new(&engine)
            .put(0, id, &rec)
            .expect("put to storage");
        // engine drops here (storage flush on drop)
    }

    // Reopen + verify
    {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = StorageEngine::open(&config).expect("reopen");
        let restored = LocalNodeStore::new(&engine)
            .get(0, id)
            .expect("get from storage")
            .expect("key should exist after reopen");
        assert_eq!(restored.primary_label(), "Persistent");
        assert_eq!(restored.get(1), Some(&PropertyValue::Document(doc)));
    }
}

/// Dot-notation evaluation on Document property through eval pipeline.
/// Simulates what the query executor does when MATCH returns a Document property
/// and RETURN/WHERE accesses nested paths.
#[test]
fn document_dot_notation_eval_through_pipeline() {
    use coordinode_core::graph::types::Value;
    use coordinode_query::cypher::ast::Expr;
    use coordinode_query::executor::eval::eval_expr;
    use coordinode_query::executor::row::Row;

    let config = rmpv::Value::Map(vec![
        (
            rmpv::Value::String("network".into()),
            rmpv::Value::Map(vec![
                (
                    rmpv::Value::String("ssid".into()),
                    rmpv::Value::String("home-wifi".into()),
                ),
                (
                    rmpv::Value::String("channel".into()),
                    rmpv::Value::Integer(6.into()),
                ),
            ]),
        ),
        (
            rmpv::Value::String("version".into()),
            rmpv::Value::String("2.1.3".into()),
        ),
    ]);

    // Simulate row as query executor would populate from MATCH scan
    let mut row = Row::new();
    row.insert("n".into(), Value::Int(1));
    row.insert("n.__label__".into(), Value::String("Device".into()));
    row.insert("n.name".into(), Value::String("sensor-001".into()));
    row.insert("n.config".into(), Value::Document(config));

    // n.config.version → "2.1.3"
    let expr_version = Expr::PropertyAccess {
        expr: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "config".into(),
        }),
        property: "version".into(),
    };
    assert_eq!(
        eval_expr(&expr_version, &row),
        Value::String("2.1.3".into())
    );

    // n.config.network.ssid → "home-wifi" (two hops into Document)
    let expr_ssid = Expr::PropertyAccess {
        expr: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("n".into())),
                property: "config".into(),
            }),
            property: "network".into(),
        }),
        property: "ssid".into(),
    };
    assert_eq!(
        eval_expr(&expr_ssid, &row),
        Value::String("home-wifi".into())
    );

    // n.config.network.channel → Int(6)
    let expr_ch = Expr::PropertyAccess {
        expr: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("n".into())),
                property: "config".into(),
            }),
            property: "network".into(),
        }),
        property: "channel".into(),
    };
    assert_eq!(eval_expr(&expr_ch, &row), Value::Int(6));

    // n.config.nonexistent → Null
    let expr_miss = Expr::PropertyAccess {
        expr: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "config".into(),
        }),
        property: "nonexistent".into(),
    };
    assert_eq!(eval_expr(&expr_miss, &row), Value::Null);

    // n.name still works (flat property, not Document)
    let expr_name = Expr::PropertyAccess {
        expr: Box::new(Expr::Variable("n".into())),
        property: "name".into(),
    };
    assert_eq!(
        eval_expr(&expr_name, &row),
        Value::String("sensor-001".into())
    );
}

/// Full end-to-end: create node with Document property, query with
/// dot-notation through full Cypher pipeline (parse → plan → execute → result).
#[test]
fn document_dot_notation_e2e_cypher_pipeline() {
    use coordinode_embed::Database;

    let mut db = Database::open_in_memory().expect("open db");

    // Step 1: Create node with a placeholder 'config' field via Cypher
    // so the interner learns the field name
    db.execute_cypher("CREATE (n:Device {name: 'sensor-001', config: 'placeholder'})")
        .expect("create");

    // Step 2: Replace the placeholder with a real Document value in storage
    let config_field_id = db
        .interner()
        .lookup("config")
        .expect("config should be interned from CREATE");

    let engine = db.engine();
    let mut iter = engine
        .prefix_scan(Partition::Node, b"node:")
        .expect("prefix scan");
    let first = iter.next().expect("should have at least one node");
    let node_key = first.into_inner().expect("kv").0.to_vec();

    let raw = engine
        .get(Partition::Node, &node_key)
        .expect("get")
        .expect("exists");
    let mut rec = NodeRecord::from_msgpack(&raw).expect("decode");
    let doc = rmpv::Value::Map(vec![
        (
            rmpv::Value::String("network".into()),
            rmpv::Value::Map(vec![(
                rmpv::Value::String("ssid".into()),
                rmpv::Value::String("home-wifi".into()),
            )]),
        ),
        (
            rmpv::Value::String("version".into()),
            rmpv::Value::String("2.1.3".into()),
        ),
    ]);
    rec.set(config_field_id, PropertyValue::Document(doc));
    let bytes = rec.to_msgpack().expect("encode");
    engine
        .put(Partition::Node, &node_key, &bytes)
        .expect("put updated node");

    // Step 3: Query with dot-notation through full Cypher pipeline

    // n.config should return the Document
    let rows = db
        .execute_cypher("MATCH (n:Device) RETURN n.config")
        .expect("query config");
    assert_eq!(rows.len(), 1);
    let config_val = rows[0].get("n.config");
    assert!(
        matches!(config_val, Some(PropertyValue::Document(_))),
        "n.config should be Document, got: {config_val:?}"
    );

    // n.config.version → "2.1.3"
    let rows = db
        .execute_cypher("MATCH (n:Device) RETURN n.config.version")
        .expect("query nested");
    assert_eq!(rows.len(), 1);
    let version = rows[0].get("n.config.version");
    assert_eq!(
        version,
        Some(&PropertyValue::String("2.1.3".into())),
        "n.config.version should be '2.1.3', got: {version:?}"
    );

    // n.config.network.ssid → "home-wifi"
    let rows = db
        .execute_cypher("MATCH (n:Device) RETURN n.config.network.ssid")
        .expect("query deep nested");
    assert_eq!(rows.len(), 1);
    let ssid = rows[0].get("n.config.network.ssid");
    assert_eq!(
        ssid,
        Some(&PropertyValue::String("home-wifi".into())),
        "n.config.network.ssid should be 'home-wifi', got: {ssid:?}"
    );

    // WHERE with dot-notation on Document: filter by nested value
    let rows = db
        .execute_cypher("MATCH (n:Device) WHERE n.config.version = '2.1.3' RETURN n.name")
        .expect("where nested");
    assert_eq!(rows.len(), 1, "WHERE on Document dot-notation should match");
    assert_eq!(
        rows[0].get("n.name"),
        Some(&PropertyValue::String("sensor-001".into()))
    );

    // WHERE negative: no match
    let rows = db
        .execute_cypher("MATCH (n:Device) WHERE n.config.version = '9.9.9' RETURN n.name")
        .expect("where no match");
    assert_eq!(rows.len(), 0, "WHERE with wrong value should return 0 rows");
}
