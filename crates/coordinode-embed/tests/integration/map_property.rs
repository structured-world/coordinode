//! A property declared MAP takes a map passed as a parameter, keeps it
//! across a reopen, and equals the same map whatever order its fields were
//! given in; a DOCUMENT property keeps taking the same value.

#![allow(clippy::expect_used, clippy::panic)]

use std::collections::{BTreeMap, HashMap};

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

/// `{a: 1, b: "x"}` as a parameter map.
fn fields() -> Value {
    Value::Map(BTreeMap::from([
        ("a".to_string(), Value::Int(1)),
        ("b".to_string(), Value::String("x".to_string())),
    ]))
}

fn params(value: Value) -> HashMap<String, Value> {
    HashMap::from([("m".to_string(), value)])
}

/// How many nodes of `label` hold `value` in `property`, by equality with
/// the parameter.
fn holding(db: &mut Database, label: &str, property: &str, value: Value) -> usize {
    db.execute_cypher_with_params(
        &format!("MATCH (n:{label}) WHERE n.{property} = $m RETURN n"),
        params(value),
    )
    .expect("match")
    .len()
}

/// A map parameter is stored in a MAP property and found again by equality
/// with a map parameter, before and after a reopen; a map of other fields
/// is not found.
#[test]
fn a_map_property_takes_a_map_parameter_and_equals_it() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE NODE TYPE Holder WITH (meta: MAP)")
            .expect("declare");
        db.execute_cypher_with_params("CREATE (:Holder {meta: $m})", params(fields()))
            .expect("a MAP property takes a map parameter");
        assert_eq!(holding(&mut db, "Holder", "meta", fields()), 1);
    }
    let mut db = Database::open(dir.path()).expect("reopen");
    assert_eq!(
        holding(&mut db, "Holder", "meta", fields()),
        1,
        "after reopen"
    );
    let other = Value::Map(BTreeMap::from([("a".to_string(), Value::Int(2))]));
    assert_eq!(holding(&mut db, "Holder", "meta", other), 0, "other fields");
    // A value that is no map is still refused.
    let refused =
        db.execute_cypher_with_params("CREATE (:Holder {meta: $m})", params(Value::Int(1)));
    assert!(refused.is_err(), "an INT in a MAP property: {refused:?}");
}

/// A DOCUMENT property takes the same map and equals it as a parameter.
#[test]
fn a_document_property_equals_the_map_it_was_written_from() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher("CREATE NODE TYPE Doc WITH (body: DOCUMENT)")
        .expect("declare");
    db.execute_cypher_with_params("CREATE (:Doc {body: $m})", params(fields()))
        .expect("create");
    assert_eq!(holding(&mut db, "Doc", "body", fields()), 1);
}
