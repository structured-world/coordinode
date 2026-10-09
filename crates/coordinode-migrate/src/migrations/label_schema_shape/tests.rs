use coordinode_core::schema::definition::{ConstraintKind, ConstraintState, LabelSchema};
use rmpv::Value;

use super::upgrade_schema;

fn prop(name: &str, ty: &str, not_null: bool, unique: Option<bool>) -> (Value, Value) {
    let mut def = vec![name.into(), ty.into(), not_null.into(), Value::Nil];
    if let Some(u) = unique {
        def.push(u.into());
    }
    (name.into(), Value::Array(def))
}

fn constraint(name: &str, property: &str, state: &str) -> Value {
    Value::Array(vec![
        name.into(),
        Value::Array(vec![property.into()]),
        "Unique".into(),
        state.into(),
    ])
}

/// A label schema as a build before the change wrote it: five-field
/// properties and four-field constraints, the shape found in a store.
fn old_schema(props: Vec<(Value, Value)>, constraints: Vec<Value>) -> Vec<u8> {
    let schema = Value::Array(vec![
        "Agent".into(),
        Value::Map(props),
        "Validated".into(),
        "NodeId".into(),
        Value::Array(vec![Value::Array(vec![
            "__node_id__".into(),
            "Primary".into(),
            "NodeId".into(),
            2.into(),
        ])]),
        2.into(),
        false.into(),
        Value::Array(vec![]),
        "Row".into(),
        false.into(),
        Value::Array(constraints),
    ]);
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &schema).expect("encode");
    out
}

fn read(bytes: &[u8]) -> LabelSchema {
    LabelSchema::from_msgpack(bytes).expect("reads as current")
}

/// The shape found in a store: unique flags of false are dropped, the
/// constraint gets no scope, and everything else reads as it was.
#[test]
fn an_old_schema_reads_as_current() {
    let old = old_schema(
        vec![
            prop("last_seen_at", "Timestamp", false, Some(false)),
            prop("session_id", "String", true, Some(false)),
        ],
        vec![constraint(
            "agent_session_unique",
            "session_id",
            "Validating",
        )],
    );
    let err = LabelSchema::from_msgpack(&old).expect_err("refused as it was");
    assert!(err.to_string().contains("expected 4"), "{err}");

    let schema = read(&upgrade_schema(&old).expect("upgrade").expect("rewritten"));
    assert_eq!(schema.name, "Agent");
    assert_eq!(schema.schema_revision, 2);
    assert!(schema.properties["session_id"].not_null);
    assert!(!schema.properties["last_seen_at"].not_null);
    let [c] = schema.constraints() else {
        unreachable!("one constraint")
    };
    assert_eq!(c.name, "agent_session_unique");
    assert_eq!(c.properties, vec!["session_id".to_string()]);
    assert_eq!(c.kind, ConstraintKind::Unique);
    assert_eq!(c.state, ConstraintState::Validating);
    assert_eq!(c.scope, None);
}

/// A unique flag of true is dropped when a uniqueness constraint on that
/// property holds it already.
#[test]
fn a_unique_flag_held_by_a_constraint_is_dropped() {
    let old = old_schema(
        vec![prop("session_id", "String", true, Some(true))],
        vec![constraint("agent_session_unique", "session_id", "Active")],
    );
    let schema = read(&upgrade_schema(&old).expect("upgrade").expect("rewritten"));
    assert_eq!(schema.constraints().len(), 1);
}

/// A unique flag of true with no constraint behind it stops the run: the
/// rewrite would lose the uniqueness.
#[test]
fn a_unique_flag_without_a_constraint_stops_the_run() {
    let old = old_schema(vec![prop("session_id", "String", true, Some(true))], vec![]);
    let err = upgrade_schema(&old).expect_err("refused");
    assert!(format!("{err:#}").contains("session_id"), "{err:#}");
}

/// A current schema is left alone, and a shape that is neither stops the
/// run.
#[test]
fn current_schemas_stay_and_unknown_shapes_stop() {
    let old = old_schema(vec![prop("id", "String", true, Some(false))], vec![]);
    let current = upgrade_schema(&old).expect("upgrade").expect("rewritten");
    assert_eq!(upgrade_schema(&current).expect("survey"), None);

    let six = old_schema(
        vec![(
            "id".into(),
            Value::Array(vec![
                "id".into(),
                "String".into(),
                true.into(),
                Value::Nil,
                false.into(),
                false.into(),
            ]),
        )],
        vec![],
    );
    let err = upgrade_schema(&six).expect_err("six fields");
    assert!(format!("{err:#}").contains("6 fields"), "{err:#}");
}
