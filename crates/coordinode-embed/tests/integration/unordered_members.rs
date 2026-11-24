//! A document compares by what it holds: its named fields in any order,
//! and the members of a list its property declares unordered in any order.
//! A member such a list would hold twice is refused, at the first write and
//! at every later one; a list that is not declared, and a binary payload,
//! compare as they always did.

#![allow(clippy::expect_used, clippy::panic)]

use std::collections::HashMap;
use std::time::Duration;

use coordinode_core::graph::types::Value;
use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType, SchemaMode};
use coordinode_embed::{
    Database, DatabaseError, Intent, LabelDeclaration, Readiness, SchemaPackage,
};
use rmpv::Value as Doc;

fn s(text: &str) -> Doc {
    Doc::from(text)
}

fn map(fields: Vec<(&str, Doc)>) -> Doc {
    Doc::Map(fields.into_iter().map(|(k, v)| (s(k), v)).collect())
}

/// The same map with its pairs in the opposite order, at every depth: built
/// pair by pair, so no sorted container can undo it.
fn reversed(doc: &Doc) -> Doc {
    match doc {
        Doc::Map(fields) => Doc::Map(
            fields
                .iter()
                .rev()
                .map(|(k, v)| (k.clone(), reversed(v)))
                .collect(),
        ),
        Doc::Array(items) => Doc::Array(items.iter().map(reversed).collect()),
        other => other.clone(),
    }
}

/// One member of the declared list.
fn member(node: u64, revision: u64, operation: &str, result: Option<i64>, kind: &str) -> Doc {
    let mut fields = vec![
        ("node", Doc::from(node)),
        ("revision", Doc::from(revision)),
        ("operation", map(vec![("name", s(operation))])),
        ("kind", s(kind)),
    ];
    if let Some(result) = result {
        fields.push(("result", Doc::from(result)));
    }
    map(fields)
}

/// A document holding `ops`, the declared list, beside an ordinary list and
/// a binary payload.
fn body(ops: Vec<Doc>) -> Doc {
    map(vec![
        ("ops", Doc::Array(ops)),
        ("tags", Doc::Array(vec![s("a"), s("b")])),
        ("payload", Doc::Binary(vec![3, 1, 2])),
    ])
}

fn members() -> Vec<Doc> {
    vec![
        member(1, u64::MAX, "put", Some(7), "A"),
        member(2, 5, "del", None, "B"),
        member(3, 5, "put", Some(0), "A"),
    ]
}

/// The label `Fact`: a key and a document whose `ops` are unordered.
fn fact(mode: SchemaMode) -> LabelSchema {
    let mut schema = LabelSchema::new_node_id("Fact");
    schema.set_mode(mode);
    schema.add_property(PropertyDef::new("key", PropertyType::String));
    schema.add_property(
        PropertyDef::new("body", PropertyType::Document).with_unordered_members(["ops"]),
    );
    schema
}

fn publish(
    db: &Database,
    schema: LabelSchema,
    intent: Intent,
) -> coordinode_embed::SchemaReadiness {
    let published = db
        .publish_schema(SchemaPackage {
            labels: vec![LabelDeclaration { schema, intent }],
            ..SchemaPackage::default()
        })
        .expect("publish");
    db.await_schema(&published, Duration::from_secs(30))
        .expect("await")
}

fn open_declared(dir: &std::path::Path) -> Database {
    let db = Database::open(dir).expect("open");
    assert!(publish(&db, fact(SchemaMode::Strict), Intent::Ensure).all_ready());
    db
}

fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn create(db: &mut Database, key: &str, doc: Doc) -> Result<(), DatabaseError> {
    db.execute_cypher_with_params(
        "CREATE (:Fact {key: $key, body: $body})",
        params(vec![
            ("key", Value::String(key.to_string())),
            ("body", Value::Document(doc)),
        ]),
    )
    .map(drop)
}

/// Whether the fact under `key` holds `candidate`, asked the three ways a
/// statement compares a property: a projected `=`, a WHERE, an inline map.
fn holds(db: &mut Database, key: &str, candidate: &Doc) -> bool {
    let with = || {
        params(vec![
            ("key", Value::String(key.to_string())),
            ("c", Value::Document(candidate.clone())),
        ])
    };
    let projected = db
        .execute_cypher_with_params(
            "MATCH (n:Fact) WHERE n.key = $key RETURN n.body = $c AS same, n.body <> $c AS other",
            with(),
        )
        .expect("projected");
    assert_eq!(projected.len(), 1, "one fact under {key}");
    let same = projected[0].get("same") == Some(&Value::Bool(true));
    assert_eq!(
        projected[0].get("other"),
        Some(&Value::Bool(!same)),
        "<> is the opposite of ="
    );
    let filtered = db
        .execute_cypher_with_params(
            "MATCH (n:Fact) WHERE n.key = $key AND n.body = $c RETURN n",
            with(),
        )
        .expect("filtered")
        .len();
    let inline = db
        .execute_cypher_with_params("MATCH (n:Fact {key: $key, body: $c}) RETURN n", with())
        .expect("inline")
        .len();
    assert_eq!(filtered, usize::from(same), "WHERE agrees");
    assert_eq!(inline, usize::from(same), "the inline map agrees");
    same
}

/// The document stored under `key`, as it is read back.
fn stored(db: &mut Database, key: &str) -> Value {
    db.execute_cypher_with_params(
        "MATCH (n:Fact) WHERE n.key = $key RETURN n.body AS body",
        params(vec![("key", Value::String(key.to_string()))]),
    )
    .expect("read")
    .pop()
    .and_then(|mut row| row.remove("body"))
    .expect("a body")
}

/// The fields of a document compare by name with nothing declared: a
/// document written in one order of its pairs equals, after a reopen, the
/// same document supplied with its pairs reversed, whichever order was
/// written first.
#[test]
fn named_fields_compare_in_any_order_after_a_reopen() {
    let doc = map(vec![
        ("a", Doc::from(1)),
        ("b", map(vec![("x", s("p")), ("y", s("q"))])),
        ("c", Doc::Array(vec![Doc::from(1), Doc::from(2)])),
    ]);
    for (written, candidate) in [(doc.clone(), reversed(&doc)), (reversed(&doc), doc.clone())] {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let mut db = Database::open(dir.path()).expect("open");
            db.execute_cypher("CREATE NODE TYPE Plain WITH (body: DOCUMENT)")
                .expect("declare");
            db.execute_cypher_with_params(
                "CREATE (:Plain {body: $body})",
                params(vec![("body", Value::Document(written))]),
            )
            .expect("create");
        }
        let mut db = Database::open(dir.path()).expect("reopen");
        let rows = db
            .execute_cypher_with_params(
                "MATCH (n:Plain) RETURN n.body = $candidate AS same",
                params(vec![("candidate", Value::Document(candidate.clone()))]),
            )
            .expect("compare");
        assert_eq!(rows[0].get("same"), Some(&Value::Bool(true)));
        // A list keeps its order with nothing declared.
        let Doc::Map(mut fields) = candidate else {
            panic!("a map")
        };
        for (key, value) in &mut fields {
            if key.as_str() == Some("c") {
                *value = Doc::Array(vec![Doc::from(2), Doc::from(1)]);
            }
        }
        let rows = db
            .execute_cypher_with_params(
                "MATCH (n:Plain) RETURN n.body = $candidate AS same",
                params(vec![("candidate", Value::Document(Doc::Map(fields)))]),
            )
            .expect("compare");
        assert_eq!(rows[0].get("same"), Some(&Value::Bool(false)));
    }
}

/// Two databases first written with different orders of one value's members
/// and fields hold the same value after a reopen: each equals the other's
/// order, a retry in another order finds the stored fact and writes
/// nothing, and what is stored stays as first written.
#[test]
fn a_value_written_in_any_order_of_its_members_is_one_value() {
    let first = body(members());
    let mut permuted = members();
    permuted.rotate_left(1);
    permuted.swap(0, 1);
    let second = reversed(&body(permuted));
    for (written, other) in [(first.clone(), second.clone()), (second, first)] {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let mut db = open_declared(dir.path());
            create(&mut db, "k", written.clone()).expect("first write");
        }
        let mut db = Database::open(dir.path()).expect("reopen");
        assert!(holds(&mut db, "k", &written), "its own order");
        assert!(holds(&mut db, "k", &other), "the other order");
        // The retry of the same immutable fact reads it by its key, finds it
        // holds the value, and so writes nothing: it is as first written.
        assert_eq!(stored(&mut db, "k"), Value::Document(written));
        // The declared list read on its own compares the same way.
        let Doc::Map(fields) = &other else {
            panic!("a map")
        };
        let ops = fields
            .iter()
            .find(|(k, _)| k.as_str() == Some("ops"))
            .map(|(_, v)| v.clone())
            .expect("ops");
        let rows = db
            .execute_cypher_with_params(
                "MATCH (n:Fact) RETURN n.body.ops = $ops AS same",
                params(vec![("ops", Value::Document(ops))]),
            )
            .expect("compare the list");
        assert_eq!(rows[0].get("same"), Some(&Value::Bool(true)));
    }
}

/// A member differing in any one of its fields is another member, and so
/// the value another value: its node, its revision, its nested operation,
/// its result present or absent, its kind.
#[test]
fn a_change_of_any_one_field_of_a_member_makes_another_value() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_declared(dir.path());
    create(&mut db, "k", body(members())).expect("write");
    let with_first = |first: Doc| {
        let mut ops = members();
        ops[0] = first;
        ops.reverse();
        body(ops)
    };
    assert!(holds(
        &mut db,
        "k",
        &with_first(member(1, u64::MAX, "put", Some(7), "A"))
    ));
    for changed in [
        member(9, u64::MAX, "put", Some(7), "A"),
        member(1, u64::MAX - 1, "put", Some(7), "A"),
        member(1, u64::MAX, "del", Some(7), "A"),
        member(1, u64::MAX, "put", Some(8), "A"),
        member(1, u64::MAX, "put", None, "A"),
        member(1, u64::MAX, "put", Some(7), "B"),
    ] {
        assert!(
            !holds(&mut db, "k", &with_first(changed.clone())),
            "{changed}"
        );
    }
    // One member fewer is another value too.
    let mut fewer = members();
    fewer.pop();
    assert!(!holds(&mut db, "k", &body(fewer)));
}

/// A list the property does not declare still differs by order, and a
/// binary payload by its bytes.
#[test]
fn an_undeclared_list_and_a_binary_payload_compare_as_before() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_declared(dir.path());
    create(&mut db, "k", body(members())).expect("write");
    let with = |tags: Vec<Doc>, payload: Vec<u8>| {
        map(vec![
            ("ops", Doc::Array(members())),
            ("tags", Doc::Array(tags)),
            ("payload", Doc::Binary(payload)),
        ])
    };
    assert!(holds(
        &mut db,
        "k",
        &with(vec![s("a"), s("b")], vec![3, 1, 2])
    ));
    assert!(
        !holds(&mut db, "k", &with(vec![s("b"), s("a")], vec![3, 1, 2])),
        "an ordinary list in another order"
    );
    assert!(
        !holds(&mut db, "k", &with(vec![s("a"), s("b")], vec![1, 2, 3])),
        "the same bytes in another order"
    );
}

/// A member held twice is refused with the property and the path of its
/// list, whatever the order of the fields of its two copies: at the first
/// write, which stores nothing, and at a later one, which leaves the stored
/// version as it was. The declaration holds after a reopen.
#[test]
fn a_repeated_member_is_refused_and_the_stored_version_stays() {
    let dir = tempfile::tempdir().expect("tempdir");
    let twice = || {
        let mut ops = members();
        ops.push(reversed(&ops[1]));
        body(ops)
    };
    let refused = |e: Result<(), DatabaseError>| match e {
        Err(DatabaseError::DuplicateMember { property, path }) => {
            assert_eq!((property.as_str(), path.as_str()), ("body", "ops"));
        }
        other => panic!("expected a duplicate member refusal, got {other:?}"),
    };
    {
        let mut db = open_declared(dir.path());
        refused(create(&mut db, "k", twice()));
        let none = db
            .execute_cypher("MATCH (n:Fact) RETURN n")
            .expect("scan")
            .len();
        assert_eq!(none, 0, "the refused write stored nothing");
        create(&mut db, "k", body(members())).expect("a valid first version");
    }
    let mut db = Database::open(dir.path()).expect("reopen");
    let set = db
        .execute_cypher_with_params(
            "MATCH (n:Fact) WHERE n.key = 'k' SET n.body = $body",
            params(vec![("body", Value::Document(twice()))]),
        )
        .map(drop);
    refused(set);
    assert_eq!(stored(&mut db, "k"), Value::Document(body(members())));
    // A whole value without a repeat replaces the stored one.
    let mut fewer = members();
    fewer.pop();
    db.execute_cypher_with_params(
        "MATCH (n:Fact) WHERE n.key = 'k' SET n.body = $body",
        params(vec![("body", Value::Document(body(fewer.clone())))]),
    )
    .expect("a valid later version");
    assert!(holds(&mut db, "k", &body(fewer)));
    // What stands where the list is declared is a list.
    let not_a_list = create(&mut db, "x", map(vec![("ops", s("no"))]));
    assert!(
        matches!(not_a_list, Err(DatabaseError::SchemaViolation(_))),
        "{not_a_list:?}"
    );
}

/// Whether `refused` is the refusal of a member repeated in `body.ops`.
fn repeats_a_member<T: std::fmt::Debug>(refused: Result<T, DatabaseError>) {
    match refused {
        Err(DatabaseError::DuplicateMember { property, path }) => {
            assert_eq!((property.as_str(), path.as_str()), ("body", "ops"));
        }
        other => panic!("expected a duplicate member refusal, got {other:?}"),
    }
}

/// A small member, by its two fields.
fn pair(node: i64, x: Option<i64>) -> Doc {
    let mut fields = vec![("node", Doc::from(node))];
    if let Some(x) = x {
        fields.push(("x", Doc::from(x)));
    }
    map(fields)
}

/// A write below the property that touches no declared list goes through
/// as it always did; one that touches a declared list is judged by the
/// document it leaves: a valid one lands, one that would repeat a member is
/// refused and changes nothing.
#[test]
fn a_write_below_the_property_is_judged_by_what_it_leaves() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_declared(dir.path());
    let ops = || vec![pair(1, Some(1)), pair(1, None), pair(2, None)];
    create(&mut db, "k", body(ops())).expect("write");
    let run = |db: &mut Database, statement: &str, value: Option<Doc>| {
        db.execute_cypher_with_params(
            statement,
            value.map_or_else(HashMap::new, |v| params(vec![("v", Value::Document(v))])),
        )
    };

    // Beside the declared list: nothing of it is read or judged.
    run(&mut db, "MATCH (n:Fact) SET n.body.tags = ['z', 'z']", None).expect("an ordinary list");
    run(
        &mut db,
        "MATCH (n:Fact) SET doc_push(n.body.tags, 'z')",
        None,
    )
    .expect("push beside");
    run(&mut db, "MATCH (n:Fact) REMOVE n.body.payload", None).expect("remove beside");

    // The declared list itself: a new member lands, a held one does not.
    run(
        &mut db,
        "MATCH (n:Fact) SET doc_push(n.body.ops, $v)",
        Some(pair(3, None)),
    )
    .expect("a new member");
    repeats_a_member(run(
        &mut db,
        "MATCH (n:Fact) SET doc_push(n.body.ops, $v)",
        Some(reversed(&pair(1, Some(1)))),
    ));
    // The transaction's own earlier change is part of what is judged.
    repeats_a_member(run(
        &mut db,
        "MATCH (n:Fact) SET doc_push(n.body.ops, $v), doc_push(n.body.ops, $v)",
        Some(pair(4, None)),
    ));
    // A path cannot reach inside a member: a write through the list would
    // put a map where the list stands, which the declaration refuses, and a
    // removal through it finds nothing to remove.
    let through = run(&mut db, "MATCH (n:Fact) SET n.body.ops.`2`.node = 3", None);
    assert!(
        matches!(through, Err(DatabaseError::SchemaViolation(_))),
        "{through:?}"
    );
    run(&mut db, "MATCH (n:Fact) REMOVE n.body.ops.`0`.x", None).expect("nothing to remove");
    // The list replaced whole, and the document above it.
    repeats_a_member(run(
        &mut db,
        "MATCH (n:Fact) SET n.body.ops = $v",
        Some(Doc::Array(vec![pair(5, None), pair(5, None)])),
    ));

    let expected = map(vec![
        (
            "ops",
            Doc::Array(vec![
                pair(1, Some(1)),
                pair(1, None),
                pair(2, None),
                pair(3, None),
            ]),
        ),
        ("tags", Doc::Array(vec![s("z"), s("z"), s("z")])),
    ]);
    assert!(holds(&mut db, "k", &expected), "{:?}", stored(&mut db, "k"));

    run(
        &mut db,
        "MATCH (n:Fact) SET n.body.ops = $v",
        Some(Doc::Array(vec![pair(6, None), pair(5, None)])),
    )
    .expect("a valid list");
    let replaced = map(vec![
        ("ops", Doc::Array(vec![pair(5, None), pair(6, None)])),
        ("tags", Doc::Array(vec![s("z"), s("z"), s("z")])),
    ]);
    assert!(holds(&mut db, "k", &replaced), "{:?}", stored(&mut db, "k"));
}

/// Two transactions each add the same member, each valid alone: the one
/// that commits second is refused, and the list holds the member once.
#[test]
fn two_writers_cannot_land_one_member_twice() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_declared(dir.path());
    create(&mut db, "k", body(vec![pair(1, None)])).expect("write");
    let push = |db: &mut Database, tx| {
        db.execute_in_transaction(
            tx,
            "MATCH (n:Fact) SET doc_push(n.body.ops, $v)",
            Some(params(vec![("v", Value::Document(pair(2, None)))])),
        )
    };
    let first = db.begin_transaction().expect("begin");
    let second = db.begin_transaction().expect("begin");
    push(&mut db, first).expect("valid alone");
    push(&mut db, second).expect("valid alone");
    db.commit_transaction(first).expect("the first commits");
    let refused = db.commit_transaction(second);
    assert!(refused.is_err(), "the second commit landed: {refused:?}");

    let expected = map(vec![
        ("ops", Doc::Array(vec![pair(1, None), pair(2, None)])),
        ("tags", Doc::Array(vec![s("a"), s("b")])),
        ("payload", Doc::Binary(vec![3, 1, 2])),
    ]);
    assert!(holds(&mut db, "k", &expected), "{:?}", stored(&mut db, "k"));
}

/// The declaration holds on a FLEXIBLE label, which enforces nothing else
/// of its definition.
#[test]
fn the_declaration_holds_on_a_flexible_label() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    assert!(publish(&db, fact(SchemaMode::Flexible), Intent::Ensure).all_ready());
    let mut ops = members();
    ops.push(ops[0].clone());
    assert!(matches!(
        create(&mut db, "k", body(ops)),
        Err(DatabaseError::DuplicateMember { .. })
    ));
    create(&mut db, "k", body(members())).expect("write");
    let mut permuted = members();
    permuted.reverse();
    assert!(holds(&mut db, "k", &body(permuted)));
}

/// A declaration made over stored documents is proved against them: one
/// holding a member twice refuses it, and the label keeps the definition it
/// had.
#[test]
fn a_declaration_is_proved_over_the_stored_documents() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    let mut plain = LabelSchema::new_node_id("Fact");
    plain.add_property(PropertyDef::new("key", PropertyType::String));
    plain.add_property(PropertyDef::new("body", PropertyType::Document));
    assert!(publish(&db, plain, Intent::Ensure).all_ready());
    let mut ops = members();
    ops.push(ops[0].clone());
    create(&mut db, "k", body(ops.clone())).expect("nothing declared yet");

    let revision = db
        .label_schemas()
        .expect("read")
        .into_iter()
        .find(|schema| schema.name == "Fact")
        .expect("defined")
        .schema_revision;
    let readiness = publish(
        &db,
        fact(SchemaMode::Strict),
        Intent::Change {
            expected_revision: revision,
        },
    );
    assert!(
        matches!(
            readiness.of(coordinode_embed::CatalogObject::Label, "Fact"),
            Some(Readiness::Failed(_))
        ),
        "{readiness:?}"
    );
    // The definition in force declares nothing: the list is still ordered.
    let mut permuted = ops;
    permuted.reverse();
    assert!(!holds(&mut db, "k", &body(permuted)));
}

/// Only a DOCUMENT property of a label declares unordered members, by
/// paths that name fields.
#[test]
fn a_declaration_out_of_place_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    let declare = |property: PropertyDef| {
        let mut schema = LabelSchema::new_node_id("Wrong");
        schema.add_property(property);
        db.publish_schema(SchemaPackage {
            labels: vec![LabelDeclaration {
                schema,
                intent: Intent::Ensure,
            }],
            ..SchemaPackage::default()
        })
    };
    for wrong in [
        PropertyDef::new("p", PropertyType::String).with_unordered_members(["a"]),
        PropertyDef::new("p", PropertyType::Document).with_unordered_members([""]),
        PropertyDef::new("p", PropertyType::Document)
            .with_unordered_members(["a"])
            .with_unordered_members(["a"]),
    ] {
        let refused = declare(wrong.clone());
        assert!(
            matches!(refused, Err(DatabaseError::CatalogRefused(_))),
            "{wrong:?}: {refused:?}"
        );
    }
}

/// A MERGE naming the value in another order of its members finds the node
/// the first one made, and makes no second.
#[test]
fn a_merge_in_another_order_finds_the_same_node() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_declared(dir.path());
    let merge = |db: &mut Database, doc: Doc| {
        db.execute_cypher_with_params(
            "MERGE (n:Fact {key: 'k', body: $body}) RETURN n",
            params(vec![("body", Value::Document(doc))]),
        )
        .expect("merge");
    };
    merge(&mut db, body(members()));
    let mut permuted = members();
    permuted.reverse();
    merge(&mut db, reversed(&body(permuted)));
    let nodes = db
        .execute_cypher("MATCH (n:Fact) RETURN n")
        .expect("scan")
        .len();
    assert_eq!(nodes, 1);
    assert_eq!(stored(&mut db, "k"), Value::Document(body(members())));
}

/// One parameter compares with each property under that property's own
/// declaration: rows of a type that declares the list equal a permuted
/// value, rows of a type that does not differ by order, in one statement;
/// and a node bound under another name keeps its type's declaration.
#[test]
fn one_parameter_follows_the_declaration_of_each_row() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_declared(dir.path());
    let mut plain = LabelSchema::new_node_id("Loose");
    plain.add_property(PropertyDef::new("key", PropertyType::String));
    plain.add_property(PropertyDef::new("body", PropertyType::Document));
    assert!(publish(&db, plain, Intent::Ensure).all_ready());
    create(&mut db, "k", body(members())).expect("declared");
    db.execute_cypher_with_params(
        "CREATE (:Loose {key: 'k', body: $body})",
        params(vec![("body", Value::Document(body(members())))]),
    )
    .expect("undeclared");

    let mut permuted = members();
    permuted.reverse();
    let candidate = || params(vec![("c", Value::Document(body(permuted.clone())))]);
    let mut matched: Vec<String> = db
        .execute_cypher_with_params(
            "MATCH (n) WHERE n.body = $c RETURN labels(n) AS labels",
            candidate(),
        )
        .expect("both types")
        .into_iter()
        .map(|row| format!("{:?}", row.get("labels")))
        .collect();
    matched.sort();
    assert_eq!(matched.len(), 1, "{matched:?}");
    assert!(matched[0].contains("Fact"), "{matched:?}");

    let renamed = db
        .execute_cypher_with_params(
            "MATCH (n:Fact) WITH n AS m WHERE m.body = $c RETURN m.body = $c AS same",
            candidate(),
        )
        .expect("another name");
    assert_eq!(renamed.len(), 1);
    assert_eq!(renamed[0].get("same"), Some(&Value::Bool(true)));
}

/// A number in a member is the value it represents: an integer and the
/// float of the same value are one member, in a comparison and as a repeat.
#[test]
fn a_number_in_a_member_is_its_value() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_declared(dir.path());
    let of = |values: Vec<Doc>| map(vec![("ops", Doc::Array(values))]);
    create(&mut db, "k", of(vec![Doc::from(1), Doc::from(2)])).expect("write");
    assert!(holds(&mut db, "k", &of(vec![Doc::F64(2.0), Doc::F64(1.0)])));
    assert!(!holds(
        &mut db,
        "k",
        &of(vec![Doc::F64(2.5), Doc::F64(1.0)])
    ));
    repeats_a_member(create(&mut db, "x", of(vec![Doc::from(1), Doc::F64(1.0)])));
    // 2^53 + 1 is no float: it is not the float beside it.
    let exact = 1_i64 << 53;
    create(
        &mut db,
        "big",
        of(vec![Doc::from(exact + 1), Doc::F64(exact as f64)]),
    )
    .expect("two numbers");
}
