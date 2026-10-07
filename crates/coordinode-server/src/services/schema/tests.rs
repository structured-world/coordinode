use super::*;
use crate::proto::v2::graph::schema_service_server::SchemaService;
use crate::services::error_details::ERROR_DOMAIN;
use tonic_types::StatusExt;

fn test_service() -> (SchemaServiceImpl, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let database = Arc::new(RwLock::new(
        Database::open(dir.path()).expect("open database"),
    ));
    (SchemaServiceImpl::new(database), dir)
}

fn scalar(s: schema::ScalarType) -> Option<schema::PropertyType> {
    Some(schema::PropertyType {
        r#type: Some(schema::property_type::Type::Scalar(s as i32)),
    })
}

/// A stored property of scalar type `s`.
fn prop(name: &str, s: schema::ScalarType, required: bool) -> schema::PropertyDefinition {
    schema::PropertyDefinition {
        name: name.to_string(),
        r#type: scalar(s),
        required,
        default_value: None,
    }
}

/// A label definition with `properties` in `mode`.
fn label(
    name: &str,
    properties: Vec<schema::PropertyDefinition>,
    mode: schema::SchemaMode,
) -> schema::CreateLabelRequest {
    schema::CreateLabelRequest {
        name: name.to_string(),
        properties,
        computed_properties: vec![],
        schema_mode: mode as i32,
        temporal: false,
    }
}

fn unique(name: &str, label: &str, property: &str) -> schema::CreateConstraintRequest {
    schema::CreateConstraintRequest {
        name: name.to_string(),
        target: Some(schema::create_constraint_request::Target::Label(
            label.to_string(),
        )),
        properties: vec![property.to_string()],
        kind: schema::ConstraintKind::Unique as i32,
        property_type: None,
        if_not_exists: false,
        wait: None,
        on_duplicate_rename: String::new(),
        scope: None,
    }
}

/// The reason a refusal names, checked against this server's domain.
fn reason(status: &Status) -> String {
    let info = status
        .get_details_error_info()
        .unwrap_or_else(|| panic!("no ErrorInfo on {status:?}"));
    assert_eq!(info.domain, ERROR_DOMAIN);
    info.reason
}

/// list_labels returns empty on a fresh database.
#[tokio::test]
async fn list_labels_empty_db() {
    let (svc, _dir) = test_service();

    let resp = svc
        .list_labels(Request::new(schema::ListLabelsRequest {}))
        .await
        .expect("list_labels should succeed");

    assert_eq!(resp.into_inner().labels.len(), 0);
}

/// list_labels returns labels of nodes that exist in the database.
#[tokio::test]
async fn list_labels_returns_existing_labels() {
    let (svc, _dir) = test_service();

    // Insert nodes with two different labels
    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (n:Person {name: 'Alice'})")
            .expect("create");
        db.execute_cypher("CREATE (n:City {name: 'Kyiv'})")
            .expect("create");
    }

    let labels = svc
        .list_labels(Request::new(schema::ListLabelsRequest {}))
        .await
        .expect("list_labels should succeed")
        .into_inner()
        .labels;

    // Labels the stored nodes carry without a definition are listed as such.
    for name in ["City", "Person"] {
        let found = labels
            .iter()
            .find(|l| l.name == name)
            .unwrap_or_else(|| panic!("{name} missing from {labels:?}"));
        assert!(!found.declared, "{name} has no definition");
    }
}

/// list_edge_types returns empty on a fresh database.
#[tokio::test]
async fn list_edge_types_empty_db() {
    let (svc, _dir) = test_service();

    let resp = svc
        .list_edge_types(Request::new(schema::ListEdgeTypesRequest {}))
        .await
        .expect("list_edge_types should succeed");

    assert_eq!(resp.into_inner().edge_types.len(), 0);
}

/// list_edge_types returns edge types that exist in the database.
#[tokio::test]
async fn list_edge_types_returns_existing_types() {
    let (svc, _dir) = test_service();

    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (a:P {name: 'A'})-[:KNOWS]->(b:P {name: 'B'})")
            .expect("create");
        db.execute_cypher("CREATE (c:P {name: 'C'})-[:LIKES]->(d:P {name: 'D'})")
            .expect("create");
    }

    let edge_types: Vec<String> = svc
        .list_edge_types(Request::new(schema::ListEdgeTypesRequest {}))
        .await
        .expect("list_edge_types should succeed")
        .into_inner()
        .edge_types
        .into_iter()
        .map(|et| et.name)
        .collect();

    assert!(
        edge_types.contains(&"KNOWS".to_string()),
        "should contain KNOWS, got: {edge_types:?}"
    );
    assert!(
        edge_types.contains(&"LIKES".to_string()),
        "should contain LIKES, got: {edge_types:?}"
    );
}

// ── create_label / create_edge_type persist the schema ──

/// A property type, schema mode or constraint kind the protocol does not
/// define is refused as an invalid field, not stored as some default: a
/// client sending a newer enum value would otherwise get a schema it never
/// asked for, silently.
#[tokio::test]
async fn unknown_enum_values_are_refused_not_defaulted() {
    let (svc, _dir) = test_service();
    let invalid = |status: Status, field: &str| {
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
        assert_eq!(reason(&status), "INVALID_FIELD");
        let violations = status
            .get_details_bad_request()
            .expect("BadRequest")
            .field_violations;
        assert_eq!(violations[0].field, field, "{violations:?}");
    };

    let mut unknown_type = prop("p", schema::ScalarType::String, false);
    unknown_type.r#type = Some(schema::PropertyType {
        r#type: Some(schema::property_type::Type::Scalar(42)),
    });
    invalid(
        svc.create_label(Request::new(label(
            "Unknown",
            vec![unknown_type.clone()],
            schema::SchemaMode::Strict,
        )))
        .await
        .expect_err("an unknown property type must be refused"),
        "properties[0].type",
    );

    let mut unknown_mode = label("Unknown", vec![], schema::SchemaMode::Strict);
    unknown_mode.schema_mode = 42;
    invalid(
        svc.create_label(Request::new(unknown_mode))
            .await
            .expect_err("an unknown schema mode must be refused"),
        "schema_mode",
    );

    invalid(
        svc.create_edge_type(Request::new(schema::CreateEdgeTypeRequest {
            name: "UNKNOWN".to_string(),
            properties: vec![unknown_type],
            temporal: false,
            discriminator: String::new(),
        }))
        .await
        .expect_err("an unknown edge property type must be refused"),
        "properties[0].type",
    );

    let mut unknown_kind = unique("u", "User", "email");
    unknown_kind.kind = 42;
    invalid(
        svc.create_constraint(Request::new(unknown_kind))
            .await
            .expect_err("an unknown constraint kind must be refused"),
        "kind",
    );

    // None of the refused requests reached the catalog.
    let db = svc.database.read();
    assert!(db.label_schemas().expect("labels").is_empty());
    assert!(db.constraints().expect("constraints").is_empty());
}

/// A property without a type, a repeated name, or a constraint whose shape
/// its kind does not allow is refused before anything is written.
#[tokio::test]
async fn malformed_definitions_and_constraints_are_refused() {
    let (svc, _dir) = test_service();

    let mut untyped = prop("p", schema::ScalarType::String, false);
    untyped.r#type = None;
    let status = svc
        .create_label(Request::new(label(
            "L",
            vec![untyped],
            schema::SchemaMode::Strict,
        )))
        .await
        .expect_err("a property needs a type");
    assert_eq!(status.code(), tonic::Code::InvalidArgument);

    let status = svc
        .create_label(Request::new(label(
            "L",
            vec![
                prop("p", schema::ScalarType::String, false),
                prop("p", schema::ScalarType::Int64, false),
            ],
            schema::SchemaMode::Strict,
        )))
        .await
        .expect_err("a property is declared once");
    assert_eq!(status.code(), tonic::Code::InvalidArgument);

    let mut two_not_null = unique("n", "L", "a");
    two_not_null.kind = schema::ConstraintKind::NotNull as i32;
    two_not_null.properties.push("b".into());
    let status = svc
        .create_constraint(Request::new(two_not_null))
        .await
        .expect_err("NOT NULL constrains one property");
    assert_eq!(status.code(), tonic::Code::InvalidArgument);

    let mut typed_unique = unique("u", "L", "a");
    typed_unique.property_type = scalar(schema::ScalarType::String);
    let status = svc
        .create_constraint(Request::new(typed_unique))
        .await
        .expect_err("only a type constraint names a type");
    assert_eq!(status.code(), tonic::Code::InvalidArgument);

    let db = svc.database.read();
    assert!(db.label_schemas().expect("labels").is_empty());
    assert!(db.constraints().expect("constraints").is_empty());
}

/// create_label persists the type facts, answers with them as stored, and
/// declares no constraint.
#[tokio::test]
async fn create_label_persists_schema() {
    let (svc, _dir) = test_service();

    let mut slug = prop("slug", schema::ScalarType::String, false);
    slug.default_value = Some(value_to_proto_pub(&Value::String("untitled".into())));
    let label = svc
        .create_label(Request::new(label(
            "Article",
            vec![prop("title", schema::ScalarType::String, true), slug],
            // Unspecified means STRICT.
            schema::SchemaMode::Unspecified,
        )))
        .await
        .expect("create_label should succeed")
        .into_inner();

    assert_eq!(label.name, "Article");
    assert!(label.declared);
    assert_eq!(label.schema_mode, schema::SchemaMode::Strict as i32);
    assert!(label.schema_revision > 0);
    assert_eq!(label.properties.len(), 2);
    let slug = label
        .properties
        .iter()
        .find(|p| p.name == "slug")
        .expect("slug");
    assert_eq!(
        slug.default_value.as_ref().map(proto_to_value_pub),
        Some(Value::String("untitled".into()))
    );

    let db = svc.database.read();
    let stored = db
        .label_schemas()
        .expect("labels")
        .into_iter()
        .find(|s| s.name == "Article")
        .expect("schema must be persisted");
    assert_eq!(stored.properties.len(), 2);
    assert!(stored.get_property("title").is_some_and(|p| p.not_null));
    assert!(
        stored.constraints().is_empty(),
        "a type declares no constraint"
    );
}

/// Defining a label that already has a definition is refused as an existing
/// catalog object, and the first definition stays.
#[tokio::test]
async fn a_label_is_defined_once() {
    let (svc, _dir) = test_service();
    svc.create_label(Request::new(label(
        "Article",
        vec![prop("title", schema::ScalarType::String, true)],
        schema::SchemaMode::Strict,
    )))
    .await
    .expect("first definition");

    let status = svc
        .create_label(Request::new(label(
            "Article",
            vec![],
            schema::SchemaMode::Flexible,
        )))
        .await
        .expect_err("a second definition is refused");
    assert_eq!(status.code(), tonic::Code::AlreadyExists, "{status:?}");
    assert_eq!(reason(&status), "CATALOG_OBJECT_EXISTS");
    let resource = status.get_details_resource_info().expect("ResourceInfo");
    assert_eq!(
        (
            resource.resource_type.as_str(),
            resource.resource_name.as_str()
        ),
        ("label", "Article")
    );
    let stored = svc
        .database
        .read()
        .label_schemas()
        .expect("labels")
        .into_iter()
        .find(|s| s.name == "Article")
        .expect("Article");
    assert_eq!(stored.mode, SchemaMode::Strict);
}

/// A CreateConstraint whose wait runs out first answers VALIDATING with its
/// build, which goes on server-side: GetIndexBuild waits for its outcome and
/// sees it published, the listed constraint is then ACTIVE, a cancellation
/// afterwards keeps the published outcome, and an operation no build has, or
/// an absent one, is refused.
#[tokio::test]
async fn a_constraint_build_outlives_its_call_and_is_inspected_by_operation() {
    let (svc, _dir) = test_service();
    svc.database
        .write()
        .execute_cypher("CREATE (:Customer {email: 'a@x'})")
        .expect("seed");
    let older = svc.database.read().begin_transaction();

    let mut request = unique("customer_email", "Customer", "email");
    request.wait = Some(prost_types::Duration::default());
    let created = svc
        .create_constraint(Request::new(request))
        .await
        .expect("admitted")
        .into_inner();
    assert_eq!(created.state, schema::ConstraintState::Validating as i32);
    let build = created.build.expect("its build");
    assert_ne!(build.state, schema::IndexBuildState::Published as i32);
    let operation = build.operation;

    svc.database
        .read()
        .rollback_transaction(older)
        .expect("end the older transaction");
    let done = svc
        .get_index_build(Request::new(schema::GetIndexBuildRequest {
            operation: Some(operation),
            wait: Some(prost_types::Duration {
                seconds: 20,
                nanos: 0,
            }),
        }))
        .await
        .expect("inspect")
        .into_inner();
    assert_eq!(done.state, schema::IndexBuildState::Published as i32);
    let listed = svc
        .list_constraints(Request::new(schema::ListConstraintsRequest {}))
        .await
        .expect("list")
        .into_inner()
        .constraints;
    assert_eq!(listed[0].state, schema::ConstraintState::Active as i32);
    assert_eq!(
        listed[0].build.as_ref().map(|b| b.operation),
        Some(operation)
    );
    let builds = svc
        .list_index_builds(Request::new(schema::ListIndexBuildsRequest {}))
        .await
        .expect("list builds")
        .into_inner()
        .builds;
    assert!(builds.iter().any(|b| b.operation == operation));

    let kept = svc
        .cancel_index_build(Request::new(schema::CancelIndexBuildRequest {
            operation: Some(operation),
        }))
        .await
        .expect("cancel")
        .into_inner();
    assert_eq!(kept.state, schema::IndexBuildState::Published as i32);

    let missing = svc
        .get_index_build(Request::new(schema::GetIndexBuildRequest {
            operation: Some(operation + 1000),
            wait: None,
        }))
        .await
        .expect_err("no such build");
    assert_eq!(missing.code(), tonic::Code::NotFound);
    assert_eq!(reason(&missing), "CATALOG_OBJECT_NOT_FOUND");
    let absent = svc
        .cancel_index_build(Request::new(schema::CancelIndexBuildRequest {
            operation: None,
        }))
        .await
        .expect_err("no operation");
    assert_eq!(absent.code(), tonic::Code::InvalidArgument);
    let mut negative = unique("other", "Customer", "email");
    negative.wait = Some(prost_types::Duration {
        seconds: -1,
        nanos: 0,
    });
    let refused = svc
        .create_constraint(Request::new(negative))
        .await
        .expect_err("a negative wait");
    assert_eq!(refused.code(), tonic::Code::InvalidArgument);
}

/// A CreateConstraint with on_duplicate_rename repairs the stored
/// duplicate: the constraint is ACTIVE, its build reports the property and
/// one repair, and ListIndexBuildRepairs names the node and both values.
#[tokio::test]
async fn a_constraint_build_repairs_a_duplicate_and_lists_it() {
    let (svc, _dir) = test_service();
    svc.database
        .write()
        .execute_cypher("CREATE (:Customer {email: 'same@x'}), (:Customer {email: 'same@x'})")
        .expect("seed");

    let mut request = unique("customer_email", "Customer", "email");
    request.on_duplicate_rename = "email".into();
    let created = svc
        .create_constraint(Request::new(request))
        .await
        .expect("repaired")
        .into_inner();
    assert_eq!(created.state, schema::ConstraintState::Active as i32);
    let build = created.build.expect("its build");
    assert_eq!(build.rename_property, "email");
    assert_eq!(build.repaired, 1);

    let repairs = svc
        .list_index_build_repairs(Request::new(schema::ListIndexBuildRepairsRequest {
            operation: Some(build.operation),
        }))
        .await
        .expect("repairs")
        .into_inner()
        .repairs;
    assert_eq!(repairs.len(), 1);
    assert_eq!(repairs[0].property, "email");
    assert_eq!(repairs[0].old_value, "same@x");
    assert!(repairs[0].new_value.starts_with("same@x_"));
    assert!(!repairs[0].element_id.is_empty());

    let mut wrong = unique("other", "Customer", "email");
    wrong.on_duplicate_rename = "name".into();
    let refused = svc
        .create_constraint(Request::new(wrong))
        .await
        .expect_err("not a covered property");
    assert_eq!(reason(&refused), "CATALOG_CHANGE_REFUSED");
}

/// A uniqueness declared with a scope holds only among the nodes it admits:
/// it is listed with the scope, enforced inside it and not outside, and a
/// scope on a constraint other than UNIQUE is refused.
#[tokio::test]
async fn a_scoped_uniqueness_is_declared_listed_and_enforced_in_scope() {
    let (svc, _dir) = test_service();
    let scope = schema::ConstraintScope {
        property: "active".into(),
        test: Some(schema::constraint_scope::Test::EqualsBool(true)),
    };
    let mut request = unique("active_email", "Customer", "email");
    request.scope = Some(scope.clone());
    let created = svc
        .create_constraint(Request::new(request))
        .await
        .expect("scoped uniqueness")
        .into_inner();
    assert_eq!(created.scope, Some(scope));

    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (:Customer {email: 'a@x', active: true})")
            .expect("first in scope");
        db.execute_cypher("CREATE (:Customer {email: 'a@x', active: false})")
            .expect("out of scope");
        assert!(
            db.execute_cypher("CREATE (:Customer {email: 'a@x', active: true})")
                .is_err(),
            "a second holder in scope is refused"
        );
    }

    let mut not_null = unique("c", "Customer", "email");
    not_null.kind = schema::ConstraintKind::NotNull as i32;
    not_null.scope = Some(schema::ConstraintScope {
        property: "active".into(),
        test: Some(schema::constraint_scope::Test::Present(())),
    });
    let refused = svc
        .create_constraint(Request::new(not_null))
        .await
        .expect_err("only a uniqueness takes a scope");
    assert_eq!(reason(&refused), "CATALOG_CHANGE_REFUSED");
}

/// A uniqueness constraint is created apart from the type: it answers with
/// its state and the index it owns, is listed, refuses a duplicate value,
/// and dropping it leaves the type as it was.
#[tokio::test]
async fn a_uniqueness_constraint_is_created_listed_enforced_and_dropped() {
    let (svc, _dir) = test_service();
    svc.create_label(Request::new(label(
        "Customer",
        vec![prop("email", schema::ScalarType::String, false)],
        schema::SchemaMode::Strict,
    )))
    .await
    .expect("create_label");

    let created = svc
        .create_constraint(Request::new(unique("customer_email", "Customer", "email")))
        .await
        .expect("create_constraint")
        .into_inner();
    assert_eq!(created.kind, schema::ConstraintKind::Unique as i32);
    assert_eq!(created.state, schema::ConstraintState::Active as i32);
    assert_eq!(created.backing_index, "customer_email");
    assert_eq!(created.property_type, None);
    let build = created.build.clone().expect("the backing index's build");
    assert_eq!(build.state, schema::IndexBuildState::Published as i32);
    assert_eq!(build.index, "customer_email");
    assert_eq!(build.label, "Customer");

    let listed = svc
        .list_constraints(Request::new(schema::ListConstraintsRequest {}))
        .await
        .expect("list")
        .into_inner()
        .constraints;
    assert_eq!(listed, vec![created]);

    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (c:Customer {email: 'x@test.com'})")
            .expect("first create");
        assert!(
            db.execute_cypher("CREATE (c:Customer {email: 'x@test.com'})")
                .is_err(),
            "duplicate email must be rejected by the constraint"
        );
    }

    let status = svc
        .create_constraint(Request::new(unique("customer_email", "Customer", "email")))
        .await
        .expect_err("the name is taken");
    assert_eq!(status.code(), tonic::Code::AlreadyExists);
    assert_eq!(reason(&status), "CATALOG_OBJECT_EXISTS");

    svc.drop_constraint(Request::new(schema::DropConstraintRequest {
        name: "customer_email".into(),
        if_exists: false,
    }))
    .await
    .expect("drop");
    let status = svc
        .drop_constraint(Request::new(schema::DropConstraintRequest {
            name: "customer_email".into(),
            if_exists: false,
        }))
        .await
        .expect_err("already dropped");
    assert_eq!(status.code(), tonic::Code::NotFound);
    assert_eq!(reason(&status), "CATALOG_OBJECT_NOT_FOUND");
    svc.drop_constraint(Request::new(schema::DropConstraintRequest {
        name: "customer_email".into(),
        if_exists: true,
    }))
    .await
    .expect("IF EXISTS");
    let labels = svc
        .list_labels(Request::new(schema::ListLabelsRequest {}))
        .await
        .expect("labels")
        .into_inner()
        .labels;
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0].properties.len(), 1, "the type is unchanged");
}

/// A type constraint answers with the type it requires; one that contradicts
/// the declared type is refused by the catalog's state.
#[tokio::test]
async fn a_type_constraint_names_its_type_and_must_fit_the_declaration() {
    let (svc, _dir) = test_service();
    svc.create_label(Request::new(label(
        "Doc",
        vec![prop("size", schema::ScalarType::Int64, false)],
        schema::SchemaMode::Flexible,
    )))
    .await
    .expect("create_label");

    let mut typed = unique("doc_size_type", "Doc", "size");
    typed.kind = schema::ConstraintKind::PropertyType as i32;
    typed.property_type = scalar(schema::ScalarType::Int64);
    let created = svc
        .create_constraint(Request::new(typed.clone()))
        .await
        .expect("a fitting type constraint")
        .into_inner();
    assert_eq!(created.property_type, scalar(schema::ScalarType::Int64));
    assert_eq!(created.backing_index, "", "a type constraint owns no index");

    typed.name = "doc_size_string".into();
    typed.property_type = scalar(schema::ScalarType::String);
    let status = svc
        .create_constraint(Request::new(typed))
        .await
        .expect_err("STRING contradicts the declared INT64");
    assert_eq!(status.code(), tonic::Code::FailedPrecondition, "{status:?}");
    assert_eq!(reason(&status), "CATALOG_CHANGE_REFUSED");
}

/// Structured property types round-trip: vector dimensions and metric, and
/// array element types, as declared.
#[tokio::test]
async fn structured_property_types_round_trip() {
    let (svc, _dir) = test_service();
    let vector = schema::PropertyType {
        r#type: Some(schema::property_type::Type::Vector(schema::VectorType {
            dimensions: 384,
            metric: crate::proto::v1::query::DistanceMetric::L2 as i32,
        })),
    };
    let tags = schema::PropertyType {
        r#type: Some(schema::property_type::Type::Array(Box::new(
            schema::ArrayType {
                element: scalar(schema::ScalarType::String).map(Box::new),
            },
        ))),
    };
    let mut emb = prop("emb", schema::ScalarType::String, false);
    emb.r#type = Some(vector.clone());
    let mut tag_list = prop("tags", schema::ScalarType::String, false);
    tag_list.r#type = Some(tags.clone());
    let created = svc
        .create_label(Request::new(label(
            "Item",
            vec![emb, tag_list],
            schema::SchemaMode::Validated,
        )))
        .await
        .expect("create_label")
        .into_inner();
    let type_of = |name: &str| {
        created
            .properties
            .iter()
            .find(|p| p.name == name)
            .and_then(|p| p.r#type.clone())
    };
    assert_eq!(type_of("emb"), Some(vector));
    assert_eq!(type_of("tags"), Some(tags));
}

/// A uniqueness constraint created after the type is enforced on the next
/// CREATE.
#[tokio::test]
async fn create_label_unique_property_enforces_constraint() {
    let (svc, _dir) = test_service();

    svc.create_label(Request::new(label(
        "Customer",
        vec![prop("email", schema::ScalarType::String, false)],
        // Unspecified means STRICT.
        schema::SchemaMode::Unspecified,
    )))
    .await
    .expect("create_label");
    svc.create_constraint(Request::new(unique("customer_email", "Customer", "email")))
        .await
        .expect("create_constraint");

    // First node — should succeed.
    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (c:Customer {email: 'x@test.com'})")
            .expect("first create");
    }

    // Second node with same email — must fail.
    {
        let mut db = svc.database.write();
        let result = db.execute_cypher("CREATE (c:Customer {email: 'x@test.com'})");
        assert!(
            result.is_err(),
            "duplicate email must be rejected by unique constraint"
        );
    }
}

/// MERGE on an existing node with a unique property must match the node
/// (ON MATCH), NOT fall through to CREATE and throw "unique constraint violated".
///
/// Regression test for the gRPC repro:
///   CreateLabel + CreateConstraint (unique id) → CREATE node → MERGE same id
#[tokio::test]
async fn merge_on_existing_unique_node_does_not_error() {
    let (svc, _dir) = test_service();

    svc.create_label(Request::new(label(
        "TestNode",
        vec![prop("id", schema::ScalarType::String, true)],
        schema::SchemaMode::Strict,
    )))
    .await
    .expect("create_label");
    svc.create_constraint(Request::new(unique("testnode_id", "TestNode", "id")))
        .await
        .expect("create_constraint");

    // Create initial node.
    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (n:TestNode {id: 'x1'})")
            .expect("create node");
    }

    // MERGE on existing id — must find the node via NodeScan and apply ON MATCH.
    // Bug: NodeScan misses the node → MERGE falls through to CREATE →
    //      B-tree unique index: "unique constraint violated on index testnode_id".
    {
        let mut db = svc.database.write();
        let result =
            db.execute_cypher("MERGE (n:TestNode {id: 'x1'}) ON MATCH SET n.id = 'x1' RETURN n.id");
        assert!(
            result.is_ok(),
            "MERGE on existing unique node must succeed: {:?}",
            result.err()
        );
        let rows = result.unwrap();
        assert_eq!(rows.len(), 1, "MERGE must match exactly one node");
    }
}

/// create_edge_type persists the definition and lists it with its type facts;
/// a second definition of the type is refused.
#[tokio::test]
async fn create_edge_type_persists_schema() {
    let (svc, _dir) = test_service();

    let et = svc
        .create_edge_type(Request::new(schema::CreateEdgeTypeRequest {
            name: "FOLLOWS".to_string(),
            properties: vec![prop("since", schema::ScalarType::Timestamp, true)],
            temporal: true,
            discriminator: String::new(),
        }))
        .await
        .expect("create_edge_type should succeed")
        .into_inner();
    assert_eq!(et.name, "FOLLOWS");
    assert!(et.declared && et.temporal);
    assert!(et.schema_revision > 0);
    assert!(et.properties[0].required);
    assert_eq!(
        et.discriminator, "valid_from",
        "a temporal type naming none is start-identified"
    );

    let listed = svc
        .list_edge_types(Request::new(schema::ListEdgeTypesRequest {}))
        .await
        .expect("list")
        .into_inner()
        .edge_types;
    assert_eq!(listed, vec![et]);

    let status = svc
        .create_edge_type(Request::new(schema::CreateEdgeTypeRequest {
            name: "FOLLOWS".to_string(),
            properties: vec![],
            temporal: false,
            discriminator: String::new(),
        }))
        .await
        .expect_err("defined once");
    assert_eq!(status.code(), tonic::Code::AlreadyExists);
}

/// An edge type names the property that identifies its edges, temporal or
/// not, and lists it as given; a discriminator the type does not declare,
/// may hold null or has a type that cannot identify an edge is refused
/// with the field named and nothing created.
#[tokio::test]
async fn create_edge_type_declares_its_discriminator() {
    let (svc, _dir) = test_service();
    let invalid = |status: Status, field: &str| {
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
        assert_eq!(reason(&status), "INVALID_FIELD");
        let violations = status
            .get_details_bad_request()
            .expect("BadRequest")
            .field_violations;
        assert_eq!(violations[0].field, field, "{violations:?}");
    };
    let create =
        |name: &str, properties, temporal, discriminator: &str| schema::CreateEdgeTypeRequest {
            name: name.to_string(),
            properties,
            temporal,
            discriminator: discriminator.to_string(),
        };

    let knows = svc
        .create_edge_type(Request::new(create(
            "KNOWS",
            vec![prop("context", schema::ScalarType::String, true)],
            false,
            "context",
        )))
        .await
        .expect("categorical discriminator")
        .into_inner();
    assert_eq!(knows.discriminator, "context");
    assert!(!knows.temporal);
    let asserts = svc
        .create_edge_type(Request::new(create(
            "ASSERTS",
            vec![prop("assertion_key", schema::ScalarType::String, true)],
            true,
            "assertion_key",
        )))
        .await
        .expect("temporal with an independent discriminator")
        .into_inner();
    assert_eq!(asserts.discriminator, "assertion_key");
    assert!(asserts.temporal);

    for (name, properties, discriminator) in [
        (
            "UNDECLARED",
            vec![prop("context", schema::ScalarType::String, true)],
            "missing",
        ),
        (
            "NULLABLE",
            vec![prop("context", schema::ScalarType::String, false)],
            "context",
        ),
        (
            "MAPPED",
            vec![prop("context", schema::ScalarType::Map, true)],
            "context",
        ),
    ] {
        invalid(
            svc.create_edge_type(Request::new(create(name, properties, false, discriminator)))
                .await
                .expect_err(name),
            "discriminator",
        );
    }
    let listed: Vec<String> = svc
        .list_edge_types(Request::new(schema::ListEdgeTypesRequest {}))
        .await
        .expect("list")
        .into_inner()
        .edge_types
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert_eq!(
        listed,
        ["ASSERTS", "KNOWS"],
        "the refused ones left nothing"
    );
}

/// list_labels returns schema properties for declared labels.
#[tokio::test]
async fn list_labels_returns_schema_properties() {
    let (svc, _dir) = test_service();

    svc.create_label(Request::new(label(
        "Post",
        vec![
            prop("title", schema::ScalarType::String, true),
            prop("views", schema::ScalarType::Int64, false),
        ],
        // Unspecified means STRICT.
        schema::SchemaMode::Unspecified,
    )))
    .await
    .expect("create_label");

    let labels = svc
        .list_labels(Request::new(schema::ListLabelsRequest {}))
        .await
        .expect("list_labels")
        .into_inner()
        .labels;

    let post_label = labels
        .iter()
        .find(|l| l.name == "Post")
        .expect("Post label must be in list");

    assert_eq!(
        post_label.properties.len(),
        2,
        "Post label must expose 2 declared properties"
    );
    assert!(
        post_label
            .properties
            .iter()
            .any(|p| p.name == "title" && p.required),
        "title must be required"
    );
}

// ── ComputedPropertyDefinition via gRPC ──────────────────────────────────

/// create_label with TTL ComputedPropertyDefinition → schema persisted with
/// ComputedSpec::Ttl; TtlReaper deletes an expired node.
///
/// Regression: ComputedSpec must reach storage via the gRPC API path, not
/// only via direct Rust API (PropertyDef::computed). This test validates
/// the full proto → SchemaService → LabelSchema → storage → TtlReaper chain.
#[tokio::test]
async fn create_label_with_ttl_computed_property_reaper_deletes_expired_node() {
    let (svc, _dir) = test_service();

    // Declare label with a TIMESTAMP anchor and a TTL COMPUTED property (60s,
    // Node scope). The TtlReaper background thread is not used here — we call
    // reap_computed_ttl() directly to avoid real-time sleeping.
    let session = |name: &str| {
        let mut request = label(
            name,
            vec![prop("started_at", schema::ScalarType::Timestamp, true)],
            // The test exercises the TTL reaper, not schema enforcement.
            schema::SchemaMode::Flexible,
        );
        request.computed_properties = vec![schema::ComputedPropertyDefinition {
            name: "_ttl".to_string(),
            computed_type: schema::ComputedType::Ttl as i32,
            // A TTL does not use the formula.
            formula_type: schema::DecayFormulaType::Unspecified as i32,
            duration_secs: 60, // 60 seconds lifetime
            anchor_field: "started_at".to_string(),
            scope: schema::TtlScopeType::Node as i32,
            ..Default::default()
        }];
        request
    };
    svc.create_label(Request::new(session("Session")))
        .await
        .expect("create_label with TTL should succeed");

    // The response carries the computed property as stored.
    let label = svc
        .create_label(Request::new(session("Session2")))
        .await
        .expect("create_label should succeed")
        .into_inner();
    assert_eq!(
        label.computed_properties.len(),
        1,
        "response must carry computed_properties"
    );
    assert_eq!(label.computed_properties[0].name, "_ttl");
    assert_eq!(
        label.computed_properties[0].computed_type,
        schema::ComputedType::Ttl as i32
    );
    assert_eq!(label.computed_properties[0].duration_secs, 60);
    assert_eq!(label.computed_properties[0].anchor_field, "started_at");

    // Verify schema was persisted correctly: LabelSchema must contain ComputedSpec::Ttl.
    {
        use coordinode_core::schema::computed::{ComputedSpec, TtlScope};
        use coordinode_core::schema::definition::PropertyType;
        use coordinode_storage::engine::partition::Partition;

        let db = svc.database.write();
        let key = coordinode_core::schema::definition::encode_label_schema_key("Session", 1);
        let bytes = db
            .engine()
            .get(Partition::Schema, &key)
            .expect("storage get")
            .expect("Session schema must be persisted");
        let schema = coordinode_core::schema::definition::LabelSchema::from_msgpack(&bytes)
            .expect("deserialize");

        let ttl_prop = schema
            .get_property("_ttl")
            .expect("_ttl property must exist");
        assert!(
            matches!(
                &ttl_prop.property_type,
                PropertyType::Computed(ComputedSpec::Ttl {
                    duration_secs: 60,
                    scope: TtlScope::Node,
                    ..
                })
            ),
            "ComputedSpec::Ttl must be persisted with correct parameters, got: {:?}",
            ttl_prop.property_type
        );
    }

    // Now create two nodes: one expired (started_at far in the past), one fresh.
    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;
    let expired_us = now_us - 120 * 1_000_000; // 120s ago → past 60s TTL

    {
        let mut db = svc.database.write();
        db.execute_cypher(&format!("CREATE (s:Session {{started_at: {expired_us}}})"))
            .expect("create expired session");
        db.execute_cypher(&format!("CREATE (s:Session {{started_at: {now_us}}})"))
            .expect("create fresh session");
    }

    // Verify both nodes exist before reaping.
    {
        let mut db = svc.database.write();
        let rows = db
            .execute_cypher("MATCH (s:Session) RETURN s.started_at AS ts")
            .expect("match before reap");
        assert_eq!(rows.len(), 2, "must have 2 Session nodes before reap");
    }

    // Run TtlReaper directly — no background thread, no waiting.
    let result = {
        let db = svc.database.write();
        coordinode_query::index::ttl_reaper::reap_computed_ttl(&db.engine_shared(), 1, 1000)
    };

    assert_eq!(
        result.nodes_deleted, 1,
        "reaper must delete exactly 1 expired Session node"
    );

    // Verify only the fresh node remains.
    {
        let mut db = svc.database.write();
        let rows = db
            .execute_cypher("MATCH (s:Session) RETURN s.started_at AS ts")
            .expect("match after reap");
        assert_eq!(rows.len(), 1, "exactly 1 Session node must survive");
    }
}

/// create_label with DECAY ComputedPropertyDefinition → schema persisted,
/// list_labels returns computed_properties.
#[tokio::test]
async fn create_label_with_decay_computed_property_list_labels_returns_it() {
    let (svc, _dir) = test_service();

    let mut request = label(
        "Article",
        vec![prop("published_at", schema::ScalarType::Timestamp, true)],
        // Unspecified means STRICT.
        schema::SchemaMode::Unspecified,
    );
    request.computed_properties = vec![schema::ComputedPropertyDefinition {
        name: "relevance".to_string(),
        computed_type: schema::ComputedType::Decay as i32,
        formula_type: schema::DecayFormulaType::Linear as i32,
        initial: 1.0,
        target: 0.0,
        duration_secs: 604800, // 7 days
        anchor_field: "published_at".to_string(),
        ..Default::default()
    }];
    svc.create_label(Request::new(request))
        .await
        .expect("create_label with DECAY should succeed");

    // list_labels must return the computed property back to the caller.
    let labels = svc
        .list_labels(Request::new(schema::ListLabelsRequest {}))
        .await
        .expect("list_labels")
        .into_inner()
        .labels;

    let article = labels
        .iter()
        .find(|l| l.name == "Article")
        .expect("Article label must be in list");

    // Regular properties are separated from computed properties.
    assert_eq!(
        article.properties.len(),
        1,
        "Article must have 1 regular property (published_at)"
    );
    assert_eq!(
        article.computed_properties.len(),
        1,
        "Article must have 1 computed property (relevance)"
    );

    let cp = &article.computed_properties[0];
    assert_eq!(cp.name, "relevance");
    assert_eq!(cp.computed_type, schema::ComputedType::Decay as i32);
    assert_eq!(cp.formula_type, schema::DecayFormulaType::Linear as i32);
    assert!((cp.initial - 1.0).abs() < f64::EPSILON);
    assert!((cp.target - 0.0).abs() < f64::EPSILON);
    assert_eq!(cp.duration_secs, 604800);
    assert_eq!(cp.anchor_field, "published_at");
}

/// proto_to_computed_spec rejects COMPUTED_TYPE_UNSPECIFIED.
#[tokio::test]
async fn create_label_computed_type_unspecified_returns_error() {
    let (svc, _dir) = test_service();

    let mut request = label("Bad", vec![], schema::SchemaMode::Unspecified);
    request.computed_properties = vec![schema::ComputedPropertyDefinition {
        name: "broken".to_string(),
        computed_type: schema::ComputedType::Unspecified as i32,
        duration_secs: 60,
        anchor_field: "ts".to_string(),
        ..Default::default()
    }];
    let status = svc
        .create_label(Request::new(request))
        .await
        .expect_err("UNSPECIFIED computed_type must be rejected");
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
}

/// proto_to_computed_spec rejects empty anchor_field.
#[tokio::test]
async fn create_label_computed_empty_anchor_field_returns_error() {
    let (svc, _dir) = test_service();

    let mut request = label("Bad2", vec![], schema::SchemaMode::Unspecified);
    request.computed_properties = vec![schema::ComputedPropertyDefinition {
        name: "ttl".to_string(),
        computed_type: schema::ComputedType::Ttl as i32,
        duration_secs: 60,
        anchor_field: String::new(), // empty: must be rejected
        ..Default::default()
    }];
    let status = svc
        .create_label(Request::new(request))
        .await
        .expect_err("empty anchor_field must be rejected");
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    let violations = status
        .get_details_bad_request()
        .expect("BadRequest")
        .field_violations;
    assert_eq!(violations[0].field, "computed_properties[0].anchor_field");
}

// ── SchemaMode via gRPC: CREATE/SET enforcement ─────────────────────────

/// create_label with schema_mode=STRICT persists the mode; SET of an
/// unknown property on a STRICT label is rejected.
///
/// Regression: schema_mode field must flow from proto → LabelSchema → executor
/// enforcement. STRICT = no undeclared properties allowed at write time.
#[tokio::test]
async fn strict_mode_set_unknown_property_rejected() {
    let (svc, _dir) = test_service();

    // Declare User label with schema_mode=STRICT, only `name` declared.
    svc.create_label(Request::new(label(
        "User",
        vec![prop("name", schema::ScalarType::String, false)],
        schema::SchemaMode::Strict,
    )))
    .await
    .expect("create_label should succeed");

    // Verify schema was persisted with STRICT mode.
    {
        use coordinode_core::schema::definition::SchemaMode;
        use coordinode_storage::engine::partition::Partition;

        let db = svc.database.write();
        let key = coordinode_core::schema::definition::encode_label_schema_key("User", 1);
        let bytes = db
            .engine()
            .get(Partition::Schema, &key)
            .expect("storage get")
            .expect("User schema must be persisted");
        let schema = coordinode_core::schema::definition::LabelSchema::from_msgpack(&bytes)
            .expect("deserialize");
        assert_eq!(
            schema.mode,
            SchemaMode::Strict,
            "schema mode must be Strict"
        );
    }

    // CREATE a User node with the declared property.
    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (u:User {name: 'Alice'})")
            .expect("CREATE with declared property must succeed on STRICT label");
    }

    // SET an unknown property on the STRICT label — must be rejected.
    {
        let mut db = svc.database.write();
        let result =
            db.execute_cypher("MATCH (u:User {name: 'Alice'}) SET u.unknown_field = 'boom'");
        assert!(
            result.is_err(),
            "SET unknown property must be rejected on STRICT label, got: {result:?}"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("unknown_field") || err_msg.contains("strict"),
            "error must mention the unknown property or strict mode, got: {err_msg}"
        );
    }
}

/// create_label with schema_mode=FLEXIBLE — SET of any property is allowed.
///
/// Regression: FLEXIBLE mode must pass through all properties without enforcement.
#[tokio::test]
async fn flexible_mode_set_unknown_property_allowed() {
    let (svc, _dir) = test_service();

    // Declare Device label with schema_mode=FLEXIBLE, only `id` declared.
    svc.create_label(Request::new(label(
        "Device",
        vec![prop("id", schema::ScalarType::String, false)],
        schema::SchemaMode::Flexible,
    )))
    .await
    .expect("create_label should succeed");

    // CREATE a Device node.
    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (d:Device {id: 'd1'})")
            .expect("CREATE should succeed");
    }

    // SET an undeclared property — must succeed on FLEXIBLE label.
    {
        let mut db = svc.database.write();
        db.execute_cypher("MATCH (d:Device {id: 'd1'}) SET d.firmware_version = '1.2.3'")
            .expect("SET undeclared property must succeed on FLEXIBLE label");
    }
}

/// create_label response includes schema_mode field reflecting the
/// requested mode; list_labels echoes the persisted mode.
#[tokio::test]
async fn schema_mode_echoed_in_response_and_list() {
    let (svc, _dir) = test_service();

    // Create with VALIDATED mode.
    let created = svc
        .create_label(Request::new(label(
            "Event",
            vec![prop("ts", schema::ScalarType::Timestamp, false)],
            schema::SchemaMode::Validated,
        )))
        .await
        .expect("create_label should succeed")
        .into_inner();
    assert_eq!(
        created.schema_mode,
        schema::SchemaMode::Validated as i32,
        "create_label response must carry schema_mode=VALIDATED"
    );

    // list_labels must return the persisted mode.
    let labels = svc
        .list_labels(Request::new(schema::ListLabelsRequest {}))
        .await
        .expect("list_labels")
        .into_inner()
        .labels;

    let event = labels
        .iter()
        .find(|l| l.name == "Event")
        .expect("Event label must be in list");

    assert_eq!(
        event.schema_mode,
        schema::SchemaMode::Validated as i32,
        "list_labels must return schema_mode=VALIDATED for Event"
    );
}

/// STRICT label rejects type mismatch at CREATE time.
///
/// Regression: type validation for declared properties must be applied at
/// write time in STRICT mode, not silently stored with wrong type.
#[tokio::test]
async fn strict_mode_create_type_mismatch_rejected() {
    let (svc, _dir) = test_service();

    svc.create_label(Request::new(label(
        "Sensor",
        vec![prop("reading", schema::ScalarType::Float64, false)],
        schema::SchemaMode::Strict,
    )))
    .await
    .expect("create_label should succeed");

    // CREATE with wrong type for declared property — must be rejected.
    let mut db = svc.database.write();
    let result = db.execute_cypher("CREATE (s:Sensor {reading: 'not_a_float'})");
    assert!(
        result.is_err(),
        "CREATE with type mismatch must be rejected in STRICT mode, got: {result:?}"
    );
}

/// VALIDATED label enforces type for declared properties but accepts undeclared.
///
/// Regression: VALIDATED mode = "type-check declared, accept undeclared".
/// Type mismatch on a declared property must be rejected; extra property accepted.
#[tokio::test]
async fn validated_mode_type_mismatch_rejected_but_extra_accepted() {
    let (svc, _dir) = test_service();

    svc.create_label(Request::new(label(
        "Log",
        vec![prop("level", schema::ScalarType::Int64, false)],
        schema::SchemaMode::Validated,
    )))
    .await
    .expect("create_label should succeed");

    {
        let mut db = svc.database.write();

        // Type mismatch on declared property — must be rejected even in VALIDATED.
        let result = db.execute_cypher("CREATE (l:Log {level: 'warn'})");
        assert!(
            result.is_err(),
            "type mismatch on declared property must be rejected in VALIDATED mode"
        );

        // Undeclared property — must be accepted in VALIDATED mode.
        db.execute_cypher("CREATE (l:Log {level: 5, extra_tag: 'deployment'})")
            .expect("CREATE with undeclared property must succeed in VALIDATED mode");
    }
}

/// CREATE with an undeclared property on a STRICT label is rejected.
///
/// Regression: schema enforcement must trigger at CREATE time, not only at SET
/// time. STRICT mode means "no write path allows undeclared properties".
#[tokio::test]
async fn strict_mode_create_with_unknown_property_rejected() {
    let (svc, _dir) = test_service();

    // Declare Product label with STRICT mode; only `sku` declared.
    svc.create_label(Request::new(label(
        "Product",
        vec![prop("sku", schema::ScalarType::String, false)],
        schema::SchemaMode::Strict,
    )))
    .await
    .expect("create_label should succeed");

    // CREATE with only the declared property — must succeed.
    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (p:Product {sku: 'SKU-001'})")
            .expect("CREATE with declared property must succeed on STRICT label");
    }

    // CREATE with an undeclared property — must be rejected.
    {
        let mut db = svc.database.write();
        let result =
            db.execute_cypher("CREATE (p:Product {sku: 'SKU-002', extra_field: 'forbidden'})");
        assert!(
            result.is_err(),
            "CREATE with undeclared property must be rejected on STRICT label, got: {result:?}"
        );
    }
}

/// VALIDATED label: SET with undeclared property is accepted; SET with type
/// mismatch on declared property is rejected.
///
/// Regression: VALIDATED enforcement must apply to the SET path (MATCH … SET),
/// not only to CREATE. Both acceptance of extras and rejection of type mismatches
/// must be consistent across write paths.
#[tokio::test]
async fn validated_mode_set_extra_accepted_mismatch_rejected() {
    let (svc, _dir) = test_service();

    svc.create_label(Request::new(label(
        "Metric",
        vec![prop("value", schema::ScalarType::Float64, false)],
        schema::SchemaMode::Validated,
    )))
    .await
    .expect("create_label should succeed");

    {
        let mut db = svc.database.write();

        // Create node with only the declared property.
        db.execute_cypher("CREATE (m:Metric {value: 3.14})")
            .expect("CREATE declared property must succeed");

        // SET undeclared property on VALIDATED label — must be accepted.
        db.execute_cypher("MATCH (m:Metric) SET m.source = 'sensor-01'")
            .expect("SET undeclared property must be accepted in VALIDATED mode");

        // SET type mismatch on declared property — must be rejected.
        let result = db.execute_cypher("MATCH (m:Metric) SET m.value = 'not_a_float'");
        assert!(
            result.is_err(),
            "SET with type mismatch on declared property must be rejected in VALIDATED mode, got: {result:?}"
        );
    }
}

/// STRICT label: CREATE omitting a required (NOT NULL) property is rejected.
///
/// Regression: required field enforcement must apply at CREATE time in STRICT
/// and VALIDATED modes. The executor must check that all NOT NULL properties
/// without a default are present in the CREATE clause, not only validate
/// properties that are provided.
#[tokio::test]
async fn strict_mode_create_missing_required_property_rejected() {
    let (svc, _dir) = test_service();

    svc.create_label(Request::new(label(
        "Task",
        vec![
            prop("title", schema::ScalarType::String, true),
            prop("priority", schema::ScalarType::Int64, false),
        ],
        schema::SchemaMode::Strict,
    )))
    .await
    .expect("create_label should succeed");

    {
        let mut db = svc.database.write();

        // CREATE with required `title` present — must succeed.
        db.execute_cypher("CREATE (t:Task {title: 'Buy milk', priority: 1})")
            .expect("CREATE with all required properties must succeed");

        // CREATE omitting required `title` — must be rejected.
        let result = db.execute_cypher("CREATE (t:Task {priority: 2})");
        assert!(
            result.is_err(),
            "CREATE omitting required property must be rejected in STRICT mode, got: {result:?}"
        );

        // Optional property only — must be rejected (title still required).
        let result2 = db.execute_cypher("CREATE (t:Task {priority: 3})");
        assert!(
            result2.is_err(),
            "CREATE omitting required property must be rejected in STRICT mode, got: {result2:?}"
        );
    }
}

/// Multi-update (MATCH returns mixed labels): STRICT nodes fail, FLEXIBLE pass.
///
/// When MATCH returns nodes of different labels — some with STRICT schema,
/// some without schema (FLEXIBLE) — SET enforcement is per-node based on
/// that node's primary label schema. A violation on any STRICT node fails
/// the entire query (transactional semantics).
///
/// This test also verifies forward-only enforcement: nodes created BEFORE a
/// schema was declared retain their schemaless properties — SET on such a node
/// validates only the specific property being SET, not existing properties.
#[tokio::test]
async fn multi_update_strict_node_fails_whole_query() {
    let (svc, _dir) = test_service();

    // Declare STRICT schema for Monitored label (only `host` property).
    svc.create_label(Request::new(label(
        "Monitored",
        vec![prop("host", schema::ScalarType::String, false)],
        schema::SchemaMode::Strict,
    )))
    .await
    .expect("create_label should succeed");

    {
        let mut db = svc.database.write();

        // Create one STRICT node and one schemaless node.
        db.execute_cypher("CREATE (m:Monitored {host: 'server-01'})")
            .expect("create strict node");
        db.execute_cypher("CREATE (s:Server {host: 'server-02'})")
            .expect("create schemaless node");

        // SET on Monitored (STRICT) with an undeclared property — fails the whole query.
        // Even though Server has no schema (FLEXIBLE), the STRICT violation on
        // Monitored causes the entire multi-update to be rejected.
        let result =
            db.execute_cypher("MATCH (n) WHERE n.host IS NOT NULL SET n.unknown_prop = 'x'");
        assert!(
            result.is_err(),
            "multi-update touching a STRICT node with unknown property must fail entire query, got: {result:?}"
        );

        // SET only on schemaless (Server) nodes — succeeds even with unknown property.
        db.execute_cypher("MATCH (s:Server) SET s.extra = 'ok'")
            .expect("SET on schemaless node must succeed regardless of unknown property");
    }
}

/// MERGE on a STRICT label — ON CREATE SET path must be schema-checked.
/// Merging with a declared property succeeds; merging with an unknown property fails.
#[tokio::test]
async fn strict_mode_merge_on_create_set_rejected_for_unknown_property() {
    let (svc, _dir) = test_service();

    svc.create_label(Request::new(label(
        "Product",
        vec![
            prop("sku", schema::ScalarType::String, false),
            prop("price", schema::ScalarType::Float64, false),
        ],
        schema::SchemaMode::Strict,
    )))
    .await
    .expect("create_label Product");

    // ON CREATE SET with declared property — must succeed (creates node).
    {
        let mut db = svc.database.write();
        db.execute_cypher("MERGE (p:Product {sku: 'A1'}) ON CREATE SET p.price = 9.99")
            .expect("MERGE ON CREATE SET with declared property must succeed");
    }

    // ON MATCH SET with unknown property — must fail (node now exists → ON MATCH path).
    {
        let mut db = svc.database.write();
        let result =
            db.execute_cypher("MERGE (p:Product {sku: 'A1'}) ON MATCH SET p.unknown_field = 'x'");
        assert!(
            result.is_err(),
            "MERGE ON MATCH SET with unknown property on STRICT label must fail, got: {result:?}"
        );
    }
}

/// SET n = {map} (ReplaceProperties) on STRICT label — unknown key in map must be rejected.
#[tokio::test]
async fn strict_mode_replace_properties_rejects_unknown_key() {
    let (svc, _dir) = test_service();

    svc.create_label(Request::new(label(
        "Config",
        vec![
            prop("host", schema::ScalarType::String, false),
            prop("port", schema::ScalarType::Int64, false),
        ],
        schema::SchemaMode::Strict,
    )))
    .await
    .expect("create_label Config");

    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (c:Config {host: 'localhost', port: 5432})")
            .expect("create Config node");
    }

    // Replace with only declared keys — must succeed.
    {
        let mut db = svc.database.write();
        db.execute_cypher("MATCH (c:Config) SET c = {host: 'db.internal', port: 3306}")
            .expect("SET n = {map} with declared keys must succeed");
    }

    // Replace introducing unknown key — must fail.
    {
        let mut db = svc.database.write();
        let result =
            db.execute_cypher("MATCH (c:Config) SET c = {host: 'x', port: 3306, extra: 'oops'}");
        assert!(
            result.is_err(),
            "SET n = {{map}} with unknown key on STRICT label must fail, got: {result:?}"
        );
    }
}

/// SET n += {map} (MergeProperties) on STRICT label — unknown key in merge map must be rejected.
#[tokio::test]
async fn strict_mode_merge_properties_rejects_unknown_key() {
    let (svc, _dir) = test_service();

    svc.create_label(Request::new(label(
        "AppService",
        vec![
            prop("name", schema::ScalarType::String, false),
            prop("version", schema::ScalarType::String, false),
        ],
        schema::SchemaMode::Strict,
    )))
    .await
    .expect("create_label AppService");

    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (s:AppService {name: 'api', version: '1.0'})")
            .expect("create AppService node");
    }

    // Merge with declared keys — must succeed.
    {
        let mut db = svc.database.write();
        db.execute_cypher("MATCH (s:AppService) SET s += {version: '2.0'}")
            .expect("SET n += {map} with declared keys must succeed");
    }

    // Merge introducing unknown key — must fail.
    {
        let mut db = svc.database.write();
        let result =
            db.execute_cypher("MATCH (s:AppService) SET s += {version: '3.0', secret: 'leaked'}");
        assert!(
            result.is_err(),
            "SET n += {{map}} with unknown key on STRICT label must fail, got: {result:?}"
        );
    }
}

/// ON VIOLATION SKIP: nodes that violate schema are excluded from results; compliant nodes proceed.
#[tokio::test]
async fn on_violation_skip_excludes_violating_nodes() {
    let (svc, _dir) = test_service();

    // Gadget has a strict schema: only 'name' and 'status' are allowed.
    svc.create_label(Request::new(label(
        "Gadget",
        vec![
            prop("name", schema::ScalarType::String, false),
            prop("status", schema::ScalarType::String, false),
        ],
        schema::SchemaMode::Strict,
    )))
    .await
    .expect("create_label Gadget");

    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (g:Gadget {name: 'light', status: 'on'})")
            .expect("create g1");
        db.execute_cypher("CREATE (g:Gadget {name: 'fan', status: 'off'})")
            .expect("create g2");
    }

    // Without ON VIOLATION SKIP: setting unknown property on all Gadget nodes fails.
    {
        let mut db = svc.database.write();
        let result = db.execute_cypher("MATCH (g:Gadget) SET g.firmware = 'v1'");
        assert!(
            result.is_err(),
            "SET unknown property on STRICT label without SKIP must fail"
        );
    }

    // With ON VIOLATION SKIP: query succeeds; violating nodes are excluded from output.
    // 'firmware' is unknown → all Gadget nodes are skipped → empty result set.
    {
        let mut db = svc.database.write();
        let rows = db
            .execute_cypher(
                "MATCH (g:Gadget) SET g.firmware = 'v1' ON VIOLATION SKIP RETURN g.name",
            )
            .expect("ON VIOLATION SKIP must not fail");
        assert_eq!(
            rows.len(),
            0,
            "all violating nodes skipped → empty result, got: {rows:?}"
        );
    }

    // ON VIOLATION SKIP with a valid property: all nodes are updated, all returned.
    {
        let mut db = svc.database.write();
        let rows = db
            .execute_cypher(
                "MATCH (g:Gadget) SET g.status = 'idle' ON VIOLATION SKIP RETURN g.name",
            )
            .expect("ON VIOLATION SKIP with valid property must succeed");
        assert_eq!(rows.len(), 2, "both Gadget nodes updated, got: {rows:?}");
    }
}

/// SET n.unknownDoc.subfield = val on a STRICT label must fail:
/// the root property 'unknownDoc' is not declared.
#[tokio::test]
async fn strict_mode_property_path_rejects_unknown_root() {
    let (svc, _dir) = test_service();
    svc.create_label(Request::new(label(
        "Device",
        vec![prop("serial", schema::ScalarType::String, true)],
        schema::SchemaMode::Strict,
    )))
    .await
    .expect("create_label Device");

    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (d:Device {serial: 'SN-001'})")
            .expect("create device");
    }

    // 'config' is not declared on Device (STRICT) — SET d.config.timeout = 30 must fail.
    let mut db = svc.database.write();
    let result = db.execute_cypher("MATCH (d:Device) SET d.config.timeout = 30 RETURN d.serial");
    assert!(
        result.is_err(),
        "PropertyPath with unknown root on STRICT label must fail, got: {result:?}"
    );
}

/// doc_push(n.unknownList, val) on a STRICT label must fail:
/// the root property 'unknownList' is not declared.
#[tokio::test]
async fn strict_mode_doc_function_rejects_unknown_root() {
    let (svc, _dir) = test_service();
    svc.create_label(Request::new(label(
        "Shelf",
        vec![prop("name", schema::ScalarType::String, true)],
        schema::SchemaMode::Strict,
    )))
    .await
    .expect("create_label Shelf");

    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (s:Shelf {name: 'A1'})")
            .expect("create shelf");
    }

    // 'items' is not declared on Shelf (STRICT) — doc_push on unknown root prop must fail.
    // Syntax: `SET doc_push(n.prop, val)` — function call IS the set item.
    let mut db = svc.database.write();
    let result = db.execute_cypher("MATCH (s:Shelf) SET doc_push(s.items, 'book')");
    assert!(
        result.is_err(),
        "doc_push on unknown root property of STRICT label must fail, got: {result:?}"
    );
}

/// SET n = {host: 'x', extra: 'ok'} on a VALIDATED label must succeed:
/// in VALIDATED mode unknown keys are allowed (declared keys are type-checked).
#[tokio::test]
async fn validated_mode_replace_properties_allows_unknown_key() {
    let (svc, _dir) = test_service();
    svc.create_label(Request::new(label(
        "Server",
        vec![prop("host", schema::ScalarType::String, false)],
        schema::SchemaMode::Validated,
    )))
    .await
    .expect("create_label Server");

    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (s:Server {host: 'db1'})")
            .expect("create server");
    }

    // Replace with map that has extra unknown key 'datacenter' — must succeed in VALIDATED.
    let mut db = svc.database.write();
    let rows = db
        .execute_cypher(
            "MATCH (s:Server) SET s = {host: 'db2', datacenter: 'eu-west-1'} RETURN s.host",
        )
        .expect("SET n = {map} with unknown key on VALIDATED label must succeed");
    assert_eq!(rows.len(), 1, "expected one updated row, got: {rows:?}");
    assert_eq!(
        rows[0].get("s.host").cloned().unwrap_or(Value::Null),
        Value::String("db2".into()),
        "declared property 'host' must be updated"
    );
}

/// SET n += {extra: 'ok'} on a VALIDATED label must succeed:
/// in VALIDATED mode unknown keys in merge-properties are allowed.
#[tokio::test]
async fn validated_mode_merge_properties_allows_unknown_key() {
    let (svc, _dir) = test_service();
    svc.create_label(Request::new(label(
        "Cache",
        vec![prop("size", schema::ScalarType::Int64, false)],
        schema::SchemaMode::Validated,
    )))
    .await
    .expect("create_label Cache");

    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (c:Cache {size: 100})")
            .expect("create cache");
    }

    // Merge with unknown key 'policy' — must succeed in VALIDATED.
    let mut db = svc.database.write();
    let rows = db
        .execute_cypher("MATCH (c:Cache) SET c += {size: 200, policy: 'lru'} RETURN c.size")
        .expect("SET n += {map} with unknown key on VALIDATED label must succeed");
    assert_eq!(rows.len(), 1, "expected one updated row, got: {rows:?}");
    assert_eq!(
        rows[0].get("c.size").cloned().unwrap_or(Value::Null),
        Value::Int(200),
        "declared property 'size' must be updated to 200"
    );
}

/// labels(n)[0] — subscript access on function result returns the primary label.
#[tokio::test]
async fn labels_function_subscript_access() {
    let (svc, _dir) = test_service();
    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (n:Robot {name: 'R2D2'})")
            .expect("create");
    }

    let mut db = svc.database.write();
    let rows = db
        .execute_cypher("MATCH (n:Robot) RETURN labels(n)[0] AS lbl")
        .expect("labels(n)[0] must be supported");
    assert_eq!(rows.len(), 1, "expected one row");
    let lbl = rows[0].get("lbl").cloned().unwrap_or(Value::Null);
    assert_eq!(
        lbl,
        Value::String("Robot".into()),
        "labels(n)[0] must return primary label 'Robot', got: {lbl:?}"
    );
}

/// SET n.unknownDoc.field = val on a VALIDATED label must succeed:
/// in VALIDATED mode unknown root properties are accepted (only declared props type-checked).
#[tokio::test]
async fn validated_mode_property_path_allows_unknown_root() {
    let (svc, _dir) = test_service();
    svc.create_label(Request::new(label(
        "Sensor",
        vec![prop("id", schema::ScalarType::String, false)],
        schema::SchemaMode::Validated,
    )))
    .await
    .expect("create_label Sensor");

    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (s:Sensor {id: 'S1'})")
            .expect("create sensor");
    }

    // 'readings' is not declared on Sensor (VALIDATED) — deep SET must succeed.
    let mut db = svc.database.write();
    let result = db.execute_cypher("MATCH (s:Sensor) SET s.readings.temp = 42 RETURN s.id");
    assert!(
        result.is_ok(),
        "PropertyPath with unknown root on VALIDATED label must succeed, got: {result:?}"
    );
}

/// SET doc_push(n.unknownList, val) on a VALIDATED label must succeed:
/// in VALIDATED mode unknown root properties are accepted.
#[tokio::test]
async fn validated_mode_doc_function_allows_unknown_root() {
    let (svc, _dir) = test_service();
    svc.create_label(Request::new(label(
        "Bin",
        vec![prop("tag", schema::ScalarType::String, false)],
        schema::SchemaMode::Validated,
    )))
    .await
    .expect("create_label Bin");

    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (b:Bin {tag: 'B1'})")
            .expect("create bin");
    }

    // 'items' is not declared on Bin (VALIDATED) — doc_push must succeed.
    let mut db = svc.database.write();
    let result = db.execute_cypher("MATCH (b:Bin) SET doc_push(b.items, 'x')");
    assert!(
        result.is_ok(),
        "doc_push on unknown root of VALIDATED label must succeed, got: {result:?}"
    );
}

/// SET n.unknownProp.sub = val ON VIOLATION SKIP on a STRICT label:
/// nodes whose PropertyPath violates schema are silently excluded.
#[tokio::test]
async fn on_violation_skip_with_property_path() {
    let (svc, _dir) = test_service();
    svc.create_label(Request::new(label(
        "Relay",
        vec![prop("state", schema::ScalarType::String, false)],
        schema::SchemaMode::Strict,
    )))
    .await
    .expect("create_label Relay");

    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (r:Relay {state: 'open'})")
            .expect("create relay");
        db.execute_cypher("CREATE (r:Relay {state: 'closed'})")
            .expect("create relay 2");
    }

    // 'config' is not declared — without SKIP the whole query fails.
    {
        let mut db = svc.database.write();
        let result = db.execute_cypher("MATCH (r:Relay) SET r.config.timeout = 30 RETURN r.state");
        assert!(
            result.is_err(),
            "PropertyPath on unknown root without SKIP must fail"
        );
    }

    // With ON VIOLATION SKIP: query succeeds, all violating nodes excluded → empty result.
    {
        let mut db = svc.database.write();
        let rows = db
            .execute_cypher(
                "MATCH (r:Relay) SET r.config.timeout = 30 ON VIOLATION SKIP RETURN r.state",
            )
            .expect("ON VIOLATION SKIP must not fail");
        assert_eq!(
            rows.len(),
            0,
            "all violating Relay nodes skipped → empty result, got: {rows:?}"
        );
    }

    // With ON VIOLATION SKIP on a VALID deep path: all nodes updated, all returned.
    // 'state' is declared — SET r.state = 'on' is Property SET, not PropertyPath.
    // Use a declared Document property to test the PropertyPath success path properly.
    // (No Document property declared here, so just verify SKIP on all-fail → empty.)
}

/// Multiple PropertyPath SET-items on the same node in one statement (the
/// per-statement schema-label cache).
///
/// Verifies that N PropertyPath items targeting the same node in a single SET
/// clause all succeed. This exercises the `schema_label_cache` path: the first
/// item reads + caches the label; subsequent items hit the cache (O(1), no I/O).
/// Also guards against the read-your-own-writes merge-delta bug: each
/// DocDelta must land independently when multiple paths target the same node.
#[tokio::test]
async fn schema_label_cache_multiple_paths_same_node() {
    let (svc, _dir) = test_service();
    svc.create_label(Request::new(label(
        "Config",
        vec![prop("cfg", schema::ScalarType::String, false)],
        // Extra paths are allowed.
        schema::SchemaMode::Validated,
    )))
    .await
    .expect("create_label Config");

    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (c:Config)").expect("create node");
    }

    // Three PropertyPath items on the same node in one statement.
    // Each uses schema_label_for_node — only the first should hit the engine;
    // items 2 and 3 should read the cached label.
    let mut db = svc.database.write();
    db.execute_cypher(
        "MATCH (c:Config) SET c.cfg.host = 'db1', c.cfg.port = 5432, c.cfg.ssl = true",
    )
    .expect("multi PropertyPath on same node must succeed");

    let rows = db
        .execute_cypher("MATCH (c:Config) RETURN c.cfg.host, c.cfg.port, c.cfg.ssl")
        .expect("return multi paths");
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("c.cfg.host"),
        Some(&Value::String("db1".to_string())),
        "c.cfg.host must be 'db1'"
    );
    assert_eq!(
        rows[0].get("c.cfg.port"),
        Some(&Value::Int(5432)),
        "c.cfg.port must be 5432"
    );
    assert_eq!(
        rows[0].get("c.cfg.ssl"),
        Some(&Value::Bool(true)),
        "c.cfg.ssl must be true"
    );
}

/// Multiple DocFunction SET-items on the same node in one statement (the
/// per-statement schema-label cache).
///
/// Verifies that N `doc_push` items targeting the same node's arrays in a single
/// SET clause all succeed. DocFunction uses the same `schema_label_for_node` cache
/// as PropertyPath — first item reads + caches, subsequent items hit cache (O(1)).
#[tokio::test]
async fn schema_label_cache_multiple_doc_functions_same_node() {
    let (svc, _dir) = test_service();
    svc.create_label(Request::new(label(
        "Queue",
        vec![prop("jobs", schema::ScalarType::String, false)],
        // Extra roots are allowed.
        schema::SchemaMode::Validated,
    )))
    .await
    .expect("create_label Queue");

    {
        let mut db = svc.database.write();
        db.execute_cypher("CREATE (q:Queue)").expect("create node");
    }

    // Three doc_push items on the same node in one statement.
    // 'jobs', 'errors', 'metrics' are all undeclared roots on VALIDATED →
    // all accepted. schema_label_for_node cache: 1 engine read for all three.
    let mut db = svc.database.write();
    db.execute_cypher(
            "MATCH (q:Queue) SET doc_push(q.jobs, 'job1'), doc_push(q.errors, 'err1'), doc_push(q.metrics, 42)",
        )
        .expect("multi doc_push on same node must succeed");

    let rows = db
        .execute_cypher("MATCH (q:Queue) RETURN q.jobs, q.errors, q.metrics")
        .expect("return after multi doc_push");
    assert_eq!(rows.len(), 1);
    // Arrays are stored as lists; each doc_push appends one element.
    assert!(
        rows[0].contains_key("q.jobs"),
        "q.jobs must be set after doc_push"
    );
    assert!(
        rows[0].contains_key("q.errors"),
        "q.errors must be set after doc_push"
    );
    assert!(
        rows[0].contains_key("q.metrics"),
        "q.metrics must be set after doc_push"
    );
}
