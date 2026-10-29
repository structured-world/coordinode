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

fn string_property(name: &str) -> schema::PropertyDefinition {
    schema::PropertyDefinition {
        name: name.to_string(),
        r#type: Some(schema::PropertyType {
            r#type: Some(schema::property_type::Type::Scalar(
                schema::ScalarType::String as i32,
            )),
        }),
        required: false,
        default_value: None,
    }
}

/// A STRICT `User` with `email`, and a uniqueness on it.
fn request() -> schema::PublishSchemaRequest {
    schema::PublishSchemaRequest {
        labels: vec![schema::LabelDeclaration {
            name: "User".into(),
            properties: vec![string_property("email")],
            computed_properties: vec![],
            schema_mode: schema::SchemaMode::Strict as i32,
            temporal: false,
            expected_revision: None,
            kind: schema::LabelKind::Type as i32,
            implies: vec![],
        }],
        edge_types: vec![schema::EdgeTypeDeclaration {
            name: "FOLLOWS".into(),
            properties: vec![],
            temporal: false,
            discriminator: String::new(),
            expected_revision: None,
        }],
        constraints: vec![schema::ConstraintDeclaration {
            name: "user_email".into(),
            target: Some(schema::constraint_declaration::Target::Label("User".into())),
            properties: vec!["email".into()],
            kind: schema::ConstraintKind::Unique as i32,
            property_type: None,
            on_duplicate_rename: String::new(),
            scope: None,
        }],
        operation: None,
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

async fn await_all(
    svc: &SchemaServiceImpl,
    published: &schema::PublishSchemaResponse,
) -> schema::AwaitSchemaResponse {
    svc.await_schema(Request::new(schema::AwaitSchemaRequest {
        operations: published
            .objects
            .iter()
            .filter_map(|o| o.operation)
            .collect(),
        wait: Some(prost_types::Duration {
            seconds: 30,
            nanos: 0,
        }),
    }))
    .await
    .expect("await")
    .into_inner()
}

/// One request publishes the label, the edge type and the constraint, each
/// with the work putting it in force; awaiting those operations reports
/// them ready, after which the label is listed STRICT and in force.
#[tokio::test]
async fn a_package_is_published_and_awaited() {
    let (svc, _dir) = test_service();
    let published = svc
        .publish_schema(Request::new(request()))
        .await
        .expect("publish")
        .into_inner();
    assert!(published.committed);
    let kinds: Vec<_> = published
        .objects
        .iter()
        .map(|o| (o.kind, o.name.as_str(), o.disposition))
        .collect();
    use schema::{PublicationDisposition as D, SchemaObjectKind as K};
    assert_eq!(
        kinds,
        vec![
            (K::Label as i32, "User", D::Created as i32),
            (K::EdgeType as i32, "FOLLOWS", D::Created as i32),
            (K::Constraint as i32, "user_email", D::Created as i32),
        ]
    );
    assert!(published.objects[0].operation.is_some(), "STRICT is proved");
    assert!(published.objects[1].operation.is_none());
    assert!(
        published.objects[2].operation.is_some(),
        "the index is built"
    );

    let waited = await_all(&svc, &published).await;
    assert_eq!(waited.operations.len(), 2);
    assert!(
        waited
            .operations
            .iter()
            .all(|o| o.state == schema::SchemaReadinessState::Ready as i32),
        "{waited:?}"
    );
    let labels = svc
        .list_labels(Request::new(schema::ListLabelsRequest {}))
        .await
        .expect("list")
        .into_inner()
        .labels;
    let user = labels.iter().find(|l| l.name == "User").expect("User");
    assert_eq!(user.schema_mode, schema::SchemaMode::Strict as i32);
    assert!(user.candidate.is_none(), "proved and in force");
}

/// The same request again commits nothing and reports every object
/// retained.
#[tokio::test]
async fn an_unchanged_package_commits_nothing() {
    let (svc, _dir) = test_service();
    let first = svc
        .publish_schema(Request::new(request()))
        .await
        .expect("publish")
        .into_inner();
    await_all(&svc, &first).await;
    let again = svc
        .publish_schema(Request::new(request()))
        .await
        .expect("again")
        .into_inner();
    assert!(!again.committed);
    assert!(
        again
            .objects
            .iter()
            .all(|o| o.disposition == schema::PublicationDisposition::Retained as i32),
        "{again:?}"
    );
}

/// A declaration differing from the object in place, without naming its
/// revision, is FAILED_PRECONDITION with reason CATALOG_OBJECT_DIFFERS
/// naming the revision a change would name; a change naming a stale
/// revision is CATALOG_REVISION_MISMATCH.
#[tokio::test]
async fn mismatches_are_typed_conflicts() {
    let (svc, _dir) = test_service();
    let first = svc
        .publish_schema(Request::new(request()))
        .await
        .expect("publish")
        .into_inner();
    await_all(&svc, &first).await;

    let mut differing = request();
    differing.labels[0].properties.push(string_property("nick"));
    let status = svc
        .publish_schema(Request::new(differing.clone()))
        .await
        .expect_err("differs");
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    assert_eq!(reason(&status), "CATALOG_OBJECT_DIFFERS");
    let revision: u64 = status
        .get_details_error_info()
        .and_then(|i| i.metadata.get("revision").cloned())
        .expect("revision")
        .parse()
        .expect("a number");

    differing.labels[0].expected_revision = Some(revision - 1);
    let status = svc
        .publish_schema(Request::new(differing.clone()))
        .await
        .expect_err("stale");
    assert_eq!(reason(&status), "CATALOG_REVISION_MISMATCH");

    differing.labels[0].expected_revision = Some(revision);
    let changed = svc
        .publish_schema(Request::new(differing))
        .await
        .expect("change")
        .into_inner();
    assert_eq!(
        changed.objects[0].disposition,
        schema::PublicationDisposition::Changed as i32
    );
}

/// A malformed declaration is refused with the path of the field at fault,
/// and nothing of the request is published.
#[tokio::test]
async fn a_malformed_declaration_names_its_field() {
    let (svc, _dir) = test_service();
    let mut bad = request();
    bad.constraints[0].target = None;
    let status = svc
        .publish_schema(Request::new(bad))
        .await
        .expect_err("refused");
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    let field = status
        .get_details_error_info()
        .and_then(|i| i.metadata.get("field").cloned());
    assert_eq!(field.as_deref(), Some("constraints[0].label"));
    let labels = svc
        .list_labels(Request::new(schema::ListLabelsRequest {}))
        .await
        .expect("list")
        .into_inner()
        .labels;
    assert!(labels.is_empty());
}

/// A request carrying an operation takes effect once: the same request
/// again under that operation is FAILED_PRECONDITION, reason
/// OPERATION_SETTLED.
#[tokio::test]
async fn a_publication_under_an_operation_is_answered_by_its_outcome() {
    let (svc, _dir) = test_service();
    let issuer = svc
        .database
        .read()
        .enroll_issuer(coordinode_embed::operation::IssuerId([0x52; 16]))
        .expect("enroll");
    let descriptor = issuer.next().expect("descriptor");
    let mut first = request();
    first.operation = Some(crate::services::operation::descriptor_to_proto(&descriptor));
    svc.publish_schema(Request::new(first.clone()))
        .await
        .expect("publish");
    let status = svc
        .publish_schema(Request::new(first))
        .await
        .expect_err("settled");
    assert_eq!(reason(&status), "OPERATION_SETTLED");
}

/// An operation the server knows nothing of is reported failed, not left
/// waiting.
#[tokio::test]
async fn an_unknown_operation_is_reported_failed() {
    let (svc, _dir) = test_service();
    let waited = svc
        .await_schema(Request::new(schema::AwaitSchemaRequest {
            operations: vec![9_999],
            wait: None,
        }))
        .await
        .expect("await")
        .into_inner();
    assert_eq!(
        waited.operations[0].state,
        schema::SchemaReadinessState::Failed as i32
    );
}
