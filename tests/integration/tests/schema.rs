//! Schema management through the running server: the v2 service over gRPC
//! and over its REST transcoding, and the absence of the schema service it
//! replaced.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_integration::harness::CoordinodeProcess;
use coordinode_integration::proto::v2::graph::{
    ConstraintKind, ConstraintState, CreateConstraintRequest, CreateLabelRequest,
    DropConstraintRequest, ListConstraintsRequest, ListLabelsRequest, PropertyDefinition,
    PropertyType, ScalarType, SchemaMode, constraint, create_constraint_request, property_type,
};

/// The request the removed v1 `CreateLabel` took, as fixed test input: its
/// field numbers are written out here so no v1 client is built.
#[derive(Clone, PartialEq, prost::Message)]
struct RemovedCreateLabel {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(message, repeated, tag = "2")]
    properties: Vec<RemovedProperty>,
}

/// A v1 property definition, carrying the removed uniqueness flag.
#[derive(Clone, PartialEq, prost::Message)]
struct RemovedProperty {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(int32, tag = "2")]
    r#type: i32,
    #[prost(bool, tag = "3")]
    required: bool,
    #[prost(bool, tag = "4")]
    unique: bool,
}

/// The labels and constraints the catalog holds, by name.
async fn catalog(proc: &CoordinodeProcess) -> (Vec<String>, Vec<String>) {
    let mut sc = proc.schema_client().await;
    let labels = sc
        .list_labels(ListLabelsRequest {})
        .await
        .expect("list labels")
        .into_inner()
        .labels
        .into_iter()
        .map(|l| l.name)
        .collect();
    let constraints = sc
        .list_constraints(ListConstraintsRequest {})
        .await
        .expect("list constraints")
        .into_inner()
        .constraints
        .into_iter()
        .map(|c| c.name)
        .collect();
    (labels, constraints)
}

/// A call to the removed schema service is an unknown method: the transport
/// answers UNIMPLEMENTED and nothing reaches the catalog.
#[tokio::test]
async fn the_removed_schema_rpc_is_unknown_and_changes_nothing() {
    let proc = CoordinodeProcess::start().await;
    let channel = tonic::transport::Endpoint::from_shared(proc.endpoint())
        .expect("endpoint")
        .connect()
        .await
        .expect("connect");
    let mut grpc = tonic::client::Grpc::new(channel);
    grpc.ready().await.expect("ready");
    let request = RemovedCreateLabel {
        name: "Legacy".into(),
        properties: vec![RemovedProperty {
            name: "email".into(),
            r#type: 3,
            required: true,
            unique: true,
        }],
    };
    let status = grpc
        .unary::<_, (), _>(
            tonic::Request::new(request),
            tonic::codegen::http::uri::PathAndQuery::from_static(
                "/coordinode.v1.graph.SchemaService/CreateLabel",
            ),
            tonic_prost::ProstCodec::default(),
        )
        .await
        .expect_err("the removed method is unknown");
    assert_eq!(status.code(), tonic::Code::Unimplemented, "{status:?}");

    let (labels, constraints) = catalog(&proc).await;
    assert!(labels.is_empty(), "no label was created: {labels:?}");
    assert!(
        constraints.is_empty(),
        "no constraint was created: {constraints:?}"
    );
}

/// The removed schema routes are absent from the REST surface: not found,
/// not redirected to the new service, and nothing reaches the catalog.
#[tokio::test]
async fn the_removed_schema_routes_are_absent() {
    let proc = CoordinodeProcess::start().await;
    let (status, body) = proc.rest_request(
        "POST",
        "/v1/graph/schema/labels",
        Some(r#"{"name":"Legacy","properties":[{"name":"email","type":3,"unique":true}]}"#),
    );
    assert_eq!(status, 404, "{body}");
    let (status, body) = proc.rest_request("GET", "/v1/graph/schema/labels", None);
    assert_eq!(status, 404, "{body}");

    let (labels, constraints) = catalog(&proc).await;
    assert!(labels.is_empty(), "no label was created: {labels:?}");
    assert!(
        constraints.is_empty(),
        "no constraint was created: {constraints:?}"
    );
}

/// A type definition and a uniqueness constraint are created apart and read
/// back apart, over gRPC and over REST: the definition carries no
/// uniqueness, the constraint carries its state and the index it owns.
#[tokio::test]
async fn types_and_constraints_are_created_and_inspected_apart() {
    let proc = CoordinodeProcess::start().await;
    let mut sc = proc.schema_client().await;

    let label = sc
        .create_label(CreateLabelRequest {
            name: "Acct".into(),
            properties: vec![PropertyDefinition {
                name: "email".into(),
                r#type: Some(PropertyType {
                    r#type: Some(property_type::Type::Scalar(ScalarType::String as i32)),
                }),
                required: true,
                default_value: None,
            }],
            computed_properties: vec![],
            schema_mode: SchemaMode::Strict as i32,
            temporal: false,
        })
        .await
        .expect("create label")
        .into_inner();
    assert!(label.declared);
    assert_eq!(label.properties.len(), 1);
    assert!(label.properties[0].required);
    assert!(
        sc.list_constraints(ListConstraintsRequest {})
            .await
            .expect("list")
            .into_inner()
            .constraints
            .is_empty(),
        "defining a type creates no constraint"
    );

    let created = sc
        .create_constraint(CreateConstraintRequest {
            name: "acct_email".into(),
            target: Some(create_constraint_request::Target::Label("Acct".into())),
            properties: vec!["email".into()],
            kind: ConstraintKind::Unique as i32,
            property_type: None,
            if_not_exists: false,
        })
        .await
        .expect("create constraint")
        .into_inner();
    assert_eq!(created.name, "acct_email");
    assert_eq!(
        created.target,
        Some(constraint::Target::Label("Acct".into()))
    );
    assert_eq!(created.state, ConstraintState::Active as i32);
    assert_eq!(created.backing_index, "acct_email");

    // The REST surface reports the same constraint, and creates and drops
    // through the same catalog.
    let (status, body) = proc.rest_request("GET", "/v2/graph/schema/constraints", None);
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("acct_email"), "{body}");
    let (status, body) = proc.rest_request(
        "POST",
        "/v2/graph/schema/constraints",
        Some(
            r#"{"name":"acct_email_set","label":"Acct","properties":["email"],"kind":"CONSTRAINT_KIND_NOT_NULL"}"#,
        ),
    );
    assert_eq!(status, 200, "{body}");
    let names = catalog(&proc).await.1;
    assert_eq!(names, ["acct_email", "acct_email_set"]);
    let (status, body) = proc.rest_request(
        "DELETE",
        "/v2/graph/schema/constraints/acct_email_set",
        None,
    );
    assert_eq!(status, 200, "{body}");

    sc.drop_constraint(DropConstraintRequest {
        name: "acct_email".into(),
        if_exists: false,
    })
    .await
    .expect("drop constraint");
    let (labels, constraints) = catalog(&proc).await;
    assert_eq!(labels, ["Acct"], "dropping constraints keeps the type");
    assert!(constraints.is_empty(), "{constraints:?}");
}
