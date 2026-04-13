use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};

use coordinode_core::graph::types::{Value, VectorMetric};
use coordinode_core::schema::computed::{ComputedSpec, DecayFormula, TtlScope};
use coordinode_core::schema::definition::{
    EdgeTypeSchema, LabelSchema, PropertyDef, PropertyType, SchemaMode,
};
use coordinode_embed::Database;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;

use crate::proto::graph;

/// Map proto `PropertyType` integer value to internal `PropertyType`.
///
/// Proto integers:
///   0 = UNSPECIFIED, 1 = INT64, 2 = FLOAT64, 3 = STRING, 4 = BOOL,
///   5 = BYTES, 6 = TIMESTAMP, 7 = VECTOR, 8 = LIST, 9 = MAP
fn proto_type_to_property_type(t: i32) -> PropertyType {
    match t {
        1 => PropertyType::Int,
        2 => PropertyType::Float,
        3 => PropertyType::String,
        4 => PropertyType::Bool,
        5 => PropertyType::Binary,
        6 => PropertyType::Timestamp,
        // VECTOR without explicit dimensions — schema stores structural intent;
        // HNSW index creation (R-API3) will fill dimensions from CREATE VECTOR INDEX.
        7 => PropertyType::Vector {
            dimensions: 0,
            metric: VectorMetric::Cosine,
        },
        8 => PropertyType::Array(Box::new(PropertyType::String)),
        9 => PropertyType::Map,
        _ => PropertyType::String, // UNSPECIFIED → String (most permissive)
    }
}

/// Convert a proto `ComputedPropertyDefinition` to an internal `ComputedSpec`.
///
/// Returns `Err(Status::invalid_argument)` if required fields are missing or
/// the computed_type is UNSPECIFIED.
fn proto_to_computed_spec(def: &graph::ComputedPropertyDefinition) -> Result<ComputedSpec, Status> {
    let computed_type = def.computed_type;
    let formula_type = def.formula_type;

    let formula = match formula_type {
        // 0 = UNSPECIFIED → default to Linear (acceptable for TTL where formula is unused)
        0 | 1 => DecayFormula::Linear,
        2 => DecayFormula::Exponential { lambda: def.lambda },
        3 => DecayFormula::PowerLaw {
            tau: def.tau,
            alpha: def.alpha,
        },
        4 => DecayFormula::Step,
        _ => {
            return Err(Status::invalid_argument(format!(
                "unknown decay_formula_type: {formula_type}"
            )));
        }
    };

    let anchor = def.anchor_field.clone();
    if anchor.is_empty() {
        return Err(Status::invalid_argument(
            "computed_property: anchor_field must not be empty",
        ));
    }
    if def.duration_secs == 0 {
        return Err(Status::invalid_argument(
            "computed_property: duration_secs must be > 0",
        ));
    }

    match computed_type {
        1 => {
            // TTL
            let scope = match def.scope {
                0 | 3 => TtlScope::Node, // UNSPECIFIED defaults to Node
                1 => TtlScope::Field,
                2 => TtlScope::Subtree,
                _ => {
                    return Err(Status::invalid_argument(format!(
                        "unknown ttl_scope: {}",
                        def.scope
                    )));
                }
            };
            let target_field = if def.target_field.is_empty() {
                None
            } else {
                Some(def.target_field.clone())
            };
            Ok(ComputedSpec::Ttl {
                duration_secs: def.duration_secs,
                anchor_field: anchor,
                scope,
                target_field,
            })
        }
        2 => {
            // Decay
            Ok(ComputedSpec::Decay {
                formula,
                initial: def.initial,
                target: def.target,
                duration_secs: def.duration_secs,
                anchor_field: anchor,
            })
        }
        3 => {
            // VectorDecay
            Ok(ComputedSpec::VectorDecay {
                formula,
                duration_secs: def.duration_secs,
                anchor_field: anchor,
            })
        }
        0 => Err(Status::invalid_argument(
            "computed_property: computed_type must not be UNSPECIFIED",
        )),
        _ => Err(Status::invalid_argument(format!(
            "unknown computed_type: {computed_type}"
        ))),
    }
}

/// Convert an internal `ComputedSpec` to a proto `ComputedPropertyDefinition`.
fn computed_spec_to_proto(name: &str, spec: &ComputedSpec) -> graph::ComputedPropertyDefinition {
    let mut def = graph::ComputedPropertyDefinition {
        name: name.to_string(),
        ..Default::default()
    };

    match spec {
        ComputedSpec::Ttl {
            duration_secs,
            anchor_field,
            scope,
            target_field,
        } => {
            def.computed_type = 1; // TTL
            def.duration_secs = *duration_secs;
            def.anchor_field = anchor_field.clone();
            def.scope = match scope {
                TtlScope::Field => 1,
                TtlScope::Subtree => 2,
                TtlScope::Node => 3,
            };
            def.target_field = target_field.clone().unwrap_or_default();
        }
        ComputedSpec::Decay {
            formula,
            initial,
            target,
            duration_secs,
            anchor_field,
        } => {
            def.computed_type = 2; // Decay
            def.duration_secs = *duration_secs;
            def.anchor_field = anchor_field.clone();
            def.initial = *initial;
            def.target = *target;
            set_formula_fields(&mut def, formula);
        }
        ComputedSpec::VectorDecay {
            formula,
            duration_secs,
            anchor_field,
        } => {
            def.computed_type = 3; // VectorDecay
            def.duration_secs = *duration_secs;
            def.anchor_field = anchor_field.clone();
            set_formula_fields(&mut def, formula);
        }
    }

    def
}

fn set_formula_fields(def: &mut graph::ComputedPropertyDefinition, formula: &DecayFormula) {
    match formula {
        DecayFormula::Linear => {
            def.formula_type = 1;
        }
        DecayFormula::Exponential { lambda } => {
            def.formula_type = 2;
            def.lambda = *lambda;
        }
        DecayFormula::PowerLaw { tau, alpha } => {
            def.formula_type = 3;
            def.tau = *tau;
            def.alpha = *alpha;
        }
        DecayFormula::Step => {
            def.formula_type = 4;
        }
    }
}

/// Convert a proto `SchemaMode` integer to the internal `SchemaMode`.
///
/// Proto values:
///   0 = UNSPECIFIED → defaults to STRICT (most type-safe)
///   1 = STRICT, 2 = VALIDATED, 3 = FLEXIBLE
fn proto_to_schema_mode(v: i32) -> SchemaMode {
    match v {
        2 => SchemaMode::Validated,
        3 => SchemaMode::Flexible,
        _ => SchemaMode::Strict, // UNSPECIFIED (0) and STRICT (1) both → Strict
    }
}

/// Convert an internal `SchemaMode` to a proto integer.
fn schema_mode_to_proto(mode: SchemaMode) -> i32 {
    match mode {
        SchemaMode::Strict => 1,
        SchemaMode::Validated => 2,
        SchemaMode::Flexible => 3,
    }
}

/// Map internal `PropertyType` to proto integer for list_labels responses.
fn property_type_to_proto(pt: &PropertyType) -> i32 {
    match pt {
        PropertyType::Int => 1,
        PropertyType::Float => 2,
        PropertyType::String => 3,
        PropertyType::Bool => 4,
        PropertyType::Binary | PropertyType::Blob => 5,
        PropertyType::Timestamp => 6,
        PropertyType::Vector { .. } => 7,
        PropertyType::Array(_) => 8,
        PropertyType::Map => 9,
        // Document, Geo, Computed have no direct proto representation → MAP as closest.
        _ => 9,
    }
}

pub struct SchemaServiceImpl {
    database: Arc<Mutex<Database>>,
}

impl SchemaServiceImpl {
    pub fn new(database: Arc<Mutex<Database>>) -> Self {
        Self { database }
    }
}

#[tonic::async_trait]
impl graph::schema_service_server::SchemaService for SchemaServiceImpl {
    async fn create_label(
        &self,
        request: Request<graph::CreateLabelRequest>,
    ) -> Result<Response<graph::Label>, Status> {
        let req = request.into_inner();
        // Build internal LabelSchema from the proto request.
        let mut schema = LabelSchema::new(&req.name);

        // Apply schema mode (defaults to STRICT when unspecified).
        let mode = proto_to_schema_mode(req.schema_mode);
        schema.set_mode(mode);

        for prop_def in &req.properties {
            let property_type = proto_type_to_property_type(prop_def.r#type);
            let mut prop = PropertyDef::new(&prop_def.name, property_type);
            if prop_def.required {
                prop = prop.not_null();
            }
            if prop_def.unique {
                prop = prop.unique();
            }
            schema.add_property(prop);
        }

        // Wire COMPUTED property specs into the schema.
        // TtlReaper picks up TTL specs automatically; DECAY/VECTOR_DECAY are
        // evaluated inline at query time during MATCH/RETURN execution.
        let mut echo_computed: Vec<graph::ComputedPropertyDefinition> =
            Vec::with_capacity(req.computed_properties.len());
        for cp in &req.computed_properties {
            let spec = proto_to_computed_spec(cp)?;
            schema.add_property(PropertyDef::computed(&cp.name, spec));
            echo_computed.push(cp.clone());
        }

        let version = {
            let mut db = self.database.lock().unwrap_or_else(|e| e.into_inner());
            db.create_label_schema(schema)
                .map_err(|e| Status::internal(format!("create_label error: {e}")))?
        };

        Ok(Response::new(graph::Label {
            name: req.name,
            properties: req.properties,
            version,
            computed_properties: echo_computed,
            schema_mode: schema_mode_to_proto(mode),
        }))
    }

    async fn create_edge_type(
        &self,
        request: Request<graph::CreateEdgeTypeRequest>,
    ) -> Result<Response<graph::EdgeType>, Status> {
        let req = request.into_inner();

        // Build internal EdgeTypeSchema from the proto request.
        let mut schema = EdgeTypeSchema::new(&req.name);
        for prop_def in &req.properties {
            let property_type = proto_type_to_property_type(prop_def.r#type);
            let mut prop = PropertyDef::new(&prop_def.name, property_type);
            if prop_def.required {
                prop = prop.not_null();
            }
            if prop_def.unique {
                prop = prop.unique();
            }
            schema.add_property(prop);
        }

        let version = {
            let mut db = self.database.lock().unwrap_or_else(|e| e.into_inner());
            db.create_edge_type_schema(schema)
                .map_err(|e| Status::internal(format!("create_edge_type error: {e}")))?
        };

        Ok(Response::new(graph::EdgeType {
            name: req.name,
            properties: req.properties,
            version,
        }))
    }

    async fn list_labels(
        &self,
        _request: Request<graph::ListLabelsRequest>,
    ) -> Result<Response<graph::ListLabelsResponse>, Status> {
        // Two-pass: first load declared schemas (with property metadata),
        // then add any undeclared labels discovered from existing nodes.
        const SCHEMA_PREFIX: &[u8] = b"schema:label:";

        let mut db = self.database.lock().unwrap_or_else(|e| e.into_inner());

        // Pass 1: scan `schema:label:*` for persisted LabelSchema entries.
        let mut label_map: std::collections::BTreeMap<String, graph::Label> = {
            let iter = db
                .engine()
                .prefix_scan(Partition::Schema, SCHEMA_PREFIX)
                .map_err(|e| Status::internal(format!("list_labels scan error: {e}")))?;

            let mut map = std::collections::BTreeMap::new();
            for guard in iter {
                let Ok((key, val_bytes)) = guard.into_inner() else {
                    continue;
                };
                let Ok(name) = std::str::from_utf8(&key[SCHEMA_PREFIX.len()..]) else {
                    continue;
                };
                if name.is_empty() {
                    continue;
                }
                // Decode the persisted LabelSchema and convert properties.
                if let Ok(schema) =
                    coordinode_core::schema::definition::LabelSchema::from_msgpack(&val_bytes)
                {
                    let mut properties = Vec::new();
                    let mut computed_properties = Vec::new();

                    for p in schema.properties.values() {
                        if let PropertyType::Computed(ref spec) = p.property_type {
                            computed_properties.push(computed_spec_to_proto(&p.name, spec));
                        } else {
                            properties.push(graph::PropertyDefinition {
                                name: p.name.clone(),
                                r#type: property_type_to_proto(&p.property_type),
                                required: p.not_null,
                                unique: p.unique,
                            });
                        }
                    }

                    map.insert(
                        schema.name.clone(),
                        graph::Label {
                            name: schema.name.clone(),
                            properties,
                            version: schema.version,
                            computed_properties,
                            schema_mode: schema_mode_to_proto(schema.mode),
                        },
                    );
                } else {
                    // Unreadable schema entry — still expose the name.
                    map.entry(name.to_string()).or_insert_with(|| graph::Label {
                        name: name.to_string(),
                        properties: vec![],
                        version: 0,
                        computed_properties: vec![],
                        schema_mode: schema_mode_to_proto(SchemaMode::Strict),
                    });
                }
            }
            map
        };

        // Pass 2: discover undeclared labels from existing nodes via Cypher.
        // Note: `label` is a Cypher reserved keyword — use `lbl` as alias.
        let rows = db
            .execute_cypher("MATCH (n) RETURN DISTINCT n.__label__ AS lbl ORDER BY lbl")
            .map_err(|e| Status::internal(format!("list_labels cypher error: {e}")))?;

        for row in rows {
            if let Some(Value::String(name)) = row.get("lbl") {
                if !name.is_empty() {
                    label_map
                        .entry(name.clone())
                        .or_insert_with(|| graph::Label {
                            name: name.clone(),
                            properties: vec![],
                            version: 0,
                            computed_properties: vec![],
                            schema_mode: 0, // UNSPECIFIED — no declared schema
                        });
                }
            }
        }

        let mut labels: Vec<graph::Label> = label_map.into_values().collect();
        labels.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(Response::new(graph::ListLabelsResponse { labels }))
    }

    async fn list_edge_types(
        &self,
        _request: Request<graph::ListEdgeTypesRequest>,
    ) -> Result<Response<graph::ListEdgeTypesResponse>, Status> {
        // Edge types are registered in the Schema partition under the key prefix
        // `schema:edge_type:<name>` whenever an edge is created. Read directly
        // from there — wildcard MATCH ()-[r]->() won't work because the executor
        // only traverses explicitly-typed edges (empty edge_types slice = no-op).
        const PREFIX: &[u8] = b"schema:edge_type:";
        let names: Vec<String> = {
            let db = self.database.lock().unwrap_or_else(|e| e.into_inner());
            let iter = db
                .engine()
                .prefix_scan(Partition::Schema, PREFIX)
                .map_err(|e| Status::internal(format!("list_edge_types scan error: {e}")))?;

            let mut types = Vec::new();
            for guard in iter {
                if let Ok((key, _)) = guard.into_inner() {
                    if let Ok(name) = std::str::from_utf8(&key[PREFIX.len()..]) {
                        if !name.is_empty() {
                            types.push(name.to_string());
                        }
                    }
                }
            }
            types.sort();
            types
        };

        let edge_types: Vec<graph::EdgeType> = names
            .into_iter()
            .map(|name| graph::EdgeType {
                name,
                properties: vec![],
                version: 0,
            })
            .collect();

        Ok(Response::new(graph::ListEdgeTypesResponse { edge_types }))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::proto::graph::schema_service_server::SchemaService;

    fn test_service() -> (SchemaServiceImpl, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let database = Arc::new(Mutex::new(
            Database::open(dir.path()).expect("open database"),
        ));
        (SchemaServiceImpl::new(database), dir)
    }

    /// list_labels returns empty on a fresh database.
    #[tokio::test]
    async fn list_labels_empty_db() {
        let (svc, _dir) = test_service();

        let resp = svc
            .list_labels(Request::new(graph::ListLabelsRequest {}))
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
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher("CREATE (n:Person {name: 'Alice'})")
                .expect("create");
            db.execute_cypher("CREATE (n:City {name: 'Kyiv'})")
                .expect("create");
        }

        let labels: Vec<String> = svc
            .list_labels(Request::new(graph::ListLabelsRequest {}))
            .await
            .expect("list_labels should succeed")
            .into_inner()
            .labels
            .into_iter()
            .map(|l| l.name)
            .collect();

        assert!(
            labels.contains(&"Person".to_string()),
            "should contain Person, got: {labels:?}"
        );
        assert!(
            labels.contains(&"City".to_string()),
            "should contain City, got: {labels:?}"
        );
    }

    /// list_edge_types returns empty on a fresh database.
    #[tokio::test]
    async fn list_edge_types_empty_db() {
        let (svc, _dir) = test_service();

        let resp = svc
            .list_edge_types(Request::new(graph::ListEdgeTypesRequest {}))
            .await
            .expect("list_edge_types should succeed");

        assert_eq!(resp.into_inner().edge_types.len(), 0);
    }

    /// list_edge_types returns edge types that exist in the database.
    #[tokio::test]
    async fn list_edge_types_returns_existing_types() {
        let (svc, _dir) = test_service();

        {
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher("CREATE (a:P {name: 'A'})-[:KNOWS]->(b:P {name: 'B'})")
                .expect("create");
            db.execute_cypher("CREATE (c:P {name: 'C'})-[:LIKES]->(d:P {name: 'D'})")
                .expect("create");
        }

        let edge_types: Vec<String> = svc
            .list_edge_types(Request::new(graph::ListEdgeTypesRequest {}))
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

    // ── R-API1: create_label / create_edge_type now persists schema ──

    /// create_label persists schema and returns version > 0.
    #[tokio::test]
    async fn create_label_persists_schema() {
        let (svc, _dir) = test_service();

        let resp = svc
            .create_label(Request::new(graph::CreateLabelRequest {
                name: "Article".to_string(),
                properties: vec![
                    graph::PropertyDefinition {
                        name: "title".to_string(),
                        r#type: 3, // STRING
                        required: true,
                        unique: false,
                    },
                    graph::PropertyDefinition {
                        name: "slug".to_string(),
                        r#type: 3, // STRING
                        required: false,
                        unique: true,
                    },
                ],
                computed_properties: vec![],
                schema_mode: 0, // UNSPECIFIED → STRICT
            }))
            .await
            .expect("create_label should succeed");

        let label = resp.into_inner();
        assert_eq!(label.name, "Article");
        assert!(label.version > 0, "version must be positive after persist");
        assert_eq!(label.properties.len(), 2);

        // Verify schema is in storage.
        use coordinode_storage::engine::partition::Partition;
        let db = svc.database.lock().unwrap();
        let key = coordinode_core::schema::definition::encode_label_schema_key("Article");
        let bytes = db
            .engine()
            .get(Partition::Schema, &key)
            .expect("storage get")
            .expect("schema must be persisted");
        let schema = coordinode_core::schema::definition::LabelSchema::from_msgpack(&bytes)
            .expect("deserialize");
        assert_eq!(schema.name, "Article");
        assert_eq!(schema.properties.len(), 2);
        // `slug` must be unique.
        assert!(schema.get_property("slug").is_some_and(|p| p.unique));
    }

    /// create_label with unique property → duplicate CREATE fails.
    #[tokio::test]
    async fn create_label_unique_property_enforces_constraint() {
        let (svc, _dir) = test_service();

        // Declare label with unique email.
        svc.create_label(Request::new(graph::CreateLabelRequest {
            name: "Customer".to_string(),
            properties: vec![graph::PropertyDefinition {
                name: "email".to_string(),
                r#type: 3, // STRING
                required: false,
                unique: true,
            }],
            computed_properties: vec![],
            schema_mode: 0, // UNSPECIFIED → STRICT
        }))
        .await
        .expect("create_label");

        // First node — should succeed.
        {
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher("CREATE (c:Customer {email: 'x@test.com'})")
                .expect("first create");
        }

        // Second node with same email — must fail.
        {
            let mut db = svc.database.lock().unwrap();
            let result = db.execute_cypher("CREATE (c:Customer {email: 'x@test.com'})");
            assert!(
                result.is_err(),
                "duplicate email must be rejected by unique constraint"
            );
        }
    }

    /// create_edge_type persists schema and returns version > 0.
    #[tokio::test]
    async fn create_edge_type_persists_schema() {
        let (svc, _dir) = test_service();

        let resp = svc
            .create_edge_type(Request::new(graph::CreateEdgeTypeRequest {
                name: "FOLLOWS".to_string(),
                properties: vec![graph::PropertyDefinition {
                    name: "since".to_string(),
                    r#type: 6, // TIMESTAMP
                    required: true,
                    unique: false,
                }],
            }))
            .await
            .expect("create_edge_type should succeed");

        let et = resp.into_inner();
        assert_eq!(et.name, "FOLLOWS");
        assert!(et.version > 0);

        // Verify in storage.
        use coordinode_storage::engine::partition::Partition;
        let db = svc.database.lock().unwrap();
        let key = coordinode_core::schema::definition::encode_edge_type_schema_key("FOLLOWS");
        let bytes = db
            .engine()
            .get(Partition::Schema, &key)
            .expect("storage get")
            .expect("edge schema must be persisted");
        let schema = coordinode_core::schema::definition::EdgeTypeSchema::from_msgpack(&bytes)
            .expect("deserialize");
        assert_eq!(schema.name, "FOLLOWS");
        assert_eq!(schema.properties.len(), 1);
        assert!(schema.get_property("since").is_some_and(|p| p.not_null));
    }

    /// list_labels returns schema properties for declared labels.
    #[tokio::test]
    async fn list_labels_returns_schema_properties() {
        let (svc, _dir) = test_service();

        // Declare label with known properties.
        svc.create_label(Request::new(graph::CreateLabelRequest {
            name: "Post".to_string(),
            properties: vec![
                graph::PropertyDefinition {
                    name: "title".to_string(),
                    r#type: 3, // STRING
                    required: true,
                    unique: false,
                },
                graph::PropertyDefinition {
                    name: "views".to_string(),
                    r#type: 1, // INT64
                    required: false,
                    unique: false,
                },
            ],
            computed_properties: vec![],
            schema_mode: 0, // UNSPECIFIED → STRICT
        }))
        .await
        .expect("create_label");

        let labels = svc
            .list_labels(Request::new(graph::ListLabelsRequest {}))
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

    // ── R-API4: ComputedPropertyDefinition via gRPC ──────────────────────────

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
        svc.create_label(Request::new(graph::CreateLabelRequest {
            name: "Session".to_string(),
            properties: vec![graph::PropertyDefinition {
                name: "started_at".to_string(),
                r#type: 6, // TIMESTAMP
                required: true,
                unique: false,
            }],
            computed_properties: vec![graph::ComputedPropertyDefinition {
                name: "_ttl".to_string(),
                computed_type: 1,  // TTL
                formula_type: 0,   // unspecified (not used for TTL)
                duration_secs: 60, // 60 seconds lifetime
                anchor_field: "started_at".to_string(),
                scope: 3, // NODE
                ..Default::default()
            }],
            schema_mode: 3, // FLEXIBLE — test exercises TTL reaper, not schema enforcement
        }))
        .await
        .expect("create_label with TTL should succeed");

        // Verify the response echoes computed_properties.
        let resp = svc
            .create_label(Request::new(graph::CreateLabelRequest {
                name: "Session2".to_string(),
                properties: vec![graph::PropertyDefinition {
                    name: "started_at".to_string(),
                    r#type: 6, // TIMESTAMP
                    required: true,
                    unique: false,
                }],
                computed_properties: vec![graph::ComputedPropertyDefinition {
                    name: "_ttl".to_string(),
                    computed_type: 1,
                    formula_type: 0,
                    duration_secs: 60,
                    anchor_field: "started_at".to_string(),
                    scope: 3,
                    ..Default::default()
                }],
                schema_mode: 3, // FLEXIBLE — test exercises TTL reaper, not schema enforcement
            }))
            .await
            .expect("create_label should succeed");

        let label = resp.into_inner();
        assert_eq!(
            label.computed_properties.len(),
            1,
            "response must echo computed_properties"
        );
        assert_eq!(label.computed_properties[0].name, "_ttl");
        assert_eq!(label.computed_properties[0].computed_type, 1); // TTL
        assert_eq!(label.computed_properties[0].duration_secs, 60);
        assert_eq!(label.computed_properties[0].anchor_field, "started_at");

        // Verify schema was persisted correctly: LabelSchema must contain ComputedSpec::Ttl.
        {
            use coordinode_core::schema::computed::{ComputedSpec, TtlScope};
            use coordinode_core::schema::definition::PropertyType;
            use coordinode_storage::engine::partition::Partition;

            let db = svc.database.lock().unwrap();
            let key = coordinode_core::schema::definition::encode_label_schema_key("Session");
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
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher(&format!("CREATE (s:Session {{started_at: {expired_us}}})"))
                .expect("create expired session");
            db.execute_cypher(&format!("CREATE (s:Session {{started_at: {now_us}}})"))
                .expect("create fresh session");
        }

        // Verify both nodes exist before reaping.
        {
            let mut db = svc.database.lock().unwrap();
            let rows = db
                .execute_cypher("MATCH (s:Session) RETURN s.started_at AS ts")
                .expect("match before reap");
            assert_eq!(rows.len(), 2, "must have 2 Session nodes before reap");
        }

        // Run TtlReaper directly — no background thread, no waiting.
        let result = {
            let db = svc.database.lock().unwrap();
            coordinode_query::index::ttl_reaper::reap_computed_ttl(&db.engine_shared(), 1, 1000)
        };

        assert_eq!(
            result.nodes_deleted, 1,
            "reaper must delete exactly 1 expired Session node"
        );

        // Verify only the fresh node remains.
        {
            let mut db = svc.database.lock().unwrap();
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

        svc.create_label(Request::new(graph::CreateLabelRequest {
            name: "Article".to_string(),
            properties: vec![graph::PropertyDefinition {
                name: "published_at".to_string(),
                r#type: 6, // TIMESTAMP
                required: true,
                unique: false,
            }],
            computed_properties: vec![graph::ComputedPropertyDefinition {
                name: "relevance".to_string(),
                computed_type: 2, // DECAY
                formula_type: 1,  // LINEAR
                initial: 1.0,
                target: 0.0,
                duration_secs: 604800, // 7 days
                anchor_field: "published_at".to_string(),
                ..Default::default()
            }],
            schema_mode: 0, // UNSPECIFIED → STRICT
        }))
        .await
        .expect("create_label with DECAY should succeed");

        // list_labels must return the computed property back to the caller.
        let labels = svc
            .list_labels(Request::new(graph::ListLabelsRequest {}))
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
        assert_eq!(cp.computed_type, 2); // DECAY
        assert_eq!(cp.formula_type, 1); // LINEAR
        assert!((cp.initial - 1.0).abs() < f64::EPSILON);
        assert!((cp.target - 0.0).abs() < f64::EPSILON);
        assert_eq!(cp.duration_secs, 604800);
        assert_eq!(cp.anchor_field, "published_at");
    }

    /// proto_to_computed_spec rejects COMPUTED_TYPE_UNSPECIFIED.
    #[tokio::test]
    async fn create_label_computed_type_unspecified_returns_error() {
        let (svc, _dir) = test_service();

        let result = svc
            .create_label(Request::new(graph::CreateLabelRequest {
                name: "Bad".to_string(),
                properties: vec![],
                computed_properties: vec![graph::ComputedPropertyDefinition {
                    name: "broken".to_string(),
                    computed_type: 0, // UNSPECIFIED
                    duration_secs: 60,
                    anchor_field: "ts".to_string(),
                    ..Default::default()
                }],
                schema_mode: 0,
            }))
            .await;

        assert!(
            result.is_err(),
            "UNSPECIFIED computed_type must be rejected"
        );
    }

    /// proto_to_computed_spec rejects empty anchor_field.
    #[tokio::test]
    async fn create_label_computed_empty_anchor_field_returns_error() {
        let (svc, _dir) = test_service();

        let result = svc
            .create_label(Request::new(graph::CreateLabelRequest {
                name: "Bad2".to_string(),
                properties: vec![],
                computed_properties: vec![graph::ComputedPropertyDefinition {
                    name: "ttl".to_string(),
                    computed_type: 1, // TTL
                    duration_secs: 60,
                    anchor_field: String::new(), // empty — must be rejected
                    ..Default::default()
                }],
                schema_mode: 0,
            }))
            .await;

        assert!(result.is_err(), "empty anchor_field must be rejected");
    }

    // ── R-API5: SchemaMode via gRPC — CREATE/SET enforcement ────────────────

    /// create_label with schema_mode=STRICT (1) persists the mode; SET of an
    /// unknown property on a STRICT label is rejected.
    ///
    /// Regression: schema_mode field must flow from proto → LabelSchema → executor
    /// enforcement. STRICT = no undeclared properties allowed at write time.
    #[tokio::test]
    async fn strict_mode_set_unknown_property_rejected() {
        let (svc, _dir) = test_service();

        // Declare User label with schema_mode=STRICT, only `name` declared.
        svc.create_label(Request::new(graph::CreateLabelRequest {
            name: "User".to_string(),
            properties: vec![graph::PropertyDefinition {
                name: "name".to_string(),
                r#type: 3, // STRING
                required: false,
                unique: false,
            }],
            computed_properties: vec![],
            schema_mode: 1, // STRICT
        }))
        .await
        .expect("create_label should succeed");

        // Verify schema was persisted with STRICT mode.
        {
            use coordinode_core::schema::definition::SchemaMode;
            use coordinode_storage::engine::partition::Partition;

            let db = svc.database.lock().unwrap();
            let key = coordinode_core::schema::definition::encode_label_schema_key("User");
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
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher("CREATE (u:User {name: 'Alice'})")
                .expect("CREATE with declared property must succeed on STRICT label");
        }

        // SET an unknown property on the STRICT label — must be rejected.
        {
            let mut db = svc.database.lock().unwrap();
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

    /// create_label with schema_mode=FLEXIBLE (3) — SET of any property is allowed.
    ///
    /// Regression: FLEXIBLE mode must pass through all properties without enforcement.
    #[tokio::test]
    async fn flexible_mode_set_unknown_property_allowed() {
        let (svc, _dir) = test_service();

        // Declare Device label with schema_mode=FLEXIBLE, only `id` declared.
        svc.create_label(Request::new(graph::CreateLabelRequest {
            name: "Device".to_string(),
            properties: vec![graph::PropertyDefinition {
                name: "id".to_string(),
                r#type: 3, // STRING
                required: false,
                unique: false,
            }],
            computed_properties: vec![],
            schema_mode: 3, // FLEXIBLE
        }))
        .await
        .expect("create_label should succeed");

        // CREATE a Device node.
        {
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher("CREATE (d:Device {id: 'd1'})")
                .expect("CREATE should succeed");
        }

        // SET an undeclared property — must succeed on FLEXIBLE label.
        {
            let mut db = svc.database.lock().unwrap();
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
        let resp = svc
            .create_label(Request::new(graph::CreateLabelRequest {
                name: "Event".to_string(),
                properties: vec![graph::PropertyDefinition {
                    name: "ts".to_string(),
                    r#type: 6, // TIMESTAMP
                    required: false,
                    unique: false,
                }],
                computed_properties: vec![],
                schema_mode: 2, // VALIDATED
            }))
            .await
            .expect("create_label should succeed");

        let label = resp.into_inner();
        assert_eq!(
            label.schema_mode, 2,
            "create_label response must echo schema_mode=VALIDATED(2)"
        );

        // list_labels must return the persisted mode.
        let labels = svc
            .list_labels(Request::new(graph::ListLabelsRequest {}))
            .await
            .expect("list_labels")
            .into_inner()
            .labels;

        let event = labels
            .iter()
            .find(|l| l.name == "Event")
            .expect("Event label must be in list");

        assert_eq!(
            event.schema_mode, 2,
            "list_labels must return schema_mode=VALIDATED(2) for Event"
        );
    }

    /// STRICT label rejects type mismatch at CREATE time.
    ///
    /// Regression: type validation for declared properties must be applied at
    /// write time in STRICT mode, not silently stored with wrong type.
    #[tokio::test]
    async fn strict_mode_create_type_mismatch_rejected() {
        let (svc, _dir) = test_service();

        svc.create_label(Request::new(graph::CreateLabelRequest {
            name: "Sensor".to_string(),
            properties: vec![graph::PropertyDefinition {
                name: "reading".to_string(),
                r#type: 2, // FLOAT64
                required: false,
                unique: false,
            }],
            computed_properties: vec![],
            schema_mode: 1, // STRICT
        }))
        .await
        .expect("create_label should succeed");

        // CREATE with wrong type for declared property — must be rejected.
        let mut db = svc.database.lock().unwrap();
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

        svc.create_label(Request::new(graph::CreateLabelRequest {
            name: "Log".to_string(),
            properties: vec![graph::PropertyDefinition {
                name: "level".to_string(),
                r#type: 1, // INT64
                required: false,
                unique: false,
            }],
            computed_properties: vec![],
            schema_mode: 2, // VALIDATED
        }))
        .await
        .expect("create_label should succeed");

        {
            let mut db = svc.database.lock().unwrap();

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
        svc.create_label(Request::new(graph::CreateLabelRequest {
            name: "Product".to_string(),
            properties: vec![graph::PropertyDefinition {
                name: "sku".to_string(),
                r#type: 3, // STRING
                required: false,
                unique: false,
            }],
            computed_properties: vec![],
            schema_mode: 1, // STRICT
        }))
        .await
        .expect("create_label should succeed");

        // CREATE with only the declared property — must succeed.
        {
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher("CREATE (p:Product {sku: 'SKU-001'})")
                .expect("CREATE with declared property must succeed on STRICT label");
        }

        // CREATE with an undeclared property — must be rejected.
        {
            let mut db = svc.database.lock().unwrap();
            let result =
                db.execute_cypher("CREATE (p:Product {sku: 'SKU-002', extra_field: 'forbidden'})");
            assert!(
                result.is_err(),
                "CREATE with undeclared property must be rejected on STRICT label, got: {result:?}"
            );
        }
    }
}
