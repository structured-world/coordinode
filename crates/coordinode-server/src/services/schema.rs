use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};

use coordinode_core::graph::types::{Value, VectorMetric};
use coordinode_core::schema::definition::{EdgeTypeSchema, LabelSchema, PropertyDef, PropertyType};
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
            db.create_label_schema(schema)
                .map_err(|e| Status::internal(format!("create_label error: {e}")))?
        };

        Ok(Response::new(graph::Label {
            name: req.name,
            properties: req.properties,
            version,
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
                    let properties = schema
                        .properties
                        .values()
                        .map(|p| graph::PropertyDefinition {
                            name: p.name.clone(),
                            r#type: property_type_to_proto(&p.property_type),
                            required: p.not_null,
                            unique: p.unique,
                        })
                        .collect();
                    map.insert(
                        schema.name.clone(),
                        graph::Label {
                            name: schema.name,
                            properties,
                            version: schema.version,
                        },
                    );
                } else {
                    // Unreadable schema entry — still expose the name.
                    map.entry(name.to_string()).or_insert_with(|| graph::Label {
                        name: name.to_string(),
                        properties: vec![],
                        version: 0,
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
}
