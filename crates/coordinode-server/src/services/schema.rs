use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;

use crate::proto::graph;

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
        // Label schemas are optional (VALIDATED mode). A label without a schema
        // works in free-form mode. Return the request as-is — label creation is
        // implicit on first node write and does not require explicit declaration.
        Ok(Response::new(graph::Label {
            name: req.name,
            properties: req.properties,
            version: 1,
        }))
    }

    async fn create_edge_type(
        &self,
        request: Request<graph::CreateEdgeTypeRequest>,
    ) -> Result<Response<graph::EdgeType>, Status> {
        let req = request.into_inner();
        Ok(Response::new(graph::EdgeType {
            name: req.name,
            properties: req.properties,
            version: 1,
        }))
    }

    async fn list_labels(
        &self,
        _request: Request<graph::ListLabelsRequest>,
    ) -> Result<Response<graph::ListLabelsResponse>, Status> {
        // Discover labels by scanning all nodes and collecting distinct primary
        // labels (stored as n.__label__ by NodeScan). DISTINCT deduplicated by
        // the executor; ORDER BY gives stable output.
        // Note: `label` is a Cypher reserved keyword — use `lbl` as alias.
        let rows = {
            let mut db = self.database.lock().unwrap_or_else(|e| e.into_inner());
            db.execute_cypher("MATCH (n) RETURN DISTINCT n.__label__ AS lbl ORDER BY lbl")
                .map_err(|e| Status::internal(format!("list_labels error: {e}")))?
        };

        let labels: Vec<graph::Label> = rows
            .into_iter()
            .filter_map(|row| {
                if let Some(Value::String(name)) = row.get("lbl") {
                    if !name.is_empty() {
                        return Some(graph::Label {
                            name: name.clone(),
                            properties: vec![],
                            version: 0,
                        });
                    }
                }
                None
            })
            .collect();

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
}
