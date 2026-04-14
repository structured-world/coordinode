//! Vector index regression tests.
//!
//! ## Test matrix
//!
//! | Test | Bug | Scenario |
//! |------|-----|---------|
//! | `g082_set_updates_hnsw_graph_position` | G082 | `MATCH (n) SET n.emb = $vec` must update the indexed vector so that subsequent vector similarity searches reflect the new value. |
//!
//! ## Running
//!
//! ```bash
//! cargo build -p coordinode-server
//! cargo nextest run -p coordinode-integration
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_integration::harness::CoordinodeProcess;
use coordinode_integration::proto::common::{
    property_value::Value as PvKind, PropertyValue, Vector,
};
use coordinode_integration::proto::graph::{
    CreateLabelRequest, PropertyDefinition, PropertyType, SchemaMode,
};
use coordinode_integration::proto::query::{ExecuteCypherRequest, Row};
use std::collections::HashMap;

// ── Helpers ───────────────────────────────────────────────────────────────────

async fn cypher(
    proc: &CoordinodeProcess,
    query: &str,
    params: HashMap<String, PropertyValue>,
) -> Result<Vec<HashMap<String, PropertyValue>>, tonic::Status> {
    let mut client = proc.cypher_client().await;
    let resp = client
        .execute_cypher(ExecuteCypherRequest {
            query: query.to_string(),
            parameters: params,
            read_preference: 0,
            read_concern: None,
        })
        .await?
        .into_inner();

    let columns = resp.columns;
    let rows = resp
        .rows
        .into_iter()
        .map(|Row { values }| {
            columns
                .iter()
                .zip(values)
                .map(|(col, v)| (col.clone(), v))
                .collect::<HashMap<_, _>>()
        })
        .collect();
    Ok(rows)
}

fn pv_vector(values: Vec<f32>) -> PropertyValue {
    PropertyValue {
        value: Some(PvKind::VectorValue(Vector { values })),
    }
}

// ── G082: SET updates vector index ────────────────────────────────────────────

/// G082 — `MATCH (n) SET n.emb = $vec` must update the stored vector so that
/// subsequent `vector_similarity()` queries reflect the new value.
///
/// Root cause: `HnswIndex::insert()` returned early ("Already indexed") when
/// called for an existing node ID, so the HNSW graph position was stale after
/// a SET.  The node was also kept at its old position in the flat (linear scan)
/// index used by `vector_similarity()`.
///
/// Fix: `insert()` now calls `update_existing_node()` which removes stale
/// connections and rebuilds neighbourhood from the new vector.
///
/// Vector layout (4D cosine, no ties):
///   A: [0.0, 1.0, 0.0, 0.0]  "up"        — will be moved to "right" via SET
///   B: [0.0, 0.8, 0.6, 0.0]  "mostly up" — strictly wins "up" after A moves
///   C: [0.0, 0.0, 0.0, 1.0]  "depth"     — orthogonal, low score for both queries
///
/// Cosine scores after SET (A → [1,0,0,0]):
///   query "right": cos(A_new)=1.0, cos(B)=0, cos(C)=0  → A wins ✓
///   query "up":    cos(A_new)=0,   cos(B)≈0.8, cos(C)=0 → B wins, not A ✓
#[tokio::test]
async fn g082_set_updates_hnsw_graph_position() {
    let proc = CoordinodeProcess::start().await;

    let mut sc = proc.schema_client().await;
    sc.create_label(CreateLabelRequest {
        name: "Item".to_string(),
        properties: vec![PropertyDefinition {
            name: "emb".to_string(),
            r#type: PropertyType::Vector as i32,
            required: false,
            unique: false,
        }],
        computed_properties: vec![],
        schema_mode: SchemaMode::Flexible as i32,
    })
    .await
    .expect("create_label");

    let mut p = HashMap::new();
    p.insert("v".to_string(), pv_vector(vec![0.0, 1.0, 0.0, 0.0]));
    cypher(&proc, "CREATE (n:Item {emb: $v, name: 'A'})", p)
        .await
        .expect("create A");

    let mut p = HashMap::new();
    p.insert("v".to_string(), pv_vector(vec![0.0, 0.8, 0.6, 0.0]));
    cypher(&proc, "CREATE (n:Item {emb: $v, name: 'B'})", p)
        .await
        .expect("create B");

    let mut p = HashMap::new();
    p.insert("v".to_string(), pv_vector(vec![0.0, 0.0, 0.0, 1.0]));
    cypher(&proc, "CREATE (n:Item {emb: $v, name: 'C'})", p)
        .await
        .expect("create C");

    // Sanity: before SET, "up" → A is top-1 (cos=1.0 > B's ≈0.8).
    let mut p = HashMap::new();
    p.insert("q".to_string(), pv_vector(vec![0.0, 1.0, 0.0, 0.0]));
    let before = cypher(
        &proc,
        "MATCH (n:Item) RETURN n.name AS name, vector_similarity(n.emb, $q) AS score \
         ORDER BY score DESC LIMIT 1",
        p,
    )
    .await
    .expect("search before SET");
    let top_before = before
        .first()
        .and_then(|r| r.get("name"))
        .and_then(|v| match &v.value {
            Some(PvKind::StringValue(s)) => Some(s.clone()),
            _ => None,
        });
    assert_eq!(
        top_before.as_deref(),
        Some("A"),
        "before SET: 'up' query should find A (cos=1.0)"
    );

    // Move A from "up" → "right" via SET.
    let mut p = HashMap::new();
    p.insert("v".to_string(), pv_vector(vec![1.0, 0.0, 0.0, 0.0]));
    cypher(&proc, "MATCH (n:Item) WHERE n.name = 'A' SET n.emb = $v", p)
        .await
        .expect("SET vector on A");

    // After SET: "right" [1,0,0,0] → A must be top-1 (cos=1.0 vs B's cos=0).
    let mut p = HashMap::new();
    p.insert("q".to_string(), pv_vector(vec![1.0, 0.0, 0.0, 0.0]));
    let after_right = cypher(
        &proc,
        "MATCH (n:Item) RETURN n.name AS name, vector_similarity(n.emb, $q) AS score \
         ORDER BY score DESC LIMIT 1",
        p,
    )
    .await
    .expect("search after SET — right query");
    let top_right = after_right
        .first()
        .and_then(|r| r.get("name"))
        .and_then(|v| match &v.value {
            Some(PvKind::StringValue(s)) => Some(s.clone()),
            _ => None,
        });
    assert_eq!(
        top_right.as_deref(),
        Some("A"),
        "G082: after SET n.emb to 'right', 'right' query must return A (cos=1.0). \
         Got: {:?}",
        top_right
    );

    // After SET: "up" [0,1,0,0] → B must win (cos≈0.8 > A_new's cos=0).
    // Proves the indexed vector was updated, not just the stored property.
    let mut p = HashMap::new();
    p.insert("q".to_string(), pv_vector(vec![0.0, 1.0, 0.0, 0.0]));
    let after_up = cypher(
        &proc,
        "MATCH (n:Item) RETURN n.name AS name, vector_similarity(n.emb, $q) AS score \
         ORDER BY score DESC LIMIT 1",
        p,
    )
    .await
    .expect("search after SET — up query");
    let top_up = after_up
        .first()
        .and_then(|r| r.get("name"))
        .and_then(|v| match &v.value {
            Some(PvKind::StringValue(s)) => Some(s.clone()),
            _ => None,
        });
    assert_ne!(
        top_up.as_deref(),
        Some("A"),
        "G082: after SET n.emb to 'right', 'up' query must NOT return A \
         (B scores cos≈0.8 vs A's cos=0). Got: {:?}",
        top_up
    );
}
