use super::*;

use crate::proto::graph::graph_service_server::GraphService;
use crate::proto::query::text_service_server::TextService;
use crate::proto::query::vector_service_server::VectorService;
use crate::proto::{graph, query};

/// `message` as a call naming `namespace`.
fn in_ns<T>(namespace: &'static str, message: T) -> tonic::Request<T> {
    let mut request = tonic::Request::new(message);
    request.metadata_mut().insert(
        NAMESPACE_HEADER,
        tonic::metadata::MetadataValue::from_static(namespace),
    );
    request
}

/// Two namespaces holding the same label, property and words, each with
/// its own text index: a vector search, a text search and a traversal in
/// one find that namespace's nodes alone, and a traversal named in the other
/// namespace from a node of the first finds nothing.
#[tokio::test(flavor = "multi_thread")]
async fn typed_reads_see_their_own_namespace_alone() {
    use coordinode_core::namespace::DEFAULT_NAMESPACE as OWN;
    let dir = tempfile::tempdir().expect("tempdir");
    let database = coordinode_embed::Database::open(dir.path()).expect("open");
    database.create_namespace("other").expect("create other");
    let databases = Databases::of(std::sync::Arc::new(parking_lot::RwLock::new(database)));
    let graph_svc = crate::services::graph::GraphServiceImpl::serving(databases.clone());
    let vector_svc = crate::services::vector::VectorServiceImpl::serving(databases.clone());
    let text_svc = crate::services::text::TextServiceImpl::serving(databases.clone());

    let mut starts = Vec::new();
    for (namespace, word, v) in [(OWN, "alpha", 1.0_f32), ("other", "beta", 2.0)] {
        let db = databases.served.in_namespace(namespace).expect("session");
        {
            let mut db = db.write();
            db.execute_cypher("CREATE TEXT INDEX doc_body ON :Doc(body)")
                .expect("text index");
            db.execute_cypher(&format!(
                "CREATE (s:Doc {{body: 'shared {word}', emb: [{v}, 0.0]}})-[:NEXT]->\
                 (:Doc {{body: 'tail {word}', emb: [{v}, 1.0]}})"
            ))
            .expect("write");
        }
        let rows = db
            .write()
            .execute_cypher(&format!(
                "MATCH (s:Doc) WHERE s.body = 'shared {word}' RETURN s AS s"
            ))
            .expect("start");
        let Some(coordinode_core::graph::types::Value::Int(id)) = rows[0].get("s") else {
            panic!("no start node in {namespace}");
        };
        starts.push(*id as u64);
    }

    for (namespace, word, other_word, v) in [
        (OWN, "alpha", "beta", 1.0_f32),
        ("other", "beta", "alpha", 2.0),
    ] {
        let found = vector_svc
            .vector_search(in_ns(
                namespace,
                query::VectorSearchRequest {
                    label: "Doc".into(),
                    property: "emb".into(),
                    query_vector: Some(crate::proto::common::Vector {
                        values: vec![v, 0.0],
                    }),
                    top_k: 10,
                    metric: 0,
                },
            ))
            .await
            .expect("vector search")
            .into_inner()
            .results;
        assert_eq!(found.len(), 2, "{namespace} ranks its own two nodes");

        // The text worker indexes in the background.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let hits = |term: &str| {
                text_svc.text_search(in_ns(
                    namespace,
                    query::TextSearchRequest {
                        label: "Doc".into(),
                        query: term.into(),
                        limit: 10,
                        fuzzy: false,
                        language: String::new(),
                    },
                ))
            };
            let own = hits(word).await.expect("text search").into_inner().results;
            if own.len() == 2 {
                let foreign = hits(other_word)
                    .await
                    .expect("text search")
                    .into_inner()
                    .results;
                assert!(foreign.is_empty(), "{namespace} finds another's words");
                break;
            }
            assert!(std::time::Instant::now() < deadline, "never indexed");
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }

    let traverse = |namespace: &'static str, start: u64| {
        graph_svc.traverse(in_ns(
            namespace,
            graph::TraverseRequest {
                start_node_id: start,
                edge_type: "NEXT".into(),
                max_depth: 1,
                direction: 0,
                pagination: None,
            },
        ))
    };
    assert_eq!(
        traverse(OWN, starts[0])
            .await
            .expect("traverse")
            .into_inner()
            .nodes
            .len(),
        1
    );
    assert!(
        traverse("other", starts[0])
            .await
            .expect("traverse")
            .into_inner()
            .nodes
            .is_empty(),
        "a traversal reached a node of another namespace"
    );
}

/// A call names its namespace in one metadata entry; one naming none selects
/// none; an empty or unreadable name is refused rather than taken as none.
#[test]
fn a_call_names_its_namespace_in_its_metadata() {
    let mut metadata = MetadataMap::new();
    assert_eq!(from_metadata(&metadata).expect("none"), None);
    metadata.insert(NAMESPACE_HEADER, "tenant-a".parse().expect("value"));
    assert_eq!(
        from_metadata(&metadata).expect("named"),
        Some("tenant-a".to_owned())
    );
    metadata.insert(NAMESPACE_HEADER, "".parse().expect("value"));
    assert_eq!(
        from_metadata(&metadata).expect_err("empty").code(),
        tonic::Code::InvalidArgument
    );
}
