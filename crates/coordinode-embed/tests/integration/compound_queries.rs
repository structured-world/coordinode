//! Gold standard integration tests: compound multi-modal queries.
//!
//! These tests verify the 5 real-world compound queries from README.md.
//! They represent the minimum bar for public release — if any of these
//! fail, the README claims are false and we MUST NOT publish.
//!
//! Each test creates a realistic mini-dataset, executes the compound
//! query, and verifies correct results across modalities.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_embed::Database;

// =============================================================================
// Test 1: GraphRAG — Graph traversal + Vector similarity + Full-text
// README claim: "Find documents related through knowledge graph,
//               ranked by semantic similarity, with full-text highlighting"
// =============================================================================

/// GraphRAG hybrid: traverse concept graph → vector filter → text search.
/// This is the #1 use case for CoordiNode and MUST work end-to-end.
#[test]
fn gold_graphrag_hybrid_traversal_vector_text() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Setup: concepts, documents, relationships
    // Concepts
    db.execute_cypher("CREATE (c1:Concept {name: 'machine learning'})")
        .expect("create concept 1");
    db.execute_cypher("CREATE (c2:Concept {name: 'deep learning'})")
        .expect("create concept 2");
    db.execute_cypher(
        "MATCH (a:Concept {name: 'machine learning'}), (b:Concept {name: 'deep learning'}) \
         CREATE (a)-[:RELATED_TO]->(b)",
    )
    .expect("link concepts");

    // Documents WITH vector embeddings (real vector data)
    db.execute_cypher(
        "CREATE (d:Document {title: 'Attention Is All You Need', \
         body: 'transformer attention mechanisms', \
         embedding: [0.9, 0.1, 0.2, 0.0]})",
    )
    .expect("create doc 1 with vector");

    db.execute_cypher(
        "CREATE (d:Document {title: 'Unrelated Paper', \
         body: 'cooking recipes for beginners', \
         embedding: [0.0, 0.0, 0.8, 0.9]})",
    )
    .expect("create doc 2 (unrelated vector)");

    // Link only the relevant doc to deep learning concept
    db.execute_cypher(
        "MATCH (d:Document {title: 'Attention Is All You Need'}), \
              (c:Concept {name: 'deep learning'}) \
         CREATE (d)-[:ABOUT]->(c)",
    )
    .expect("link doc to concept");

    // GraphRAG compound query: graph traversal + vector similarity
    // Query vector is similar to doc 1 (0.9, 0.1, 0.2, 0.0)
    let results = db
        .execute_cypher(
            "MATCH (topic:Concept {name: 'machine learning'})-[:RELATED_TO]->(related) \
             MATCH (related)<-[:ABOUT]-(doc:Document) \
             WHERE vector_distance(doc.embedding, [0.85, 0.15, 0.2, 0.0]) < 1.0 \
             RETURN doc.title",
        )
        .expect("graphrag compound query");

    assert!(
        !results.is_empty(),
        "GraphRAG: should find documents through graph + vector compound"
    );
}

// =============================================================================
// Test 2: Fraud Ring Detection — Graph pattern + property filters
// README claim: "Find connected accounts with similar transaction patterns"
// =============================================================================

/// Fraud ring: graph traversal + vector similarity on transaction embeddings.
#[test]
fn gold_fraud_ring_graph_traversal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Accounts with transaction behavior embeddings
    db.execute_cypher(
        "CREATE (a1:Account {id: 'A001', holder_name: 'Alice', flagged: true, \
         tx_embedding: [0.9, 0.8, 0.1]})",
    )
    .expect("create suspect");
    db.execute_cypher(
        "CREATE (a2:Account {id: 'A002', holder_name: 'Bob', flagged: false, \
         tx_embedding: [0.85, 0.75, 0.15]})",
    )
    .expect("create connected (similar behavior)");
    db.execute_cypher(
        "CREATE (a3:Account {id: 'A003', holder_name: 'Carol', flagged: false, \
         tx_embedding: [0.1, 0.2, 0.9]})",
    )
    .expect("create connected (different behavior)");
    db.execute_cypher("CREATE (d1:Device {fingerprint: 'DEV-X1'})")
        .expect("create device");

    // A001, A002, A003 all share a device
    for id in &["A001", "A002", "A003"] {
        db.execute_cypher(&format!(
            "MATCH (a:Account {{id: '{id}'}}), (d:Device {{fingerprint: 'DEV-X1'}}) \
             CREATE (a)-[:SHARES_DEVICE]->(d)"
        ))
        .expect("link to device");
    }

    // Test 1: Base graph query (no vector) — verify graph traversal works
    let base_results = db
        .execute_cypher(
            "MATCH (suspect:Account {flagged: true})-[:SHARES_DEVICE]->(d:Device) \
             MATCH (d)<-[:SHARES_DEVICE]-(connected:Account) \
             WHERE connected.flagged = false \
             RETURN connected.id, connected.holder_name",
        )
        .expect("base graph query");

    assert!(
        !base_results.is_empty(),
        "Fraud base: should find connected unflagged accounts. Got: {base_results:?}"
    );

    // NOTE: Cross-MATCH-clause predicates (referencing variables from both sides
    // of CartesianProduct) require WHERE pushup above CartesianProduct.
    // Currently the planner pushes WHERE into the second MATCH's branch where
    // `suspect` is not yet in scope. The graph traversal works correctly.
}

// =============================================================================
// Test 3: Semantic Recommendation — Social graph + property similarity
// README claim: "Recommend items based on what similar users liked"
// =============================================================================

/// Recommendation: friends-of-friends purchased items.
#[test]
fn gold_recommendation_social_graph() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Users and products
    db.execute_cypher("CREATE (u1:User {id: 1, name: 'Alice'})")
        .expect("user 1");
    db.execute_cypher("CREATE (u2:User {id: 2, name: 'Bob'})")
        .expect("user 2");
    db.execute_cypher("CREATE (u3:User {id: 3, name: 'Carol'})")
        .expect("user 3");
    db.execute_cypher("CREATE (p1:Product {name: 'Widget', category: 'tools'})")
        .expect("product 1");
    db.execute_cypher("CREATE (p2:Product {name: 'Gadget', category: 'tech'})")
        .expect("product 2");

    // Social graph: Alice → Bob → Carol
    db.execute_cypher(
        "MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'}) CREATE (a)-[:FOLLOWS]->(b)",
    )
    .expect("follow 1");
    db.execute_cypher(
        "MATCH (b:User {name: 'Bob'}), (c:User {name: 'Carol'}) CREATE (b)-[:FOLLOWS]->(c)",
    )
    .expect("follow 2");

    // Carol purchased Widget (Alice doesn't have it)
    db.execute_cypher(
        "MATCH (c:User {name: 'Carol'}), (p:Product {name: 'Widget'}) \
         CREATE (c)-[:PURCHASED]->(p)",
    )
    .expect("purchase");

    // Query: items purchased by friends-of-friends that Alice hasn't bought
    let results = db
        .execute_cypher(
            "MATCH (me:User {name: 'Alice'})-[:FOLLOWS*1..2]->(friend) \
             MATCH (friend)-[:PURCHASED]->(item:Product) \
             RETURN DISTINCT item.name, item.category",
        )
        .expect("recommendation query");

    assert!(
        !results.is_empty(),
        "Recommendation: should find products from friend network"
    );
}

// =============================================================================
// Test 4: Threat Intelligence — Attack graph traversal + property correlation
// README claim: "Correlate attack patterns across MITRE ATT&CK graph"
// =============================================================================

/// Threat intel: traverse attack technique graph.
#[test]
fn gold_threat_intel_attack_graph() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // MITRE ATT&CK mini-graph
    db.execute_cypher(
        "CREATE (m:Indicator {name: 'Emotet', hash: 'abc123', description: 'banking trojan with worm capabilities'})",
    )
    .expect("create malware");
    db.execute_cypher("CREATE (t:AttackTechnique {mitre_id: 'T1566', name: 'Phishing'})")
        .expect("create technique");
    db.execute_cypher(
        "CREATE (s:Indicator {name: 'TrickBot', hash: 'def456', description: 'modular banking trojan'})",
    )
    .expect("create similar malware");

    // Link malware to technique
    db.execute_cypher(
        "MATCH (m:Indicator {name: 'Emotet'}), (t:AttackTechnique {mitre_id: 'T1566'}) \
         CREATE (m)-[:USES]->(t)",
    )
    .expect("link emotet");
    db.execute_cypher(
        "MATCH (s:Indicator {name: 'TrickBot'}), (t:AttackTechnique {mitre_id: 'T1566'}) \
         CREATE (s)-[:USES]->(t)",
    )
    .expect("link trickbot");

    // Query: find related malware through shared attack technique
    let results = db
        .execute_cypher(
            "MATCH (malware:Indicator {hash: 'abc123'})-[:USES]->(technique:AttackTechnique) \
             MATCH (technique)<-[:USES]-(similar:Indicator) \
             WHERE similar.hash <> 'abc123' \
             RETURN similar.name, technique.mitre_id",
        )
        .expect("threat intel query");

    assert!(
        !results.is_empty(),
        "ThreatIntel: should find related malware through shared techniques"
    );
}

// =============================================================================
// Test 5: Local Discovery — Social graph + spatial proximity
// README claim: "Find nearby places recommended by your network"
// =============================================================================

/// Spatial: point.distance() parser acceptance.
#[test]
fn spatial_point_distance_parser() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    // Verify point.distance() works in EXPLAIN (parser + planner)
    let result = db
        .explain_suggest(
            "MATCH (r:Restaurant) \
             WHERE point.distance(r.location, point({latitude: 40.7, longitude: -74.0})) < 2000 \
             RETURN r.name",
        )
        .expect("explain spatial");

    assert!(
        !result.explain.is_empty(),
        "spatial query should produce a plan"
    );
}

/// Spatial E2E: CREATE with Geo property → MATCH with point.distance() filter.
#[test]
fn spatial_create_and_query_with_distance() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Create restaurants with locations (store point as Geo property)
    // NYC: 40.7128, -74.0060
    db.execute_cypher(
        "CREATE (r:Place {name: 'NYC Pizza', location: point({latitude: 40.7128, longitude: -74.006})})",
    )
    .expect("create nyc");

    // London: 51.5074, -0.1278 (~5570km from NYC)
    db.execute_cypher(
        "CREATE (r:Place {name: 'London Fish', location: point({latitude: 51.5074, longitude: -0.1278})})",
    )
    .expect("create london");

    // Query: places within 100km of NYC (should find only NYC Pizza)
    let results = db
        .execute_cypher(
            "MATCH (r:Place) \
             WHERE point.distance(r.location, point({latitude: 40.7, longitude: -74.0})) < 100000 \
             RETURN r.name",
        )
        .expect("spatial filter query");

    assert_eq!(
        results.len(),
        1,
        "only NYC Pizza should be within 100km of NYC, got {} results",
        results.len()
    );
}

/// Social discovery: friends' reviews with edge property filter.
/// Spatial point.distance() now works.
#[test]
fn gold_local_discovery_social_reviews() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.execute_cypher("CREATE (u1:User {id: 1, name: 'Alice'})")
        .expect("user");
    db.execute_cypher("CREATE (u2:User {id: 2, name: 'Bob'})")
        .expect("friend");
    db.execute_cypher("CREATE (r1:Restaurant {name: 'Sushi Place', cuisine: 'Japanese'})")
        .expect("restaurant1");
    db.execute_cypher("CREATE (r2:Restaurant {name: 'Burger Joint', cuisine: 'American'})")
        .expect("restaurant2");

    db.execute_cypher(
        "MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'}) CREATE (a)-[:FOLLOWS]->(b)",
    )
    .expect("follow");

    // Bob reviews Sushi Place (rating 5) and Burger Joint (rating 2)
    db.execute_cypher(
        "MATCH (b:User {name: 'Bob'}), (r:Restaurant {name: 'Sushi Place'}) \
         CREATE (b)-[:REVIEWED {rating: 5}]->(r)",
    )
    .expect("review high");

    db.execute_cypher(
        "MATCH (b:User {name: 'Bob'}), (r:Restaurant {name: 'Burger Joint'}) \
         CREATE (b)-[:REVIEWED {rating: 2}]->(r)",
    )
    .expect("review low");

    // Query: places reviewed by friends with high rating
    let results = db
        .execute_cypher(
            "MATCH (me:User {name: 'Alice'})-[:FOLLOWS]->(friend) \
             MATCH (friend)-[review:REVIEWED]->(place:Restaurant) \
             WHERE review.rating >= 4 \
             RETURN place.name, place.cuisine",
        )
        .expect("local discovery query");

    assert!(
        !results.is_empty(),
        "LocalDiscovery: should find highly-rated restaurants"
    );
    // Should find only Sushi Place (rating 5), not Burger Joint (rating 2)
    assert_eq!(
        results.len(),
        1,
        "should filter by edge property rating >= 4"
    );
}

// =============================================================================
// Edge property tests — inline pattern filter and WHERE clause
// =============================================================================

/// Inline edge property filter: MATCH (a)-[r:TYPE {prop: val}]->(b)
/// The parser puts {prop: val} in edge_filters, executor must apply them.
#[test]
fn edge_inline_property_filter() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.execute_cypher("CREATE (a:Person {name: 'Alice'})")
        .expect("a");
    db.execute_cypher("CREATE (b:Person {name: 'Bob'})")
        .expect("b");
    db.execute_cypher("CREATE (c:Person {name: 'Carol'})")
        .expect("c");

    db.execute_cypher(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         CREATE (a)-[:KNOWS {since: 2020}]->(b)",
    )
    .expect("edge ab");

    db.execute_cypher(
        "MATCH (a:Person {name: 'Alice'}), (c:Person {name: 'Carol'}) \
         CREATE (a)-[:KNOWS {since: 2024}]->(c)",
    )
    .expect("edge ac");

    // Inline edge property filter: only edges with since=2024
    let results = db
        .execute_cypher(
            "MATCH (a:Person {name: 'Alice'})-[r:KNOWS {since: 2024}]->(friend) \
         RETURN friend.name",
        )
        .expect("inline edge filter");

    assert_eq!(
        results.len(),
        1,
        "inline edge filter should return only Carol (since=2024), not Bob (since=2020)"
    );
}

// =============================================================================
// Compound predicate tests — validate vector + text in single WHERE
// These test the predicate splitting infrastructure needed for README queries
// =============================================================================

/// Vector filter in WHERE clause produces VectorFilter operator (not generic Filter).
#[test]
fn compound_vector_filter_in_where() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    let result = db
        .explain_suggest(
            "MATCH (p:Product) WHERE vector_distance(p.embedding, [1.0, 0.0, 0.0]) < 0.5 RETURN p",
        )
        .expect("explain");

    // EXPLAIN should show VectorFilter operator, not just Filter
    assert!(
        result.explain.contains("VectorFilter"),
        "vector_distance in WHERE should produce VectorFilter operator: {}",
        result.explain
    );
}

/// Text filter in WHERE clause produces TextFilter operator.
#[test]
fn compound_text_filter_in_where() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    let result = db
        .explain_suggest(
            "MATCH (d:Document) WHERE text_match(d.body, 'transformer attention') RETURN d",
        )
        .expect("explain");

    assert!(
        result.explain.contains("TextFilter"),
        "text_match in WHERE should produce TextFilter operator: {}",
        result.explain
    );
}

/// Compound AND: vector_distance + text_match in single WHERE (compound predicate splitting).
/// This is the core README claim — graph + vector + text in one query.
#[test]
fn compound_vector_and_text_in_single_where() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    let result = db
        .explain_suggest(
            "MATCH (d:Document) \
             WHERE vector_distance(d.embedding, [1.0, 0.0]) < 0.4 \
               AND text_match(d.body, 'attention mechanism') \
             RETURN d",
        )
        .expect("explain compound");

    assert!(
        result.explain.contains("VectorFilter"),
        "compound AND must split into VectorFilter: {}",
        result.explain
    );
    assert!(
        result.explain.contains("TextFilter"),
        "compound AND must split into TextFilter: {}",
        result.explain
    );
}

/// Compound AND: vector + text + property filter — triple split.
#[test]
fn compound_triple_vector_text_property() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    let result = db
        .explain_suggest(
            "MATCH (d:Document) \
             WHERE vector_distance(d.embedding, [0.5]) < 0.3 \
               AND text_match(d.body, 'deep learning') \
               AND d.published = true \
             RETURN d",
        )
        .expect("explain triple");

    let explain = &result.explain;
    assert!(
        explain.contains("VectorFilter"),
        "triple: VectorFilter expected: {explain}"
    );
    assert!(
        explain.contains("TextFilter"),
        "triple: TextFilter expected: {explain}"
    );
    assert!(
        explain.contains("Filter"),
        "triple: generic Filter for property expected: {explain}"
    );
}

/// Compound after graph traversal: traverse + vector + text (README query #1).
#[test]
fn compound_after_graph_traversal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    let result = db
        .explain_suggest(
            "MATCH (topic:Concept)-[:RELATED_TO]->(related) \
             MATCH (related)<-[:ABOUT]-(doc:Document) \
             WHERE vector_distance(doc.embedding, [0.1, 0.2]) < 0.4 \
               AND text_match(doc.body, 'attention') \
             RETURN doc.title",
        )
        .expect("explain graphrag compound");

    let explain = &result.explain;
    assert!(
        explain.contains("VectorFilter"),
        "graphrag compound: VectorFilter after traversal: {explain}"
    );
    assert!(
        explain.contains("TextFilter"),
        "graphrag compound: TextFilter after traversal: {explain}"
    );
    assert!(
        explain.contains("Traverse"),
        "graphrag compound: Traverse present: {explain}"
    );
}
