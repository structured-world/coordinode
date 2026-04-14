# Hybrid Retrieval

CoordiNode's defining capability: graph traversal, vector similarity, and full-text search in a **single atomic query**. No application-side join. No cross-database round-trip.

## The Problem with Separate Databases

A typical AI retrieval stack looks like:

1. Graph DB query → get related node IDs
2. Vector DB query (Pinecone / Weaviate) → filter by embedding similarity
3. Application code → intersect result sets
4. Hope that nothing changed between steps (no transactional guarantee)

This is slow, error-prone, and expensive. Any data written between steps 1 and 2 may create inconsistencies.

## One Query, Three Modalities

```cypher
MATCH (topic:Concept {name: "machine learning"})-[:RELATED_TO*1..2]->(related)
MATCH (related)<-[:ABOUT]-(doc:Document)
WHERE vector_distance(doc.embedding, $query_vec) < 0.4
  AND text_match(doc.body, "transformer attention")
RETURN doc.title, vector_distance(doc.embedding, $query_vec) AS relevance
ORDER BY relevance LIMIT 5
```

This query:

1. **Graph traversal** — follows `RELATED_TO` edges 1-2 hops from "machine learning"
2. **Vector filter** — keeps only documents whose embedding is within L2 distance 0.4 of `$query_vec`
3. **Full-text filter** — keeps only documents whose body contains "transformer" or "attention"
4. **Ranks and returns** — sorted by vector relevance, top 5

All three filters apply within a single MVCC snapshot — consistent by construction.

## Extension Functions

| Function | Description | Example |
|----------|-------------|---------|
| `vector_distance(prop, $vec)` | L2 (Euclidean) distance between a stored vector and a query vector | `WHERE vector_distance(n.emb, $q) < 0.5` |
| `vector_cosine(prop, $vec)` | Cosine similarity (higher = more similar) | `ORDER BY vector_cosine(n.emb, $q) DESC` |
| `text_match(prop, "query")` | Full-text BM25 match | `WHERE text_match(n.body, "attention transformer")` |
| `text_score(prop, "query")` | BM25 relevance score (0.0–1.0) | `RETURN text_score(n.body, "attention") AS score` |
| `geo_distance(prop, $point)` | Haversine distance in metres | `WHERE geo_distance(n.location, $pt) < 1000` |
| `geo_within(prop, $polygon)` | Point-in-polygon test | `WHERE geo_within(n.location, $bbox)` |

See [Cypher Extensions](../CYPHER_EXTENSIONS) for the full reference.

## Execution Plan

CoordiNode's query planner decides the most selective filter to apply first:

- If a vector index exists → apply vector filter early (ANN pre-filter)
- If a text index exists → apply BM25 filter in tandem
- Graph traversal limits the candidate set before applying modality filters

Use `EXPLAIN` to inspect the chosen plan:

```bash
curl -X POST http://localhost:7081/v1/query/cypher/explain \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH ...", "parameters": {}}'
```

## Practical Pattern: GraphRAG

A common retrieval pattern for AI applications:

```cypher
-- Find documents semantically related to a query,
-- then expand the graph context around each document.
MATCH (doc:Document)
WHERE vector_distance(doc.embedding, $query_vec) < 0.5
WITH doc ORDER BY vector_distance(doc.embedding, $query_vec) LIMIT 20

-- Expand graph neighbourhood for richer context
MATCH (doc)-[:CITES|AUTHORED_BY|TAGGED_WITH*1..2]->(context)
RETURN doc.title, doc.body, collect(context) AS graph_context
```

The retrieved documents plus their graph neighbourhood form the context window for an LLM prompt — richer than pure vector retrieval, cheaper than fetching entire subgraphs blindly.

## Next Step

- [Data Model](./data-model) — how nodes carry vector and text properties
- [MVCC Transactions](./transactions) — consistency guarantees
- [Cypher Extensions](../CYPHER_EXTENSIONS) — full syntax reference
