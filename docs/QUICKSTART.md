# Quick Start: From Zero to Hybrid Query in 5 Minutes

## 0. Clone the Repository

The seed script and example queries live in the repo:

```bash
git clone https://github.com/structured-world/coordinode.git
cd coordinode
```

The Docker image is published at `ghcr.io/structured-world/coordinode:latest`.

## 1. Start CoordiNode

```bash
docker compose up -d
```

Wait for it to be ready:
```bash
curl http://localhost:7084/health
```

Expected response:
```json
{"status":"ok"}
```

## 2. Seed the Knowledge Graph

The seed script creates a small knowledge graph with 4 concepts, 4 research papers (with real 384-dimensional embeddings), and relationships between them.

```bash
./examples/quickstart/seed.sh
```

Expected output:
```
=== CoordiNode Quickstart Seed ===
Target: http://localhost:7081

Waiting for CoordiNode... ready!

--- Inserting concepts ---
  Created concept: machine learning
  Created concept: deep learning
  Created concept: natural language processing
  Created concept: computer vision

--- Inserting documents ---
  Created document: Attention Is All You Need
  Created document: Deep Residual Learning for Image Recognition
  Created document: BERT: Pre-training of Deep Bidirectional Transformers
  Created document: Language Models are Few-Shot Learners

--- Creating relationships ---
  Created 4 RELATED_TO edges
  Created 4 ABOUT edges

--- Verifying ---
  {"columns":["total_nodes"],"rows":[{"values":[{"intValue":"8"}]}],"stats":{...}}

=== Seed complete! ===
```

### What was created

```
  (machine learning) ──RELATED_TO──▶ (deep learning) ──RELATED_TO──▶ (computer vision)
         │                                  │                              ▲
         │                                  │                              │
    RELATED_TO                         RELATED_TO                        ABOUT
         │                                  │                              │
         ▼                                  ▼                    [Deep Residual Learning]
  (natural language processing)   (natural language processing)
         ▲               ▲
         │               │
       ABOUT           ABOUT
         │               │
  [Attention Is      [BERT: Pre-training
   All You Need]      of Transformers]

  (deep learning)
         ▲
         │
       ABOUT
         │
  [Language Models
   are Few-Shot Learners]
```

Each document node has a 384-dimensional embedding vector representing its semantic content.

## 3. Run a Hybrid Query (The Magic Moment)

This single query combines **graph traversal** + **vector similarity**:

```bash
curl -s -X POST http://localhost:7081/v1/query/cypher \
  -H "Content-Type: application/json" \
  -d @examples/quickstart/hybrid-query.json | python3 -m json.tool
```

The query (from `hybrid-query.json`):

```cypher
MATCH (topic:Concept {name: "machine learning"})-[:RELATED_TO*1..2]->(related)
MATCH (related)<-[:ABOUT]-(doc:Document)
WHERE vector_distance(doc.embedding, $query_vec) < 0.4
RETURN doc.title, vector_distance(doc.embedding, $query_vec) AS relevance
ORDER BY relevance LIMIT 5
```

Expected response — 2 documents pass the vector filter (L2 distance < 0.4). "Deep Residual Learning" and "Language Models" are filtered out (distance > 0.4):

```json
{
  "columns": ["doc.title", "relevance"],
  "rows": [
    {
      "values": [
        {"stringValue": "Attention Is All You Need"},
        {"floatValue": 0.386}
      ]
    },
    {
      "values": [
        {"stringValue": "BERT Pre-training of Deep Bidirectional Transformers"},
        {"floatValue": 0.387}
      ]
    }
  ],
  "stats": { "executionTimeMs": "..." }
}
```

**What just happened:** CoordiNode traversed the knowledge graph (2 hops from "machine learning"), filtered documents by vector similarity (L2 distance < 0.4), sorted by relevance, and returned results — all in one atomic query.

You can also add **full-text search** to the same query with `text_match()`:

```cypher
-- Same query with an additional text filter (requires text index)
MATCH (topic:Concept {name: "machine learning"})-[:RELATED_TO*1..2]->(related)
MATCH (related)<-[:ABOUT]-(doc:Document)
WHERE vector_distance(doc.embedding, $query_vec) < 0.4
  AND text_match(doc.body, "transformer attention")
RETURN doc.title, vector_distance(doc.embedding, $query_vec) AS relevance
ORDER BY relevance LIMIT 5
```

To do this with separate databases, you would need:
1. Neo4j query to traverse the graph and get document IDs
2. Pinecone/Weaviate query to filter by vector similarity
3. Application code to intersect the result sets
4. Hope that no data changed between the queries (no transaction guarantee)

## 4. Use EXPLAIN SUGGEST

CoordiNode has a built-in query advisor that analyzes your queries and suggests optimizations:

```bash
curl -s -X POST http://localhost:7081/v1/query/cypher/explain \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (u:User) WHERE u.email = $email RETURN u", "parameters": {}}' \
  | python3 -m json.tool
```

Response (suggestions are in `plan.details.suggestions`, logical plan in `plan.details.explain`):

```json
{
  "plan": {
    "operator": "LogicalPlan",
    "children": [],
    "details": {
      "explain": "Cost: 4 | Estimated rows: 1 | Est. time: 0.0ms\n\nProject(Variable(\"u\"))\n  Filter(BinaryOp { left: PropertyAccess { expr: Variable(\"u\"), property: \"email\" }, op: Eq, right: Parameter(\"email\") })\n    NodeScan(u:User)\n",
      "cost": "4",
      "estimated_time_ms": "0.0",
      "suggestions": "[CRITICAL] CREATE INDEX: Full label scan on User.email — filtering User nodes by 'email' without an index requires scanning all User nodes\n  DDL: CREATE INDEX user_email ON User(email)",
      "suggestion_count": "1"
    },
    "estimatedRows": 1.32
  }
}
```

## 5. Explore the API

| Port | Protocol | Use |
|------|----------|-----|
| 7080 | gRPC | Native high-performance API |
| 7081 | REST | HTTP/JSON via gRPC-to-REST transcoding (structured-proxy) |
| 7084 | HTTP | Prometheus `/metrics`, `/health`, `/ready` |

## Next Steps

- [Cypher Extensions](/cypher/extensions) — vector, full-text, spatial, time-travel, encrypted search syntax
- [Compatibility](COMPATIBILITY.md) — what works from the Neo4j ecosystem

## Known Limitations (alpha)

- **Single-node only** — Raft clustering in development (v0.4). Single-node handles 100K-1M nodes.
- **No Bolt protocol** — use gRPC, REST, or GraphQL. Neo4j driver support planned for v1.2.
- **HNSW index is in-memory** — vector indexes reside in RAM. At 1M vectors x 384 dims = ~1.5GB.
- **No APOC/GDS** — standard OpenCypher works; Neo4j procedure libraries are not supported.
- **text_match() requires text index** — create via programmatic API; DDL syntax coming soon.
- **Exact response values may vary** — floating-point distances depend on platform. Values shown are approximate.
