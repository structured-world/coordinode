---
layout: home
titleTemplate: false
description: "Graph + Vector + Full-Text retrieval in a single MVCC transaction. OpenCypher-compatible. Built in Rust."
head:
  - - meta
    - name: keywords
      content: graph database, vector database, full-text search, OpenCypher, GraphRAG, Rust, embedded database, hybrid retrieval

hero:
  name: CoordiNode
  text: Graph + Vector + Text in one engine
  tagline: Graph traversal, vector similarity, and full-text retrieval in a single MVCC transaction. OpenCypher-compatible. Built in Rust.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/
    - theme: alt
      text: Quick Start
      link: /QUICKSTART
    - theme: alt
      text: GitHub
      link: https://github.com/structured-world/coordinode

features:
  - icon: "🔗"
    title: Graph + Vector + Text — one query
    details: Traverse a knowledge graph, filter by semantic similarity, rank by full-text relevance. One query, one transaction. No glue code.
  - icon: "⚡"
    title: Zero GC. Predictable P99.
    details: Built in Rust with an LSM-tree storage engine. No JVM pauses, no garbage collector. Consistent tail latency under load.
  - icon: "🔐"
    title: MVCC with Snapshot Isolation
    details: Every read gets a consistent snapshot. Optimistic concurrency control. Time-travel queries with 7-day retention.
  - icon: "📐"
    title: OpenCypher-compatible
    details: Standard Cypher syntax extended with vector_distance(), text_match(), and point.distance(). EXPLAIN SUGGEST built in.
  - icon: "🔍"
    title: Vectors on edges, not just nodes
    details: Vector similarity search on edge properties. First-class support for GraphRAG, recommendation engines, and fraud detection.
  - icon: "🏠"
    title: Single binary. Embedded mode.
    details: One binary, three ports (gRPC, REST, metrics). Add coordinode-embed to your Rust project — no separate process.
---

## The Magic Moment

```cypher
MATCH (topic:Concept {name: "machine learning"})-[:RELATED_TO*1..3]->(related)
MATCH (related)<-[:ABOUT]-(doc:Document)
WHERE vector_distance(doc.embedding, $question_vector) < 0.4
  AND text_match(doc.body, "transformer attention mechanism")
RETURN doc.title,
       vector_distance(doc.embedding, $question_vector) AS relevance,
       text_score(doc.body, "transformer attention mechanism")  AS text_rank
ORDER BY relevance LIMIT 10
```

Today this query requires Neo4j + Pinecone + Elasticsearch + custom glue code. **With CoordiNode — one query.**

## Quick Install

::: code-group

```bash [Docker]
docker run -p 7080:7080 -p 7081:7081 -p 7084:7084 \
  ghcr.io/structured-world/coordinode:latest
```

```toml [Embedded (Cargo.toml)]
[dependencies]
coordinode-embed = "0.3"
```

:::

See the [Quick Start guide](/QUICKSTART) for a complete walkthrough with seed data and example queries.
