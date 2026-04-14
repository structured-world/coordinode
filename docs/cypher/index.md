---
description: "CoordiNode OpenCypher reference — supported clauses, functions, operators, and extensions."
---

# OpenCypher Reference

CoordiNode implements a subset of OpenCypher with its own extensions for vector search, full-text search, spatial queries, document operations, and time-travel reads.

::: info Implementation status convention
Throughout this reference, every feature is marked with its current status:

| Badge | Meaning |
|-------|---------|
| ✅ **Supported** | Works in this release |
| 🔷 **Extension** | CoordiNode-specific, not in standard OpenCypher |
| 📋 **Planned** | Parsed / in the roadmap, not yet functional |
:::

::: warning Functions returning null
Several standard OpenCypher functions (`toInteger`, `toLower`, `length`, `abs`, etc.) parse without error but return `null` in this release. Using them silently produces wrong results — check the [Functions reference](./functions) for the current list.
:::

## Navigation

- **[Language Reference](./reference)** — Clauses, operators, data types, patterns
- **[Functions](./functions)** — All scalar, aggregation, and extension functions with status
- **[Extensions](./extensions)** — Vector search, full-text, spatial, document ops, time-travel
- **[Neo4j Compatibility](./compatibility)** — What works with Neo4j tooling and what doesn't

## Quick Example

```cypher
-- Graph traversal + vector similarity in one query
MATCH (topic:Concept {name: $topic})-[:RELATED_TO*1..2]->(related)
MATCH (related)<-[:ABOUT]-(doc:Document)
WHERE vector_similarity(doc.embedding, $query_vec) > 0.7
  AND text_match(doc.body, $keywords)
RETURN doc.title,
       vector_similarity(doc.embedding, $query_vec) AS semantic,
       text_score(doc.body, $keywords) AS relevance
ORDER BY semantic DESC
LIMIT 10
```

## What's in This Release

| Area | Status |
|------|--------|
| Read clauses (MATCH, RETURN, WITH, UNWIND, ORDER BY, SKIP, LIMIT) | ✅ |
| Write clauses (CREATE, MERGE, DELETE, DETACH DELETE, SET, REMOVE) | ✅ |
| UPSERT MATCH (atomic upsert) | 🔷 |
| Variable-length paths `*min..max` | ✅ |
| Pattern predicates in WHERE | ✅ |
| CASE WHEN expressions | ✅ |
| Map projections `n { .prop, alias: expr }` | ✅ |
| Aggregations (count, sum, avg, min, max, collect, stDev, percentile) | ✅ |
| Index DDL (CREATE/DROP INDEX, VECTOR INDEX, TEXT INDEX, ENCRYPTED INDEX) | ✅ |
| Vector functions (vector_distance, vector_similarity, vector_dot) | ✅ 🔷 |
| Full-text functions (text_match, text_score) | ✅ 🔷 |
| Spatial functions (point, point.distance) | ✅ 🔷 |
| Document operations (doc_push, doc_pull, doc_add_to_set, doc_inc) | ✅ 🔷 |
| AS OF TIMESTAMP time-travel reads | ✅ 🔷 |
| EXPLAIN / EXPLAIN SUGGEST | ✅ 🔷 |
| CALL procedures (db.advisor.suggestions) | ✅ partial |
| Bolt protocol (Neo4j wire) | 📋 v1.2 |
| LOAD CSV | 📋 v1.2 |
| FOREACH | 📋 v1.0 |
| Scalar functions (toInteger, toLower, length, abs, …) | 📋 |
