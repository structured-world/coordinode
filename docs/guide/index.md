---
title: Introduction
description: CoordiNode is a graph-native hybrid retrieval engine combining graph traversal, vector similarity, and full-text search in a single MVCC transaction.
---

# Introduction

CoordiNode is a **graph-native hybrid retrieval engine** for AI and GraphRAG workloads.

It combines three retrieval modalities in one engine:
- **Graph traversal** — variable-length paths, pattern matching, aggregation
- **Vector similarity** — HNSW index, SQ8 quantization, cosine/L2/dot metrics
- **Full-text search** — BM25 scoring, 23+ languages, fuzzy/phrase/wildcard queries

One query language (OpenCypher-compatible), one transaction model (MVCC, Snapshot Isolation).

## Is CoordiNode Right for You?

### Use this today if you are building:

- **GraphRAG** — knowledge retrieval, relationship-aware AI
- **Fraud detection** — ring detection through shared-device graphs + behavioral embedding similarity
- **Semantic recommendations** — traverse social graphs, filter by semantic similarity
- **Threat intelligence** — correlate attack patterns with MITRE ATT&CK + vector + text search

### Not yet ready for:

- 100% drop-in Neo4j Enterprise replacement (gRPC and REST are available now; Bolt protocol planned for v1.2)
- APOC procedures, Neo4j Browser/Bloom, or GDS
- Production multi-node clustering (single-node is stable; Raft clustering in active development for v0.4)

## Next Steps

- **[Quick Start](/QUICKSTART)** — Docker → seed data → first hybrid query in 5 minutes
- **[OpenCypher Extensions](/CYPHER_EXTENSIONS)** — vector_distance(), text_match(), point.distance(), EXPLAIN SUGGEST
- **[Python SDK](/sdk/python)** — `pip install coordinode`
- **[LlamaIndex integration](/sdk/llama-index)** — PropertyGraphIndex with CoordiNode backend
- **[LangChain integration](/sdk/langchain)** — GraphCypherQAChain
