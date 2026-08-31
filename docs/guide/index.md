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

- 100% drop-in Neo4j Enterprise replacement (gRPC and REST are available now; the Bolt protocol is planned)
- APOC procedures, Neo4j Browser/Bloom, or GDS
- A cluster with years of production mileage. Raft replication, follower reads and mutual TLS ship today under an automated fault-injection suite; the operational history behind them is still short
- Datasets larger than one machine holds. The open-source edition replicates the full dataset to every node; horizontal sharding is an Enterprise feature

## Next Steps

- **[Quick Start](/QUICKSTART)** — Docker → seed data → first hybrid query in 5 minutes
- **[OpenCypher Extensions](/cypher/extensions)** — vector_distance(), text_match(), point.distance(), EXPLAIN SUGGEST
- **[Python SDK](/sdk/python)** — `pip install coordinode`
- **[LlamaIndex integration](/sdk/llama-index)** — PropertyGraphIndex with CoordiNode backend
- **[LangChain integration](/sdk/langchain)** — GraphCypherQAChain
