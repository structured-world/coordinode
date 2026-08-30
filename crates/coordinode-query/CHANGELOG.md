# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

#### Added

- *(query)* `rrf_score([method_exprs…], {vector: …, text: …})` - Reciprocal Rank Fusion Cypher function. N-method rank fusion with competition ranks, `k=60` (IR standard, non-tunable), per-method direction from HNSW metric config. Supports node vectors, edge vectors (brute-force), and BM25 text methods.
