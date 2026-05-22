# Copyright 2026 Structured World Foundation.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""ann-benchmarks adapter for CoordiNode.

Thin wrapper around ``coordinode_embedded.Hnsw`` — the native PyO3 binding
that drives ``coordinode_vector::hnsw::HnswIndex`` directly, bypassing Cypher.
This is the apples-to-apples comparison point against hnswlib / FAISS-HNSW /
ScaNN / Annoy on the ann-benchmarks leaderboard.

For the server-class (network + concurrency) comparison against Qdrant /
Milvus / Weaviate, see the VDBBench adapter (separate work, ``coordinode``
service mode via gRPC).
"""

from __future__ import annotations

import numpy as np

from ann_benchmarks.algorithms.base.module import BaseANN
from coordinode_embedded import Hnsw


_METRIC_MAP = {
    "angular": "cosine",
    "cosine": "cosine",
    "euclidean": "euclidean",
    "l2": "euclidean",
    "dot": "dot",
    "inner_product": "dot",
    "manhattan": "manhattan",
    "l1": "manhattan",
}


class CoordiNode(BaseANN):
    """Native HNSW from CoordiNode (in-process, no server)."""

    def __init__(self, metric: str, method_param: dict) -> None:
        if metric not in _METRIC_MAP:
            raise ValueError(
                f"unsupported metric {metric!r}; expected one of {sorted(_METRIC_MAP)}"
            )
        self._metric = _METRIC_MAP[metric]
        self._M = int(method_param["M"])
        self._ef_construction = int(method_param["efConstruction"])
        self._ef_query: int | None = None
        self._index: Hnsw | None = None
        self.name = f"CoordiNode(M={self._M}, efConstruction={self._ef_construction})"

    def fit(self, X) -> None:
        # ann-benchmarks hands us a numpy ndarray (sometimes float64).  The
        # PyO3 binding requires float32 — convert if needed (cheap; one copy).
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        dim = int(X.shape[1])
        self._index = Hnsw(
            dim=dim,
            metric=self._metric,
            M=self._M,
            ef_construction=self._ef_construction,
            max_elements=max(int(X.shape[0]), 1_000),
        )
        self._index.fit(X)
        # Re-apply any ef set BEFORE fit (BaseANN sometimes does this).
        if self._ef_query is not None:
            self._index.set_ef(self._ef_query)

    def set_query_arguments(self, ef: int) -> None:
        self._ef_query = int(ef)
        if self._index is not None:
            self._index.set_ef(self._ef_query)

    def query(self, v, n: int):
        if self._index is None:
            raise RuntimeError("query() called before fit()")
        v = np.ascontiguousarray(np.asarray(v, dtype=np.float32))
        # knn_query returns an int64 ndarray.  ann-benchmarks expects a
        # sequence of label IDs — np.ndarray works directly.
        return self._index.knn_query(v, k=n)

    def __str__(self) -> str:
        return self.name
