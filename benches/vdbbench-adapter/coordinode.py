# Copyright 2026 Structured World Foundation.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""``CoordinodeDB``: VectorDB-shaped wrapper over ``coordinode_embedded.Hnsw``.

VDBBench expects:

* ``insert_embeddings(embeddings, metadata)`` — caller-supplied integer IDs
  in ``metadata``; the DB must round-trip the same IDs from ``search``.
* ``search_embedding`` returns the original IDs, not sequential row indices.

Our ``Hnsw.fit`` auto-assigns its own sequential IDs (``[start, end)``
range) per call. We translate via an ``np.ndarray[int64]`` lookup keyed
on the engine-assigned ID — O(1) per query result, allocation-free per
search.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from vectordb_bench.backend.clients.api import DBCaseConfig, VectorDB
from vectordb_bench.backend.filter import Filter, FilterOp

from coordinode_embedded import Hnsw

from .config import CoordinodeHnswConfig

log = logging.getLogger(__name__)


class CoordinodeDB(VectorDB):
    """In-process HNSW. Single-process by construction — no thread sharing
    across runner workers (deep-copied per VDBBench's standard pattern)."""

    name = "CoordiNode"
    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter]
    thread_safe = False

    def __init__(
        self,
        dim: int,
        db_config: dict[str, Any],
        db_case_config: DBCaseConfig | None,
        collection_name: str = "coordinode_bench",
        drop_old: bool = False,
        **_kwargs: Any,
    ) -> None:
        del db_config, collection_name, drop_old  # in-process: no persistent state
        if db_case_config is None:
            db_case_config = CoordinodeHnswConfig()
        if not isinstance(db_case_config, CoordinodeHnswConfig):
            raise TypeError(
                f"CoordinodeDB requires CoordinodeHnswConfig, got {type(db_case_config).__name__}"
            )
        self._dim = int(dim)
        self._case = db_case_config
        self._index: Hnsw | None = None
        # engine_seq_id -> caller metadata id (np.int64 to match Hnsw.knn_query dtype)
        self._id_map: np.ndarray = np.empty(0, dtype=np.int64)

    @contextmanager
    def init(self) -> Iterator[None]:
        """Lazy index creation on first use; ``optimize`` is a no-op so all
        work happens inside the ``with`` block."""
        params = self._case.index_param()
        self._index = Hnsw(
            dim=self._dim,
            metric=params["metric"],
            M=params["M"],
            ef_construction=params["ef_construction"],
            max_elements=params["max_elements"],
        )
        # ``ef_search`` is independent of build params — apply now so a
        # search before insert (rare but legal) doesn't see the engine default.
        self._index.set_ef(self._case.search_param()["ef_search"])
        try:
            yield
        finally:
            # Drop the Rust handle so memory is reclaimed between cases.
            self._index = None
            self._id_map = np.empty(0, dtype=np.int64)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **_kwargs: Any,
    ) -> tuple[int, Exception | None]:
        if self._index is None:
            raise RuntimeError("insert_embeddings before init()")
        del labels_data  # no filter support yet
        if len(embeddings) != len(metadata):
            return 0, ValueError(
                f"length mismatch: embeddings={len(embeddings)} metadata={len(metadata)}"
            )
        try:
            arr = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
            meta = np.asarray(metadata, dtype=np.int64)
            start, end = self._index.fit(arr)
            assert end - start == len(metadata), (
                f"engine assigned {end - start} ids for {len(metadata)} rows; "
                "concurrent fit() from another runner thread?"
            )
            # Grow id_map so [start..end) maps to caller ids in order.
            if end > self._id_map.shape[0]:
                grown = np.empty(end, dtype=np.int64)
                grown[: self._id_map.shape[0]] = self._id_map
                self._id_map = grown
            self._id_map[start:end] = meta
            return len(metadata), None
        except Exception as exc:  # noqa: BLE001
            log.exception("insert_embeddings failed")
            return 0, exc

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: Filter | None = None,
        **_kwargs: Any,
    ) -> list[int]:
        if self._index is None:
            raise RuntimeError("search_embedding before init()")
        if filters is not None and filters.type is not FilterOp.NonFilter:
            raise NotImplementedError(
                "CoordiNode VDBBench adapter does not support filter pushdown yet"
            )
        q = np.ascontiguousarray(np.asarray(query, dtype=np.float32))
        ids = self._index.knn_query(q, k=k)  # int64 ndarray, len ≤ k
        # Translate engine seq ids -> caller metadata ids in one shot.
        return self._id_map[ids].tolist()

    def optimize(self, data_size: int | None = None) -> None:
        # HNSW is built incrementally during insert; nothing to do here.
        del data_size

    def need_normalize_cosine(self) -> bool:
        # The engine handles cosine via the metric kernel directly —
        # caller does NOT need to pre-normalize.
        return False
