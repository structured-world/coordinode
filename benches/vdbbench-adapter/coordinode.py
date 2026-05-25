# Copyright 2026 Structured World Foundation.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""``CoordinodeDB``: VectorDB-shaped wrapper around the CoordiNode gRPC server.

The earlier revision of this file wrapped the ``coordinode_embedded.Hnsw``
PyO3 handle directly — an in-process index in the runner's Python
process. VDBBench's runners use ``multiprocessing.spawn`` for both insert
and search phases, so the index built in the insert subprocess does NOT
survive into the search subprocess and recall collapses to ~0. That
matches the pattern Chroma / Milvus / Qdrant adapters all follow:
talk to a **server** that persists data outside the Python process.

So this adapter speaks gRPC to ``coordinode-server`` (the same binary
that drives production CoordiNode deployments). Every spawn subprocess
opens its own client to the shared server, and the underlying graph +
vector data lives there. That makes the numbers a fair Level B (server-
class) comparison against the other VDBBench engines.

Schema and indexing model:

- One label per case, ``Vec`` by default. Idempotent ``CREATE VECTOR INDEX``
  on first ``init()``; subsequent subprocesses no-op against the same
  schema.
- One node per inserted vector: ``(:Vec {id: <int>, vec: <float[D]>})``.
- KNN via ``vector_distance(n.vec, $q)`` ORDER BY ASC LIMIT k. Returns
  the same ``id`` integers VDBBench handed us as ``metadata``.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from vectordb_bench.backend.clients.api import DBCaseConfig, VectorDB
from vectordb_bench.backend.filter import Filter, FilterOp

from coordinode import CoordinodeClient

from .config import CoordinodeHnswConfig

log = logging.getLogger(__name__)

DEFAULT_LABEL = "Vec"


class CoordinodeDB(VectorDB):
    """gRPC client wrapper. One server, many subprocesses."""

    name = "CoordiNode"
    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter]
    # gRPC channels are thread-safe; multiple workers can share one
    # connection.  The actual concurrency is bounded by the server's
    # request scheduler, not by Python-side serialisation.
    thread_safe = True

    def __init__(
        self,
        dim: int,
        db_config: dict[str, Any],
        db_case_config: DBCaseConfig | None,
        collection_name: str = "Vec",
        drop_old: bool = False,
        **_kwargs: Any,
    ) -> None:
        if db_case_config is None:
            db_case_config = CoordinodeHnswConfig()
        if not isinstance(db_case_config, CoordinodeHnswConfig):
            raise TypeError(
                f"CoordinodeDB requires CoordinodeHnswConfig, got {type(db_case_config).__name__}"
            )
        self._dim = int(dim)
        self._case = db_case_config
        self._host = str(db_config.get("host", "localhost"))
        # CoordinodeClient accepts port=None → falls back to default (7080).
        self._port = db_config.get("port")
        self._label = collection_name or DEFAULT_LABEL
        self._drop_old = bool(drop_old)
        # gRPC client lives only while inside ``init()``.  Re-opened in
        # every spawned subprocess.
        self._client: CoordinodeClient | None = None

    # ── Schema bootstrap (idempotent across subprocesses) ──────────────

    def _ensure_schema(self, client: CoordinodeClient) -> None:
        """Idempotent: declare vector index. Safe to call from every
        subprocess — the first one materialises the index, the rest
        find it already there and the duplicate-create error is
        swallowed.

        CoordiNode auto-creates FLEXIBLE labels on first node insert,
        so we don't need an explicit `CREATE LABEL`. The vector index
        DDL syntax does NOT accept `IF NOT EXISTS` at the moment, so
        we catch the duplicate-create error and move on.
        """
        params = self._case.index_param()
        try:
            client.cypher(
                f"CREATE VECTOR INDEX vec_idx_{self._label} "
                f"ON :{self._label}(vec) "
                f"OPTIONS {{m: {params['M']}, "
                f"ef_construction: {params['ef_construction']}, "
                f'metric: "{params["metric"]}"}}'
            )
        except Exception as exc:  # noqa: BLE001
            # Duplicate-create from another subprocess is expected and
            # not actionable. Anything else is logged but does NOT
            # abort init — the bench can still proceed against an
            # already-built index.
            log.debug("ensure_schema: vector index ddl returned %s", exc)

    def _drop_old_data(self, client: CoordinodeClient) -> None:
        """Best-effort wipe of prior data for this label."""
        try:
            client.cypher(f"MATCH (n:{self._label}) DETACH DELETE n")
        except Exception:  # noqa: BLE001
            log.exception("drop_old: detach delete failed (continuing)")

    # ── VectorDB protocol ──────────────────────────────────────────────

    @contextmanager
    def init(self) -> Iterator[None]:
        """Open the gRPC client. The first subprocess to enter init()
        bootstraps schema + (optionally) drops old data; subsequent
        subprocesses find the schema in place and only open their own
        connection."""
        client = CoordinodeClient(self._host, self._port)
        client.__enter__()
        self._client = client
        try:
            # Drop_old fires in EVERY init() call when set, which is
            # wrong for the multi-subprocess search phase but correct
            # for the initial load.  VDBBench passes drop_old=True only
            # to the first task that opens the DB (per task_runner
            # convention), so this matches that intent.
            if self._drop_old:
                self._drop_old_data(client)
                # Reset the flag locally so re-entering init() inside
                # the same Python process doesn't wipe data we just
                # ingested.
                self._drop_old = False
            self._ensure_schema(client)
            yield
        finally:
            client.close()
            self._client = None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **_kwargs: Any,
    ) -> tuple[int, Exception | None]:
        if self._client is None:
            raise RuntimeError("insert_embeddings before init()")
        del labels_data
        if len(embeddings) != len(metadata):
            return 0, ValueError(
                f"length mismatch: embeddings={len(embeddings)} metadata={len(metadata)}"
            )
        if not metadata:
            return 0, None
        try:
            # UNWIND-driven bulk create. One round-trip per
            # `insert_embeddings` call from the runner (default
            # batch size 100 from VDBBench), which keeps the gRPC
            # overhead amortised over the batch.
            rows = [{"id": int(m), "vec": list(v)} for m, v in zip(metadata, embeddings)]
            self._client.cypher(
                f"""
                UNWIND $rows AS row
                CREATE (n:{self._label} {{id: row.id, vec: row.vec}})
                """,
                {"rows": rows},
            )
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
        if self._client is None:
            raise RuntimeError("search_embedding before init()")
        if filters is not None and filters.type is not FilterOp.NonFilter:
            raise NotImplementedError(
                "CoordiNode gRPC adapter does not support filter pushdown yet"
            )
        rows = self._client.cypher(
            f"""
            MATCH (n:{self._label})
            RETURN n.id AS id, vector_distance(n.vec, $q) AS d
            ORDER BY d ASC LIMIT $k
            """,
            {"q": list(query), "k": int(k)},
        )
        return [int(r["id"]) for r in rows]

    def optimize(self, data_size: int | None = None) -> None:
        """No-op — HNSW is maintained incrementally by the server-side
        executor; there is no explicit ``FORCE BUILD`` step in CoordiNode's
        index lifecycle."""
        del data_size

    def need_normalize_cosine(self) -> bool:
        # The server-side ``vector_distance()`` honours the metric
        # declared on the index, so the caller does NOT need to
        # pre-normalise for cosine.
        return False
