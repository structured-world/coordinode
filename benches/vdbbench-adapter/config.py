# Copyright 2026 Structured World Foundation.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""VDBBench DBConfig / DBCaseConfig for CoordiNode.

Mirrors the chroma / lancedb in-process adapters: the "connection" is
literally the constructor of a Rust HnswIndex; no host/port/auth fields.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field

from vectordb_bench.backend.clients.api import (
    DBCaseConfig,
    DBConfig,
    MetricType,
)


_METRIC_MAP: dict[MetricType, str] = {
    MetricType.L2: "euclidean",
    MetricType.COSINE: "cosine",
    MetricType.IP: "dot",
}


class CoordinodeConfig(DBConfig):
    """Connection config — empty by design (in-process)."""

    def to_dict(self) -> dict[str, Any]:
        return {}


class CoordinodeHnswConfig(BaseModel, DBCaseConfig):
    """HNSW build / search parameters.

    Mirrors the names from ``coordinode_vector::hnsw::HnswConfig`` so the
    knob set lines up with what the engine README documents — no surprise
    aliasing.
    """

    metric_type: MetricType = MetricType.L2
    M: Annotated[int, Field(ge=2, le=128)] = 16
    ef_construction: Annotated[int, Field(ge=8, le=2048)] = 200
    ef_search: Annotated[int, Field(ge=1, le=4096)] = 50
    max_elements: Annotated[int, Field(ge=1, le=10**9)] = 1_000_000

    def index_param(self) -> dict[str, Any]:
        return {
            "metric": _METRIC_MAP.get(self.metric_type, "euclidean"),
            "M": self.M,
            "ef_construction": self.ef_construction,
            "max_elements": self.max_elements,
        }

    def search_param(self) -> dict[str, Any]:
        return {"ef_search": self.ef_search}
