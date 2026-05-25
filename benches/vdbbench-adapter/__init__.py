# Copyright 2026 Structured World Foundation.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""VDBBench adapter for CoordiNode.

In-process HNSW path via ``coordinode_embedded.Hnsw``. For the networked
gRPC path against multi-tenant / server-class competitors (Milvus, Qdrant
Cloud, Weaviate), wire ``coordinode_embedded.LocalClient`` or
``CoordinodeClient`` from ``coordinode`` instead — same adapter shape,
swap out the index handle.
"""

from .config import CoordinodeConfig, CoordinodeHnswConfig
from .coordinode import CoordinodeDB

__all__ = ["CoordinodeConfig", "CoordinodeHnswConfig", "CoordinodeDB"]
