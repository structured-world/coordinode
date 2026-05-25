# Copyright 2026 Structured World Foundation.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""VDBBench adapter for CoordiNode.

gRPC client against ``coordinode-server``. Schema + data live on the
server, so the multi-subprocess spawn model VDBBench uses works
naturally — every subprocess opens its own client to the shared
server. For the in-process (library-tier) benchmark wired into
ann-benchmarks see ``benches/ann-benchmarks-adapter/`` instead.
"""

from .config import CoordinodeConfig, CoordinodeHnswConfig
from .coordinode import CoordinodeDB

__all__ = ["CoordinodeConfig", "CoordinodeHnswConfig", "CoordinodeDB"]
