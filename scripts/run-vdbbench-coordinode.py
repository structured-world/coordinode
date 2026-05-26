#!/usr/bin/env python3
# Copyright 2026 Structured World Foundation.
# Licensed under the Apache License, Version 2.0.
"""Runtime entry-point to run a VDBBench case against CoordiNode.

VDBBench's CLI (`vectordbbench coordinodehnsw ...`) requires our DB
type to be registered in upstream files
(`vectordb_bench/backend/clients/__init__.py` DB enum + factory
methods, plus `vectordb_bench/cli/vectordbbench.py` bootstrap). For
CI we'd rather not patch those upstream files on every run — every
VDBBench update would wipe the patch.

This script instead patches VDBBench's `DB` enum at runtime
(extend-enum style), then drives the bench programmatically via the
same `run()` helper the CLI uses. The adapter itself is imported
from `$GITHUB_WORKSPACE/benches/vdbbench-adapter/` (added to
sys.path), so the registration tracks the current commit's adapter
code, not whatever lives on disk under /opt/vdbbench.

Usage:
    scripts/run-vdbbench-coordinode.py \
        --case Performance1536D50K \
        --host 127.0.0.1 --port 7090 \
        --out-dir vdbbench-raw

Output: vdbbench-raw/result_<date>_<id>_coordinode.json — same shape
VDBBench's CLI writes. The post-step (vdbbench-to-json.py) flattens
these into the canonical bench-results schema.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent


def _register_coordinode_db() -> "object":
    """Add `DB.CoordiNode` and wire init_cls / config_cls /
    case_config_cls for it, without touching the upstream file.

    Implementation: VDBBench's `DB` is a stdlib `enum.Enum`. We
    can't truly extend a closed enum, but the factory methods
    (`init_cls`, `config_cls`, `case_config_cls`) are unbound
    property/method functions — we can swap them with versions
    that handle our extra constant.

    Returns the CoordiNode-DB-shaped pseudo-enum member that
    `run()` will accept (matches `__eq__` semantics).
    """
    from vectordb_bench.backend.clients import DB, IndexType  # noqa: F401

    # Inject our adapter dir into sys.path so the imports below
    # resolve to THIS commit's code, not /opt/vdbbench/.../coordinode/.
    sys.path.insert(0, str(WORKSPACE / "benches" / "vdbbench-adapter"))

    from coordinode import (  # type: ignore[import-not-found]
        CoordinodeDB,
        CoordinodeConfig,
        CoordinodeHnswConfig,
    )

    # Create a synthetic "DB.CoordiNode" — a real Enum can't accept
    # post-hoc members, so we hand-roll an object with `.value`,
    # `.name`, and the three factory accessors the framework looks
    # at when dispatching from a TaskConfig.
    class _CoordiNodeEnum:
        value = "CoordiNode"
        name = "CoordiNode"

        @property
        def init_cls(self):
            return CoordinodeDB

        @staticmethod
        def config_cls():
            return CoordinodeConfig

        @staticmethod
        def case_config_cls(_index_type=IndexType.HNSW):
            return CoordinodeHnswConfig

        def __eq__(self, other):
            return getattr(other, "value", None) == "CoordiNode" or other == "CoordiNode"

        def __hash__(self):
            return hash("CoordiNode")

    inst = _CoordiNodeEnum()
    DB.CoordiNode = inst  # type: ignore[attr-defined]
    return (inst, CoordinodeConfig, CoordinodeHnswConfig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="VDBBench case-type, e.g. Performance1536D50K")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7090)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", type=int, default=100)
    ap.add_argument("--max-elements", type=int, default=1_000_000)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="where to copy VDBBench's raw result JSON for later flattening")
    args = ap.parse_args()

    db_enum, CoordinodeConfig, CoordinodeHnswConfig = _register_coordinode_db()

    from vectordb_bench.cli.cli import run

    # `run()` calls `task = TaskConfig(db=db, ...)` and dispatches via
    # `db.init_cls`, etc. — our pseudo-enum implements all the hooks
    # the dispatch path touches.
    run(
        db=db_enum,
        db_config=CoordinodeConfig(host=args.host, port=args.port),
        db_case_config=CoordinodeHnswConfig(
            M=args.m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            max_elements=args.max_elements,
        ),
        case_type=args.case,
        db_label=os.environ.get("BENCH_DB_LABEL", "coordinode-ci"),
        # VDBBench's `run()` expects every key in `CommonTypedDict`;
        # leave the rest at their framework defaults.
        config_file=None,
        drop_old=True,
        load=True,
        search_serial=True,
        search_concurrent=True,
        load_concurrency=0,
        num_concurrency="1,4,16",
        concurrency_duration=30,
        concurrency_timeout=3600,
        k=100,
        dry_run=False,
    )

    # Copy the raw result JSON VDBBench wrote to its own results dir
    # (under the VDBBench package) into the caller-provided out-dir
    # so `vdbbench-to-json.py` can flatten them per shard.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import vectordb_bench
        vdbb_dir = Path(vectordb_bench.__file__).resolve().parent / "results" / "CoordiNode"
        for src in sorted(vdbb_dir.glob("result_*_coordinode.json")):
            shutil.copy2(src, args.out_dir / src.name)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to harvest VDBBench raw results: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
