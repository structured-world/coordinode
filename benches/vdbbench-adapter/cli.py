# Copyright 2026 Structured World Foundation.
# Licensed under the Apache License, Version 2.0.
"""``vectordbbench coordinodehnsw`` CLI subcommand (gRPC adapter)."""

from __future__ import annotations

from typing import Annotated, Unpack

import click

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

DBTYPE = DB.CoordiNode


class CoordinodeTypedDict(CommonTypedDict):
    host: Annotated[
        str,
        click.option(
            "--host",
            type=str,
            help="coordinode-server host (default: localhost)",
            default="localhost",
        ),
    ]
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            help="coordinode-server gRPC port (default: 7080)",
            default=7080,
        ),
    ]
    m: Annotated[
        int,
        click.option("--m", type=int, help="HNSW maximum neighbours (M)", default=16),
    ]
    ef_construction: Annotated[
        int,
        click.option(
            "--ef-construction",
            type=int,
            help="HNSW efConstruction (build-time candidate list)",
            default=200,
        ),
    ]
    ef_search: Annotated[
        int,
        click.option(
            "--ef-search",
            type=int,
            help="HNSW efSearch (query-time candidate list)",
            default=50,
        ),
    ]
    max_elements: Annotated[
        int,
        click.option(
            "--max-elements",
            type=int,
            help="Initial capacity hint",
            default=1_000_000,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(CoordinodeTypedDict)
def CoordinodeHNSW(**parameters: Unpack[CoordinodeTypedDict]):
    from .config import CoordinodeConfig, CoordinodeHnswConfig

    run(
        db=DBTYPE,
        db_config=CoordinodeConfig(
            host=parameters["host"],
            port=parameters["port"],
        ),
        db_case_config=CoordinodeHnswConfig(
            M=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            max_elements=parameters["max_elements"],
        ),
        **parameters,
    )
