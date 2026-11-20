# CoordiNode image: the server built against Debian's glibc, on a slim Debian
# base that carries a shell and the tools operators use around it (backup and
# restore archives, transfers, process inspection).
#
# Build context is the repository root:
#   docker build -t coordinode .
#
# Proto files are in the `proto/` git submodule. Run `git submodule update --init`
# before building if proto/ is empty.

# ─── Stage 1: Builder ────────────────────────────────────────────────
# The same Debian release as the runtime stage, so the binary links against
# the glibc it runs with.
FROM rust:1-trixie AS builder

# The current stable image, as rust-toolchain.toml names stable: the file is
# authoritative for the build, and a base image on the same channel needs no
# second toolchain pulled at build time.

RUN apt-get update && apt-get install -y --no-install-recommends \
        protobuf-compiler \
        libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY rust-toolchain.toml /build/

# Copy proto submodule + workspace
COPY proto/ /build/proto/
COPY Cargo.toml Cargo.lock /build/
COPY crates/ /build/crates/
# openraft's source: the workspace depends on it by path.
COPY vendor/ /build/vendor/
# Integration test crate is a workspace member: Cargo needs its Cargo.toml
# for workspace resolution even when building only the server binary.
# The test code itself is not compiled during Docker build.
COPY tests/ /build/tests/

# Build the coordinode binary (release profile with LTO).
# REST/JSON proxy (port 7081) is embedded via the rest-proxy feature (default).
# SYMBOLS=1 keeps the symbol table and line tables, same optimisation, so a
# profiler attached to the running server names functions:
#   docker build --build-arg SYMBOLS=1 -t coordinode:symbols .
ARG SYMBOLS=0
RUN if [ "$SYMBOLS" = 1 ]; then \
         export CARGO_PROFILE_RELEASE_STRIP=none CARGO_PROFILE_RELEASE_DEBUG=line-tables-only; \
       fi \
    && cargo build --locked --release --bin coordinode \
    && cp target/release/coordinode /coordinode-bin

# ─── Stage 2: Runtime ────────────────────────────────────────────────
FROM debian:trixie-slim

LABEL org.opencontainers.image.title="CoordiNode"
LABEL org.opencontainers.image.description="Distributed graph+vector database"
LABEL org.opencontainers.image.vendor="Dmitry Prudnikov"
LABEL org.opencontainers.image.authors="Dmitry Prudnikov <mail@polaz.com>"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"
LABEL org.opencontainers.image.source="https://github.com/structured-world/coordinode"

# A shell and the tools a backup, a restore or an investigation needs inside
# the container: archives and their compressors, a client to move dumps,
# process inspection, CA roots for TLS and time zones for timestamps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        gzip \
        less \
        procps \
        tar \
        tzdata \
        xz-utils \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# The server runs unprivileged: code that took over the process does not hold
# root in the container. uid 65532 owns the data directory, and a named volume
# mounted there starts out owned by it too.
RUN groupadd --system --gid 65532 coordinode \
    && useradd --system --uid 65532 --gid coordinode --home-dir /data \
         --shell /bin/bash coordinode \
    && install -d -o coordinode -g coordinode -m 0750 /data

COPY --from=builder /coordinode-bin /usr/local/bin/coordinode

USER coordinode
WORKDIR /data

VOLUME ["/data"]

# Ports:
#   7080 - gRPC (native API + inter-node)
#   7081 - HTTP/REST (S3, GraphQL, management)
#   7082 - Bolt (Neo4j wire protocol)
#   7083 - WebSocket (subscriptions)
#   7084 - HTTP (Prometheus /metrics, /health, /ready)
EXPOSE 7080 7081 7082 7083 7084

# Healthy once the server answers /ready on its ops port: 503 while starting
# and while draining. A server run with --config is checked with the same file
# (`healthcheck --config FILE`); one run with another --ops-addr, with that
# address (`healthcheck --ops-addr 127.0.0.1:PORT`).
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD ["coordinode", "healthcheck"]

ENTRYPOINT ["coordinode"]
CMD ["serve", "--addr", "[::]:7080", "--data", "/data"]
