# CoordiNode — multi-stage musl build
# Produces a scratch-based image ~25MB with zero runtime dependencies.
#
# Build context is the repository root:
#   docker build -t coordinode .
#
# Proto files are in the `proto/` git submodule. Run `git submodule update --init`
# before building if proto/ is empty.

# ─── Stage 1: Builder ────────────────────────────────────────────────
FROM rust:1.98-bookworm AS builder

# Must track rust-toolchain.toml. The file is authoritative for the build, so
# a base image on a different version either pulls a second toolchain at build
# time or, if overridden, compiles with something the repository does not pin.

RUN apt-get update && apt-get install -y --no-install-recommends \
        musl-tools \
        protobuf-compiler \
        libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# The toolchain file lands before any rustup call, so the musl standard library
# is installed into the toolchain the repository pins. Add the target first and
# the file switches the build to a toolchain that has no musl std, which
# surfaces much later as "can't find crate for `core`".
COPY rust-toolchain.toml /build/

# Detect build architecture and add appropriate musl target
RUN case "$(uname -m)" in \
        x86_64)  rustup target add x86_64-unknown-linux-musl ;; \
        aarch64) rustup target add aarch64-unknown-linux-musl ;; \
        *)       echo "Unsupported architecture: $(uname -m)" && exit 1 ;; \
    esac

# Copy proto submodule + workspace
COPY proto/ /build/proto/
COPY Cargo.toml Cargo.lock /build/
COPY crates/ /build/crates/
# Integration test crate is a workspace member — Cargo needs its Cargo.toml
# for workspace resolution even when building only the server binary.
# The test code itself is not compiled during Docker build.
COPY tests/ /build/tests/
# Proto file descriptor set — embedded into the coordinode binary at compile time
# via include_bytes! for the REST/JSON proxy (structured-proxy, rest-proxy feature).
COPY coordinode.descriptor.bin /build/coordinode.descriptor.bin

# Build the coordinode binary (static musl link, release profile with LTO).
# REST/JSON proxy (port 7081) is embedded via the rest-proxy feature (default).
RUN MUSL_TARGET="$(uname -m)-unknown-linux-musl" \
    && cargo build --release --target "$MUSL_TARGET" --bin coordinode \
    && strip "target/$MUSL_TARGET/release/coordinode" \
    && cp "target/$MUSL_TARGET/release/coordinode" /coordinode-bin

# ─── Stage 2: Runtime (scratch, static binary) ──────────────────────
FROM scratch

# Labels
LABEL org.opencontainers.image.title="CoordiNode"
LABEL org.opencontainers.image.description="Distributed graph+vector database"
LABEL org.opencontainers.image.vendor="structured.world"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"
LABEL org.opencontainers.image.source="https://github.com/structured-world/coordinode"

# Copy static binary (REST proxy is embedded, no separate structured-proxy binary needed)
COPY --from=builder /coordinode-bin /coordinode

# Default data directory
VOLUME ["/data"]

# Ports:
#   7080 - gRPC (native API + inter-node)
#   7081 - HTTP/REST (S3, GraphQL, management)
#   7082 - Bolt (Neo4j wire protocol)
#   7083 - WebSocket (subscriptions)
#   7084 - HTTP (Prometheus /metrics, /health, /ready)
EXPOSE 7080 7081 7082 7083 7084

ENTRYPOINT ["/coordinode"]
CMD ["serve", "--addr", "[::]:7080", "--data", "/data"]
