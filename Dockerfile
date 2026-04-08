# CoordiNode — multi-stage musl build
# Produces a scratch-based image ~25MB with zero runtime dependencies.
#
# Build context is the repository root:
#   docker build -t coordinode .
#
# Proto files are in the `proto/` git submodule. Run `git submodule update --init`
# before building if proto/ is empty.

# ─── Stage 1: Builder ────────────────────────────────────────────────
FROM rust:1.94-bookworm AS builder

# Override rust-toolchain.toml — Docker uses image's stable toolchain.
ENV RUSTUP_TOOLCHAIN=stable

RUN apt-get update && apt-get install -y --no-install-recommends \
        musl-tools \
        protobuf-compiler \
        libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

# Detect build architecture and add appropriate musl target
RUN case "$(uname -m)" in \
        x86_64)  rustup target add x86_64-unknown-linux-musl ;; \
        aarch64) rustup target add aarch64-unknown-linux-musl ;; \
        *)       echo "Unsupported architecture: $(uname -m)" && exit 1 ;; \
    esac

WORKDIR /build

# Copy proto submodule + workspace
COPY proto/ /build/proto/
COPY Cargo.toml Cargo.lock rust-toolchain.toml /build/
COPY crates/ /build/crates/

# Build the coordinode binary (static musl link, release profile with LTO)
RUN MUSL_TARGET="$(uname -m)-unknown-linux-musl" \
    && cargo build --release --target "$MUSL_TARGET" --bin coordinode \
    && strip "target/$MUSL_TARGET/release/coordinode" \
    && cp "target/$MUSL_TARGET/release/coordinode" /coordinode-bin

# Build structured-proxy (gRPC→REST transcoding)
RUN MUSL_TARGET="$(uname -m)-unknown-linux-musl" \
    && cargo install structured-proxy --target "$MUSL_TARGET" --root /proxy-out \
    && strip /proxy-out/bin/structured-proxy

# ─── Stage 2: Runtime (scratch, static binary) ──────────────────────
FROM scratch

# Labels
LABEL org.opencontainers.image.title="CoordiNode"
LABEL org.opencontainers.image.description="Distributed graph+vector database"
LABEL org.opencontainers.image.vendor="structured.world"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"
LABEL org.opencontainers.image.source="https://github.com/structured-world/coordinode"

# Copy static binaries
COPY --from=builder /coordinode-bin /coordinode
COPY --from=builder /proxy-out/bin/structured-proxy /structured-proxy

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
