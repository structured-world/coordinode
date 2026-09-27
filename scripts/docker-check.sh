#!/usr/bin/env bash
# Build the release image natively for a Docker daemon's own platform and
# smoke-test it:
#   - the binary reports its version;
#   - a container started from it becomes healthy by the image's own
#     HEALTHCHECK;
#   - `coordinode healthcheck --config` inside a container probes the ops
#     address the server's config file sets, and a bare `healthcheck` there
#     fails because nothing listens on the default one.
# Everything the check creates (containers, image, the buildx builder with its
# cache, the temporary config) is removed afterwards, whatever the outcome, so
# a shared machine keeps nothing.
#
# Usage:
#   scripts/docker-check.sh                    # the local Docker daemon
#   scripts/docker-check.sh ssh://user@host    # a remote daemon over SSH;
#                                              # the build context is sent from here
#
# Run it once per architecture on a machine of that architecture: an emulated
# build is slow enough to hide nothing and to prove nothing about native code.

set -euo pipefail

ENDPOINT="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="coordinode-check-$$"
IMAGE="coordinode:${RUN_ID}"

# Every docker call, buildx included, talks to the one daemon: a builder
# created against a separate endpoint would still `--load` into the local one.
if [[ -n "$ENDPOINT" ]]; then
    export DOCKER_HOST="$ENDPOINT"
fi

CONTAINERS=()
CONFIG=""
cleanup() {
    # The `+` form keeps an empty array from tripping `set -u` on bash 3.
    for id in ${CONTAINERS[@]+"${CONTAINERS[@]}"}; do
        docker rm -f "$id" >/dev/null 2>&1 || true
    done
    docker image rm -f "$IMAGE" >/dev/null 2>&1 || true
    docker buildx rm --force "$RUN_ID" >/dev/null 2>&1 || true
    if [[ -n "$CONFIG" ]]; then
        rm -f "$CONFIG"
    fi
}
trap cleanup EXIT

fail() {
    local id="$1"
    shift
    docker logs "$id" >&2 || true
    echo "$*" >&2
    exit 1
}

# Wait up to 60 s for container `$1` to answer `coordinode healthcheck` with
# the remaining arguments.
wait_check() {
    local id="$1"
    shift
    for _ in $(seq 1 60); do
        if docker exec "$id" /coordinode healthcheck "$@" >/dev/null 2>&1; then
            return 0
        fi
        if [[ "$(docker inspect -f '{{.State.Running}}' "$id")" != "true" ]]; then
            fail "$id" "the server exited before it became ready"
        fi
        sleep 1
    done
    fail "$id" "coordinode healthcheck $* did not pass within 60 s"
}

if [[ -z "$(ls -A "$ROOT/proto" 2>/dev/null)" ]]; then
    echo "proto/ is empty: run 'git submodule update --init' first" >&2
    exit 1
fi

PLATFORM="linux/$(docker version --format '{{.Server.Arch}}')"
echo "==> building $IMAGE for $PLATFORM on ${ENDPOINT:-the local daemon}"

# A builder of its own, so its cache is this run's alone and goes with it.
docker buildx create --name "$RUN_ID" --driver docker-container >/dev/null
docker buildx build --builder "$RUN_ID" --platform "$PLATFORM" \
    --progress plain --tag "$IMAGE" --load "$ROOT"

echo "==> version"
docker run --rm "$IMAGE" version

echo "==> the image's HEALTHCHECK"
# Run every second here instead of on its production schedule, so the check
# exercises what a deployment relies on.
SERVER_ID="$(docker run -d --health-interval 1s --health-start-period 0s "$IMAGE")"
CONTAINERS+=("$SERVER_ID")
healthy=false
for _ in $(seq 1 60); do
    case "$(docker inspect -f '{{.State.Health.Status}}' "$SERVER_ID")" in
        healthy)
            healthy=true
            break
            ;;
        unhealthy) fail "$SERVER_ID" "the image's health check failed" ;;
    esac
    if [[ "$(docker inspect -f '{{.State.Running}}' "$SERVER_ID")" != "true" ]]; then
        fail "$SERVER_ID" "the server exited before it became healthy"
    fi
    sleep 1
done
if [[ "$healthy" != "true" ]]; then
    fail "$SERVER_ID" "the server did not become healthy within 60 s"
fi

echo "==> healthcheck --config"
# The config moves the ops port off the default; it is copied into the
# container, which works against a remote daemon too.
CONFIG="$(mktemp)"
printf 'ops_addr: "127.0.0.1:9184"\n' >"$CONFIG"
CONFIG_ID="$(docker create "$IMAGE" serve --config /coordinode.conf --addr '[::]:7080' --data /data)"
CONTAINERS+=("$CONFIG_ID")
docker cp "$CONFIG" "$CONFIG_ID:/coordinode.conf"
docker start "$CONFIG_ID" >/dev/null
wait_check "$CONFIG_ID" --config /coordinode.conf
if docker exec "$CONFIG_ID" /coordinode healthcheck >/dev/null 2>&1; then
    fail "$CONFIG_ID" "a bare healthcheck passed although nothing listens on the default ops port"
fi

echo "==> $PLATFORM image is healthy"
