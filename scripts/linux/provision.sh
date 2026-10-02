#!/usr/bin/env bash
# Prepare a Linux machine for scripts/linux/check.sh: the packages the build
# needs, rustup with the toolchain this repository pins, cargo-nextest and a
# compilation cache. Safe to run again: whatever is already there is kept.
#
# Usage: COORDINODE_LINUX_HOST=<ssh target> scripts/linux/provision.sh
#
# Supported hosts: dnf (Fedora, RHEL family) and apt-get (Debian, Ubuntu).
# An existing ~/.cargo/config.toml on the host is never rewritten, since other
# projects on a shared machine build with it too; when it does not route
# rustc through sccache the script says so and leaves it alone.
set -euo pipefail

host="${COORDINODE_LINUX_HOST:?set COORDINODE_LINUX_HOST to the ssh target of the Linux machine}"
# The version the Windows CI job installs; keep the two in step.
nextest_version='0.9.146'

repo="$(git rev-parse --show-toplevel)"
channel="$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$repo/rust-toolchain.toml")"
if [ -z "$channel" ]; then
  echo "no channel in $repo/rust-toolchain.toml" >&2
  exit 1
fi

ssh "$host" bash -s -- "$channel" "$nextest_version" <<'REMOTE'
set -euo pipefail
channel="$1"
nextest_version="$2"

sudo=''
if [ "$(id -u)" -ne 0 ]; then
  sudo='sudo'
fi

if command -v dnf >/dev/null 2>&1; then
  $sudo dnf install -y -q git gcc gcc-c++ clang make protobuf-compiler protobuf-devel rustup sccache
elif command -v apt-get >/dev/null 2>&1; then
  $sudo apt-get update -q
  $sudo apt-get install -y -q git build-essential clang protobuf-compiler libprotobuf-dev rustup sccache
else
  echo "neither dnf nor apt-get found; install the packages by hand" >&2
  exit 1
fi

# The distribution's rustup package ships rustup-init (Fedora) or rustup
# itself (Debian); either way the toolchains live under ~/.rustup.
if [ ! -x "$HOME/.cargo/bin/rustup" ] && command -v rustup-init >/dev/null 2>&1; then
  rustup-init -y -q --no-modify-path --default-toolchain none
fi
if [ -f "$HOME/.cargo/env" ]; then
  . "$HOME/.cargo/env"
fi
rustup toolchain install "$channel" --profile minimal -c rustfmt -c clippy

# Non-interactive ssh shells read ~/.bashrc, and check.sh runs cargo in one.
if [ -f "$HOME/.cargo/env" ] && ! grep -qs '\.cargo/env' "$HOME/.bashrc"; then
  printf '. "$HOME/.cargo/env"\n' >> "$HOME/.bashrc"
fi

if ! rustup run "$channel" cargo nextest --version 2>/dev/null | grep -q "cargo-nextest $nextest_version"; then
  rustup run "$channel" cargo install -q --locked "cargo-nextest@$nextest_version"
fi

config="$HOME/.cargo/config.toml"
if [ ! -e "$config" ]; then
  mkdir -p "$HOME/.cargo"
  cat > "$config" <<'CONFIG'
# Compilation cache for every cargo invocation on this host, including the
# non-interactive ssh shells the remote check scripts run in.
[build]
rustc-wrapper = "sccache"

[env]
# Read by the sccache server when it starts: `sccache --stop-server` after
# changing this.
SCCACHE_CACHE_SIZE = "40G"

# The same dev profile as the developer machines, so the profile part of
# sccache's cache key matches and a build script and the library built from
# one crate resolve the same way in package and workspace builds.
[profile.dev]
debug = "line-tables-only"

[profile.dev.build-override]
debug = "line-tables-only"
CONFIG
elif ! grep -qs 'rustc-wrapper *= *"sccache"' "$config"; then
  echo "note: $config exists and does not use sccache; left unchanged" >&2
fi

echo "rust $(rustup run "$channel" rustc --version)"
echo "$(rustup run "$channel" cargo nextest --version | head -n 1)"
echo "$(protoc --version)"
echo "$(sccache --version)"
REMOTE
