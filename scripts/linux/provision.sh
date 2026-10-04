#!/usr/bin/env bash
# Prepare a Linux machine for scripts/linux/check.sh: the packages the build
# needs, rustup with the toolchain this repository pins, cargo-nextest and a
# compilation cache. Safe to run again: whatever is already there is kept.
#
# Usage: COORDINODE_LINUX_HOST=<ssh target> [COORDINODE_SCCACHE_DIR=<dir>] \
#          scripts/linux/provision.sh
#
# Supported hosts: dnf (Fedora, RHEL family) and apt-get (Debian, Ubuntu).
# A ~/.cargo/config.toml this script did not write is never rewritten, since
# other projects on a shared machine build with it too; when it does not route
# rustc through sccache the script says so and leaves it alone. The one it
# wrote is rewritten, so a cache placed on a full disk moves when run again.
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

ssh "$host" bash -s -- "$channel" "$nextest_version" "${COORDINODE_SCCACHE_DIR:-}" <<'REMOTE'
set -euo pipefail
channel="$1"
nextest_version="$2"
# Absent when empty: ssh joins the arguments into one command line.
cache_dir_override="${3:-}"

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

# The compilation cache goes on the filesystem with the most room among the
# home directory and /srv (a host often keeps a small root disk and a large
# data disk), sized to fit it: 40G, or 80% of that filesystem when smaller.
# COORDINODE_SCCACHE_DIR names the directory instead.
old_cache="$HOME/.cache/sccache"
cache_dir="${cache_dir_override:-}"
if [ -z "$cache_dir" ]; then
  cache_dir="$old_cache"
  best="$(df --output=avail "$HOME" | tail -n 1)"
  if [ -d /srv ] && [ "$(df --output=avail /srv | tail -n 1)" -gt "$best" ]; then
    cache_dir="/srv/sccache-$(id -un)"
  fi
fi
mkdir -p "$cache_dir"
fs_kib="$(df --output=size "$cache_dir" | tail -n 1)"
cache_gib=$(( fs_kib * 8 / 10 / 1024 / 1024 ))
if [ "$cache_gib" -gt 40 ]; then
  cache_gib=40
fi
if [ "$cache_gib" -lt 1 ]; then
  echo "the filesystem of $cache_dir is too small for a compilation cache" >&2
  exit 1
fi

# The config this script writes starts with this line; one that does not is
# someone else's and stays as it is.
marker='# Compilation cache for every cargo invocation on this host, including the'
config="$HOME/.cargo/config.toml"
if [ ! -e "$config" ] || [ "$(head -n 1 "$config")" = "$marker" ]; then
  mkdir -p "$HOME/.cargo"
  cat > "$config" <<CONFIG
$marker
# non-interactive ssh shells the remote check scripts run in.
[build]
rustc-wrapper = "sccache"

[env]
# Read by the sccache server when it starts: \`sccache --stop-server\` after
# changing these.
SCCACHE_DIR = "$cache_dir"
SCCACHE_CACHE_SIZE = "${cache_gib}G"

# The same dev profile as the developer machines, so the profile part of
# sccache's cache key matches and a build script and the library built from
# one crate resolve the same way in package and workspace builds.
[profile.dev]
debug = "line-tables-only"

[profile.dev.build-override]
debug = "line-tables-only"
CONFIG
  # A cache an earlier run left in the default place moves along (its
  # entries stay valid), and the server restarts on the new settings.
  if [ "$cache_dir" != "$old_cache" ] && [ -d "$old_cache" ]; then
    sccache --stop-server >/dev/null 2>&1 || true
    cp -a "$old_cache/." "$cache_dir/"
    rm -rf "$old_cache"
  fi
  sccache --stop-server >/dev/null 2>&1 || true
elif ! grep -qs 'rustc-wrapper *= *"sccache"' "$config"; then
  echo "note: $config exists and does not use sccache; left unchanged" >&2
fi
echo "sccache: $cache_dir, ${cache_gib}G"

echo "rust $(rustup run "$channel" rustc --version)"
echo "$(rustup run "$channel" cargo nextest --version | head -n 1)"
echo "$(protoc --version)"
echo "$(sccache --version)"
REMOTE
