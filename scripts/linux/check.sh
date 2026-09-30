#!/usr/bin/env bash
# Run the CI gate (clippy, server build, nextest, doc tests) for the current
# working tree on a Linux machine, uncommitted and untracked changes included,
# without touching the local index, branches or working tree.
#
# Usage: COORDINODE_LINUX_HOST=<ssh target> scripts/linux/check.sh
#
# With COORDINODE_CHECK_NEXTEST set, only nextest runs, with those arguments
# added (a filter and a stress count to chase a flaky test, for example:
# COORDINODE_CHECK_NEXTEST='-E test(name) --stress-count 20').
#
# The host needs git, a Rust toolchain, cargo-nextest and protoc. Logs and the
# status file land in target/linux-check/ locally; the run's directory on the
# host, build output included, is removed afterwards.
set -euo pipefail

host="${COORDINODE_LINUX_HOST:?set COORDINODE_LINUX_HOST to the ssh target of the Linux machine}"
# A directory of this run's own, so concurrent runs and other users of a
# shared machine never meet in it.
remote_root="/var/tmp/cn-check-$(date +%Y%m%d%H%M%S)-$$"
ref='refs/check/linux'
only_nextest="${COORDINODE_CHECK_NEXTEST:-}"

repo="$(git rev-parse --show-toplevel)"
out="$repo/target/linux-check"
mkdir -p "$out"
bundle="$out/tree.bundle"

# Snapshot the working tree as a commit on a private ref, through a scratch
# index so the real one is left as it is.
index="$(mktemp)"
cleanup() {
  rm -f "$index"
  git -C "$repo" update-ref -d "$ref" 2>/dev/null || true
  ssh "$host" "rm -rf '$remote_root'" || true
}
trap cleanup EXIT
cp "$repo/.git/index" "$index"
GIT_INDEX_FILE="$index" git -C "$repo" add -A
tree="$(GIT_INDEX_FILE="$index" git -C "$repo" write-tree)"
commit="$(git -C "$repo" commit-tree "$tree" -p HEAD -m 'linux check snapshot')"
git -C "$repo" update-ref "$ref" "$commit"
git -C "$repo" bundle create -q "$bundle" "$ref" HEAD

ssh "$host" "mkdir -p '$remote_root'"
ssh "$host" "cat > '$remote_root/tree.bundle'" < "$bundle"

if [ -n "$only_nextest" ]; then
  ssh "$host" "set -u
cd '$remote_root'
git clone -q --no-checkout tree.bundle src
cd src
git fetch -q ../tree.bundle '$ref'
git checkout -q --detach FETCH_HEAD
git submodule update --init -q
echo checkout=\$? > ../status.txt
export RUSTFLAGS='-D warnings' COORDINODE_TEST_RAFT_GENEROUS_TIMEOUTS=1 CARGO_TARGET_DIR='$remote_root/target'
cargo nextest run --all-features --workspace --no-fail-fast --failure-output final $only_nextest > ../test.log 2>&1
echo nextest=\$? >> ../status.txt
echo done >> ../status.txt" || true
  for f in status.txt test.log; do
    ssh "$host" "cat '$remote_root/$f'" > "$out/$f" 2>/dev/null || true
  done
  cat "$out/status.txt"
  exit 0
fi

# Each step's exit code goes to status.txt; a failing step does not stop the
# ones after it, and the logs are fetched either way.
ssh "$host" "set -u
cd '$remote_root'
git clone -q --no-checkout tree.bundle src
cd src
git fetch -q ../tree.bundle '$ref'
git checkout -q --detach FETCH_HEAD
git submodule update --init -q
echo checkout=\$? > ../status.txt
export RUSTFLAGS='-D warnings' COORDINODE_TEST_RAFT_GENEROUS_TIMEOUTS=1 CARGO_TARGET_DIR='$remote_root/target'
cargo clippy --workspace --all-targets --all-features -- -D warnings > ../clippy.log 2>&1
echo clippy=\$? >> ../status.txt
cargo build --all-features -p coordinode-server > ../build.log 2>&1
echo build=\$? >> ../status.txt
cargo nextest run --all-features --workspace --no-fail-fast --status-level fail --final-status-level fail --failure-output final > ../test.log 2>&1
echo nextest=\$? >> ../status.txt
cargo test --doc --all-features > ../doctest.log 2>&1
echo doctest=\$? >> ../status.txt
echo done >> ../status.txt" || true

for f in status.txt clippy.log build.log test.log doctest.log; do
  ssh "$host" "cat '$remote_root/$f'" > "$out/$f" 2>/dev/null || true
done

cat "$out/status.txt"
