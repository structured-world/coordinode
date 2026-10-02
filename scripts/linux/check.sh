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
# With COORDINODE_CHECK_BENCH set, only that `cargo bench` runs, with those
# arguments (a task's bounded measurement, for example:
# COORDINODE_CHECK_BENCH='-p coordinode-vector --bench hnsw_publication');
# its output lands in bench.log.
#
# The host needs git, the pinned Rust toolchain, cargo-nextest and protoc
# (scripts/linux/provision.sh installs them); the run stops before uploading
# anything when one is missing. Logs and the status file land in
# target/linux-check/ locally; the run's directory on the host, build output
# included, is removed afterwards.
set -euo pipefail

host="${COORDINODE_LINUX_HOST:?set COORDINODE_LINUX_HOST to the ssh target of the Linux machine}"
# One fixed directory per host, created atomically as the run's lock: the
# compilation cache keys on absolute paths, so a path that differs per run
# would never hit and every run would rebuild the workspace from nothing. A
# second run on the same host waits for no one; it stops and says why.
remote_root="/var/tmp/cn-check"
locked=0
ref='refs/check/linux'
only_nextest="${COORDINODE_CHECK_NEXTEST:-}"
only_bench="${COORDINODE_CHECK_BENCH:-}"
# Extra NAME=value assignments exported for the bench run only.
bench_env="${COORDINODE_CHECK_BENCH_ENV:-}"

repo="$(git rev-parse --show-toplevel)"

# A missing tool would otherwise surface as a failed step after the upload
# and a full checkout, with the cause buried in its log.
channel="$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$repo/rust-toolchain.toml")"
if ! ssh "$host" "missing=''
for tool in git protoc rustup; do
  command -v \$tool >/dev/null 2>&1 || missing=\"\$missing \$tool\"
done
if command -v rustup >/dev/null 2>&1; then
  rustup run '$channel' rustc --version >/dev/null 2>&1 || missing=\"\$missing rust-$channel\"
  rustup run '$channel' cargo nextest --version >/dev/null 2>&1 || missing=\"\$missing cargo-nextest\"
fi
if [ -n \"\$missing\" ]; then
  echo \"missing on the host:\$missing\" >&2
  exit 1
fi"; then
  echo "prepare the host with: COORDINODE_LINUX_HOST=$host scripts/linux/provision.sh" >&2
  exit 1
fi

out="$repo/target/linux-check"
mkdir -p "$out"
bundle="$out/tree.bundle"

# Snapshot the working tree as a commit on a private ref, through a scratch
# index so the real one is left as it is.
index="$(mktemp)"
cleanup() {
  rm -f "$index"
  git -C "$repo" update-ref -d "$ref" 2>/dev/null || true
  # Only the run that took the lock removes the directory.
  if [ "$locked" = 1 ]; then
    ssh "$host" "rm -rf '$remote_root'" || true
  fi
}
trap cleanup EXIT
# The git dir, not `$repo/.git`: in a worktree `.git` is a file.
cp "$(git -C "$repo" rev-parse --absolute-git-dir)/index" "$index"
GIT_INDEX_FILE="$index" git -C "$repo" add -A
tree="$(GIT_INDEX_FILE="$index" git -C "$repo" write-tree)"
commit="$(git -C "$repo" commit-tree "$tree" -p HEAD -m 'linux check snapshot')"
git -C "$repo" update-ref "$ref" "$commit"
git -C "$repo" bundle create -q "$bundle" "$ref" HEAD

if ! ssh "$host" "mkdir '$remote_root'"; then
  echo "another check holds $remote_root on $host; if no run is in progress, remove it with: ssh $host rm -rf $remote_root" >&2
  exit 1
fi
locked=1
ssh "$host" "cat > '$remote_root/tree.bundle'" < "$bundle"

if [ -n "$only_bench" ]; then
  ssh "$host" "set -u
cd '$remote_root'
git clone -q --no-checkout tree.bundle src
cd src
git fetch -q ../tree.bundle '$ref'
git checkout -q --detach FETCH_HEAD
git submodule update --init -q
echo checkout=\$? > ../status.txt
export CARGO_TARGET_DIR='$remote_root/target' $bench_env
cargo bench $only_bench > ../bench.log 2>&1
echo bench=\$? >> ../status.txt
echo done >> ../status.txt" || true
  for f in status.txt bench.log; do
    ssh "$host" "cat '$remote_root/$f'" > "$out/$f" 2>/dev/null || true
  done
  cat "$out/status.txt"
  exit 0
fi

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
