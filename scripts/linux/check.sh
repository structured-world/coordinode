#!/usr/bin/env bash
# Run the CI gate (clippy, server build, nextest, doc tests) for the current
# working tree on a Linux machine, uncommitted and untracked changes included,
# without touching the local index, branches or working tree.
#
# Usage: COORDINODE_LINUX_HOST=<ssh target> scripts/linux/check.sh
#
# With COORDINODE_CHECK_NEXTEST set, only nextest runs, with those arguments
# added (a filter and a stress count to chase a flaky test, for example:
# COORDINODE_CHECK_NEXTEST="-E 'test(name)' --stress-count 20"; the host's
# shell parses the arguments again, so a filter keeps its own quotes).
# The multi-node cluster schemes are left out of the default run; they run
# with COORDINODE_CHECK_NEXTEST='-P cluster'.
#
# With COORDINODE_CHECK_BENCH set, only that `cargo bench` runs, with those
# arguments (a task's bounded measurement, for example:
# COORDINODE_CHECK_BENCH='-p coordinode-vector --bench hnsw_publication');
# its output lands in bench.log.
#
# With COORDINODE_CHECK_PROFILE set (the same `cargo bench` arguments), the
# bench runs with line tables and `perf` samples it once its output prints a
# line matching COORDINODE_CHECK_PROFILE_AFTER (an extended regex; from the
# start when unset), for COORDINODE_CHECK_PROFILE_SECONDS (30 by default).
# The bench output lands in bench.log, the hottest call paths in profile.txt
# and, when the host has inferno, a flame graph in profile.svg. The host
# needs perf.
#
# The host needs git, the pinned Rust toolchain, cargo-nextest and protoc
# (scripts/linux/provision.sh installs them); the run stops before uploading
# anything when one is missing. Logs and the status file land in
# target/linux-check/<host>/ locally; the run's directory on the host, build output
# included, is removed afterwards. The exit code is 0 only when the run got to
# its end and every step in the status file passed.
set -euo pipefail

host="${COORDINODE_LINUX_HOST:?set COORDINODE_LINUX_HOST to the ssh target of the Linux machine}"
# One fixed directory per host, created atomically as the run's lock: the
# compilation cache keys on absolute paths, so a path that differs per run
# would never hit and every run would rebuild the workspace from nothing. A
# second run on the same host waits for no one; it stops and says why.
remote_root="/var/tmp/cn-check"
locked=0
only_nextest="${COORDINODE_CHECK_NEXTEST:-}"
only_bench="${COORDINODE_CHECK_BENCH:-}"
only_profile="${COORDINODE_CHECK_PROFILE:-}"
profile_after="${COORDINODE_CHECK_PROFILE_AFTER:-}"
profile_seconds="${COORDINODE_CHECK_PROFILE_SECONDS:-30}"
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

# One local directory and one snapshot ref per host, so runs on two hosts at
# once (a gate on one, a measurement on another) keep their logs apart.
host_dir="$(printf '%s' "$host" | tr -c 'A-Za-z0-9._-' '_')"
out="$repo/target/linux-check/$host_dir"
ref="refs/check/linux/$host_dir"
mkdir -p "$out"
# shellcheck source=../check-verdict.sh
. "$repo/scripts/check-verdict.sh"
# A log this run did not write must not be read as its result.
rm -f "$out"/status.txt "$out"/*.log
bundle="$out/tree.bundle"

# Snapshot the working tree as a commit on a private ref, through a scratch
# index so the real one is left as it is.
index="$(mktemp)"
cleanup() {
  rm -f "$index"
  git -C "$repo" update-ref -d "$ref" 2>/dev/null || true
  # Only the run that took the lock removes the directory. A lookup or
  # connection failure at this point would leave the lock behind and refuse
  # every later run, so the removal is tried a few times.
  if [ "$locked" = 1 ]; then
    for attempt in 1 2 3 4 5; do
      ssh "$host" "rm -rf '$remote_root'" && break
      echo "cleanup attempt $attempt on $host failed; retrying" >&2
      sleep 10
    done
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

# A submodule's commit may exist only here, committed and not yet pushed, so
# each submodule travels as a bundle of its checked-out history and the host
# takes it from there rather than from the submodule's upstream.
rm -f "$out"/sub-*.bundle
sub_names=''
sub_urls=''
while read -r key path; do
  name="${key#submodule.}"
  name="${name%.path}"
  git -C "$repo/$path" bundle create -q "$out/sub-$name.bundle" HEAD
  sub_names="$sub_names $name"
  sub_urls="${sub_urls}git config submodule.$name.url '$remote_root/sub-$name.bundle'
"
done < <(git -C "$repo" config -f .gitmodules --get-regexp '^submodule\..*\.path$' || true)
checkout="git clone -q --no-checkout tree.bundle src
cd src
git fetch -q ../tree.bundle '$ref'
git checkout -q --detach FETCH_HEAD
git submodule init -q
${sub_urls}git -c protocol.file.allow=always submodule update -q
echo checkout=\$? > ../status.txt"

if ! ssh "$host" "mkdir '$remote_root'"; then
  echo "another check holds $remote_root on $host; if no run is in progress, remove it with: ssh $host rm -rf $remote_root" >&2
  exit 1
fi
locked=1
ssh "$host" "cat > '$remote_root/tree.bundle'" < "$bundle"
for name in $sub_names; do
  ssh "$host" "cat > '$remote_root/sub-$name.bundle'" < "$out/sub-$name.bundle"
done

if [ -n "$only_bench" ]; then
  ssh "$host" "set -u
cd '$remote_root'
$checkout
export CARGO_TARGET_DIR='$remote_root/target' $bench_env
# Build first and let the compilation cache finish writing before the timed
# run: its background uploads otherwise share the CPU with the measurement.
cargo bench $only_bench --no-run > ../bench-build.log 2>&1
echo build=\$? >> ../status.txt
if command -v sccache >/dev/null 2>&1; then sccache --stop-server >/dev/null 2>&1; fi
cargo bench $only_bench > ../bench.log 2>&1
echo bench=\$? >> ../status.txt
echo done >> ../status.txt" || true
  for f in status.txt bench-build.log bench.log; do
    ssh "$host" "cat '$remote_root/$f'" > "$out/$f" 2>/dev/null || true
  done
  check_verdict "$out/status.txt"
  exit
fi

if [ -n "$only_profile" ]; then
  ssh "$host" "set -u
cd '$remote_root'
$checkout
if ! command -v perf >/dev/null 2>&1; then
  echo 'perf is missing on the host' > ../profile.txt
  echo perf=1 >> ../status.txt
  echo done >> ../status.txt
  exit 0
fi
export CARGO_TARGET_DIR='$remote_root/target' CARGO_PROFILE_BENCH_DEBUG=line-tables-only CARGO_PROFILE_BENCH_STRIP=none $bench_env
cargo bench $only_profile --no-run > ../bench-build.log 2>&1
echo build=\$? >> ../status.txt
bin=\$(sed -n 's/.*Executable .*(\(.*\))\$/\1/p' ../bench-build.log | tail -n 1)
if command -v sccache >/dev/null 2>&1; then sccache --stop-server >/dev/null 2>&1; fi
\"\$bin\" --bench > ../bench.log 2>&1 &
pid=\$!
if [ -n '$profile_after' ]; then
  until grep -Eq '$profile_after' ../bench.log || ! kill -0 \$pid 2>/dev/null; do sleep 1; done
fi
perf record -F 499 --call-graph dwarf,16384 -p \$pid -o ../perf.data -- sleep '$profile_seconds' > ../perf.log 2>&1
echo perf=\$? >> ../status.txt
kill \$pid 2>/dev/null
wait \$pid 2>/dev/null
{
  echo '== self time =='
  perf report -i ../perf.data --stdio --no-children -g none --percent-limit 0.5
  echo '== with callees =='
  perf report -i ../perf.data --stdio --children -g none --percent-limit 2
  echo '== call paths into the hottest functions =='
  perf report -i ../perf.data --stdio --no-children -g caller,2,callee,function,percent --percent-limit 5
} > ../profile.txt 2>> ../perf.log
if command -v inferno-collapse-perf >/dev/null 2>&1; then
  perf script -i ../perf.data 2>> ../perf.log | inferno-collapse-perf | inferno-flamegraph > ../profile.svg
fi
echo done >> ../status.txt" || true
  for f in status.txt bench-build.log bench.log perf.log profile.txt profile.svg; do
    ssh "$host" "cat '$remote_root/$f'" > "$out/$f" 2>/dev/null || true
  done
  check_verdict "$out/status.txt"
  exit
fi

if [ -n "$only_nextest" ]; then
  ssh "$host" "set -u
cd '$remote_root'
$checkout
export RUSTFLAGS='-D warnings' COORDINODE_TEST_RAFT_GENEROUS_TIMEOUTS=1 CARGO_TARGET_DIR='$remote_root/target'
cargo nextest run --all-features --workspace --no-fail-fast --failure-output final $only_nextest > ../test.log 2>&1
code=\$?
# A stress run (--stress-count) exits 0 when some of its iterations failed
# (cargo-nextest 0.9.146); its summary line still counts them.
if [ \$code = 0 ] && grep -Eq '^ *Summary .* [1-9][0-9]* failed' ../test.log; then code=1; fi
echo nextest=\$code >> ../status.txt
echo done >> ../status.txt" || true
  for f in status.txt test.log; do
    ssh "$host" "cat '$remote_root/$f'" > "$out/$f" 2>/dev/null || true
  done
  check_verdict "$out/status.txt"
  exit
fi

# Each step's exit code goes to status.txt; a failing step does not stop the
# ones after it, and the logs are fetched either way.
ssh "$host" "set -u
cd '$remote_root'
$checkout
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

check_verdict "$out/status.txt"
