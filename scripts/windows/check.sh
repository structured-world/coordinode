#!/usr/bin/env bash
# Run the CI gate (clippy, server build, nextest, doc tests) for the current
# working tree on a Windows machine, uncommitted and untracked changes
# included, without touching the local index, branches or working tree.
#
# Usage: COORDINODE_WINDOWS_HOST=<ssh target> scripts/windows/check.sh
#
# With COORDINODE_CHECK_NEXTEST set, only nextest runs, with those arguments
# added; they are split at whitespace on the host, so a filter holds none
# (for example: COORDINODE_CHECK_NEXTEST='-E test(name) --stress-count 20').
# The multi-node cluster schemes are left out of the default run; they run
# with COORDINODE_CHECK_NEXTEST='-P cluster'.
#
# The host needs git, a Rust toolchain, cargo-nextest and protoc, and an
# OpenSSH server whose default shell is PowerShell. Logs, the status file and
# tree.txt (what was checked) land in target/windows-check/<UTC start>-<kind>/
# locally, one directory per run; nothing stays on the host. The exit
# code is 0 only when the run got to its end and every step passed.
set -euo pipefail

host="${COORDINODE_WINDOWS_HOST:?set COORDINODE_WINDOWS_HOST to the ssh target of the Windows machine}"

# Every connection fails rather than waits: a wrong or unreachable host, or a
# key the agent does not hold, would otherwise hang the run at its first call
# with nothing printed, and a link that drops mid-run would hang it at the
# step it was in.
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
ssh() { command ssh "${ssh_opts[@]}" "$@"; }
scp() { command scp "${ssh_opts[@]}" "$@"; }
# A directory of this run's own: files another account left in a shared one
# cannot be removed by this one and would fail the cleanup. Its name carries
# this machine and this script's PID, so a later run can tell which
# directories a killed run of this machine left behind.
origin="$(hostname -s | tr -c 'A-Za-z0-9' '_')"
run_dir="cn-check-$origin-$(date +%Y%m%d%H%M%S)-$$"
remote_root="C:\\$run_dir"
remote_scp="C:/$run_dir"
ref='refs/check/windows'

repo="$(git rev-parse --show-toplevel)"
# Each run keeps its reports in a directory of its own, named by when it
# started and what it ran, so a later run never overwrites the evidence of an
# earlier one.
if [ -n "${COORDINODE_CHECK_NEXTEST:-}" ]; then
  run_kind="nextest$(printf '%s' "$COORDINODE_CHECK_NEXTEST" | tr -c 'A-Za-z0-9._-' '_')"
else
  run_kind=full
fi
out="$repo/target/windows-check/$(date -u +%Y%m%dT%H%M%SZ)-$run_kind"
mkdir -p "$out"
# shellcheck source=../check-verdict.sh
. "$repo/scripts/check-verdict.sh"
bundle="$out/tree.bundle"

# Snapshot the working tree as a commit on a private ref, through a scratch
# index so the real one is left as it is.
index="$(mktemp)"
trap 'rm -f "$index" "$out"/*.bundle; git -C "$repo" update-ref -d "$ref" 2>/dev/null || true' EXIT
# The git dir, not `$repo/.git`: in a worktree `.git` is a file.
cp "$(git -C "$repo" rev-parse --absolute-git-dir)/index" "$index"
GIT_INDEX_FILE="$index" git -C "$repo" add -A
tree="$(GIT_INDEX_FILE="$index" git -C "$repo" write-tree)"
commit="$(git -C "$repo" commit-tree "$tree" -p HEAD -m 'windows check snapshot')"
git -C "$repo" update-ref "$ref" "$commit"
git -C "$repo" bundle create -q "$bundle" "$ref" HEAD
# What this run checked, beside its results: the tree it uploaded, the commit
# that tree sits on, and every submodule's commit.
{
  echo "tree=$tree"
  echo "head=$(git -C "$repo" rev-parse HEAD)"
  git -C "$repo" submodule status --recursive | sed 's/^/submodule=/'
} > "$out/tree.txt"

# A submodule's commit may exist only here, committed and not yet pushed, so
# each submodule travels as a bundle of its checked-out history
# (sub-<name>.bundle) and check.ps1 takes it from there rather than from the
# submodule's upstream.
while read -r key path; do
  name="${key#submodule.}"
  name="${name%.path}"
  # An uninitialized submodule (a fresh worktree) is an empty directory, and
  # git there would answer for the enclosing repository instead.
  if [ "$(git -C "$repo/$path" rev-parse --show-toplevel 2>/dev/null)" != "$(cd "$repo/$path" && pwd -P)" ]; then
    echo "submodule $path is not checked out; run: git -C '$repo' submodule update --init" >&2
    exit 1
  fi
  git -C "$repo/$path" bundle create -q "$out/sub-$name.bundle" HEAD
done < <(git -C "$repo" config -f .gitmodules --get-regexp '^submodule\..*\.path$' || true)

# A run killed by a signal skips the cleanup at the end and leaves its
# directory, with its build, on the host: remove those whose run from this
# machine is gone.
for stale in $(ssh "$host" "Get-ChildItem C:\\ -Directory -Filter 'cn-check-$origin-*' | ForEach-Object Name" | tr -d '\r'); do
  pid="${stale##*-}"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "removing $stale, left on $host by a run that is gone" >&2
    ssh "$host" "Remove-Item -Recurse -Force 'C:\\$stale' -ErrorAction SilentlyContinue"
  fi
done
ssh "$host" "New-Item -ItemType Directory -Force -Path '$remote_root' | Out-Null"
scp -q "$bundle" "$host:$remote_scp/tree.bundle"
for sub in "$out"/sub-*.bundle; do
  [ -e "$sub" ] || continue
  scp -q "$sub" "$host:$remote_scp/$(basename "$sub")"
done
scp -q "$repo/scripts/windows/check.ps1" "$host:$remote_scp/check.ps1"

# A failing step is reported through status.txt; the logs are fetched and
# the host cleaned either way.
# An empty argument does not survive `powershell -File` over ssh, so the
# option is passed only when set.
nextest_arg=''
if [ -n "${COORDINODE_CHECK_NEXTEST:-}" ]; then
  nextest_arg=" -Nextest '$COORDINODE_CHECK_NEXTEST'"
fi

# check.ps1 keeps the machine awake while it runs, and the machine sleeps
# the moment that hold goes. So the run prints CHECK-DONE and holds on until
# the file `fetched` appears beside its logs: the logs are fetched in
# between, over connections of their own, while the machine is still held
# awake.
pipes="$(mktemp -d)"
mkfifo "$pipes/out"
ssh "$host" "powershell -NoProfile -ExecutionPolicy Bypass -File '$remote_root\\check.ps1' -Root '$remote_root' -Bundle '$remote_root\\tree.bundle' -Ref '$ref'$nextest_arg" \
  </dev/null >"$pipes/out" &
remote=$!
exec 4<"$pipes/out"
while IFS= read -r line <&4; do
  case "$line" in
    CHECK-DONE*) break ;;
    *) printf '%s\n' "$line" >&2 ;;
  esac
done
# A nextest-only run writes no other logs.
logs='status.txt clippy.log build.log test.log doctest.log'
if [ -n "${COORDINODE_CHECK_NEXTEST:-}" ]; then
  logs='status.txt test.log'
fi
# The run takes long enough for a local `cargo clean` to remove this
# directory meanwhile; recreate it so the results are not lost.
mkdir -p "$out"
for f in $logs; do
  scp -q "$host:$remote_scp/$f" "$out/$f" || true
done
ssh "$host" "New-Item -ItemType File -Force -Path '$remote_root\\fetched' | Out-Null" || true
wait "$remote" || true
exec 4<&-
rm -rf "$pipes"
ssh "$host" "Remove-Item -Recurse -Force '$remote_root' -ErrorAction SilentlyContinue"

check_verdict "$out/status.txt"
