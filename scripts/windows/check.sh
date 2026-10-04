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
# OpenSSH server whose default shell is PowerShell. Logs and the status file
# land in target/windows-check/ locally; nothing stays on the host. The exit
# code is 0 only when the run got to its end and every step passed.
set -euo pipefail

host="${COORDINODE_WINDOWS_HOST:?set COORDINODE_WINDOWS_HOST to the ssh target of the Windows machine}"
# A directory of this run's own: files another account left in a shared one
# cannot be removed by this one and would fail the cleanup.
run_dir="cn-check-$(date +%Y%m%d%H%M%S)-$$"
remote_root="C:\\$run_dir"
remote_scp="C:/$run_dir"
ref='refs/check/windows'

repo="$(git rev-parse --show-toplevel)"
out="$repo/target/windows-check"
mkdir -p "$out"
# shellcheck source=../check-verdict.sh
. "$repo/scripts/check-verdict.sh"
bundle="$out/tree.bundle"

# Snapshot the working tree as a commit on a private ref, through a scratch
# index so the real one is left as it is.
index="$(mktemp)"
trap 'rm -f "$index"; git -C "$repo" update-ref -d "$ref" 2>/dev/null || true' EXIT
# The git dir, not `$repo/.git`: in a worktree `.git` is a file.
cp "$(git -C "$repo" rev-parse --absolute-git-dir)/index" "$index"
GIT_INDEX_FILE="$index" git -C "$repo" add -A
tree="$(GIT_INDEX_FILE="$index" git -C "$repo" write-tree)"
commit="$(git -C "$repo" commit-tree "$tree" -p HEAD -m 'windows check snapshot')"
git -C "$repo" update-ref "$ref" "$commit"
git -C "$repo" bundle create -q "$bundle" "$ref" HEAD

# A submodule's commit may exist only here, committed and not yet pushed, so
# each submodule travels as a bundle of its checked-out history
# (sub-<name>.bundle) and check.ps1 takes it from there rather than from the
# submodule's upstream.
rm -f "$out"/sub-*.bundle
while read -r key path; do
  name="${key#submodule.}"
  name="${name%.path}"
  git -C "$repo/$path" bundle create -q "$out/sub-$name.bundle" HEAD
done < <(git -C "$repo" config -f .gitmodules --get-regexp '^submodule\..*\.path$' || true)

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
ssh "$host" "powershell -NoProfile -ExecutionPolicy Bypass -File '$remote_root\\check.ps1' -Root '$remote_root' -Bundle '$remote_root\\tree.bundle' -Ref '$ref'$nextest_arg" || true

# A log this run did not write must not be read as its result.
for f in status.txt clippy.log build.log test.log doctest.log; do
  rm -f "$out/$f"
done
for f in status.txt clippy.log build.log test.log doctest.log; do
  scp -q "$host:$remote_scp/$f" "$out/$f" || true
done
ssh "$host" "Remove-Item -Recurse -Force '$remote_root' -ErrorAction SilentlyContinue"

check_verdict "$out/status.txt"
