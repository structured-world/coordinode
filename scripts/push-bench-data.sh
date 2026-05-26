#!/usr/bin/env bash
# Force-push the accumulated `bench-results/` tree to the `bench-data`
# orphan branch.  Mirrors the inline logic in `.github/workflows/bench.yml`
# so manual runs and CI runs end up with the same branch state.
#
# Workflow:
#   1. Snapshot the existing bench-data tree into a temp dir.
#   2. Overlay any newly-produced files from the local `bench-results/`.
#   3. Re-create bench-data as a fresh orphan branch with the merged tree.
#   4. Force-push.  History is intentionally throwaway on this branch.
#
# Usage:
#   scripts/push-bench-data.sh
#
# Env-var overrides:
#   COMMIT_MSG    — commit message (default: derived from git HEAD)
#   GIT_USER_NAME  — committer name  (default: coordinode-bench-bot)
#   GIT_USER_EMAIL — committer email (default: bench-bot@users.noreply.github.com)

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# `git status --porcelain bench-results/` was unreliable on the CI
# pages job: `actions/download-artifact@v4` extracts the
# bench-aggregated artifact into `bench-results/` AFTER the checkout
# step, and on the runner `git status` then sometimes reports an
# empty list even though new files are demonstrably present on disk.
# Suspect cause is checkout's temp-HOME / safe.directory dance that
# leaves the index out of sync with the post-extraction working tree.
#
# Switch to a deterministic filesystem ↔ HEAD diff: enumerate every
# file under `bench-results/`, subtract every file HEAD has tracked
# under the same prefix; anything left over is genuinely new. If
# that set is empty we have nothing to push.
NEW_FILES_COUNT=$(
  comm -23 \
    <(find bench-results -type f 2>/dev/null | sort -u) \
    <(git ls-tree -r HEAD --name-only -- bench-results 2>/dev/null | sort -u) \
    | wc -l | tr -d ' '
)
if [[ "$NEW_FILES_COUNT" == "0" ]]; then
  echo "no new bench-results files present in working tree — nothing to push"
  exit 0
fi
echo "$NEW_FILES_COUNT new bench-results file(s) detected — proceeding"

SNAPSHOT=$(mktemp -d)
trap 'rm -rf "$SNAPSHOT"' EXIT

if git ls-remote --exit-code --heads origin bench-data > /dev/null 2>&1; then
  git fetch origin bench-data:bench-data || true
  git archive bench-data -- bench-results/ 2>/dev/null \
    | tar -x -C "$SNAPSHOT" 2>/dev/null \
    || echo "::notice::bench-data archive empty — starting fresh"
fi

# Overlay current working-tree bench-results on top of the snapshot.
mkdir -p "$SNAPSHOT/bench-results"
cp -R bench-results/. "$SNAPSHOT/bench-results/" 2>/dev/null || true

GIT_USER_NAME="${GIT_USER_NAME:-coordinode-bench-bot}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-bench-bot@users.noreply.github.com}"
COMMIT_MSG="${COMMIT_MSG:-chore(bench): results @ $(git rev-parse HEAD)}"

git -c user.name="$GIT_USER_NAME" -c user.email="$GIT_USER_EMAIL" \
    branch -D bench-data 2>/dev/null || true

# Stash working-tree bench-results so the orphan-branch reset doesn't lose them.
WORK_SNAP=$(mktemp -d)
cp -R bench-results "$WORK_SNAP/" 2>/dev/null || true
trap 'rm -rf "$SNAPSHOT" "$WORK_SNAP"' EXIT

git checkout --orphan bench-data
git rm -rf . 2>/dev/null || true
mkdir -p bench-results
cp -R "$SNAPSHOT/bench-results/." bench-results/

git -c user.name="$GIT_USER_NAME" -c user.email="$GIT_USER_EMAIL" \
    add bench-results/
git -c user.name="$GIT_USER_NAME" -c user.email="$GIT_USER_EMAIL" \
    commit -m "$COMMIT_MSG"
git push origin bench-data --force

# Return to main and restore the local working tree state.
git checkout main
cp -R "$WORK_SNAP/bench-results" . 2>/dev/null || true

echo "pushed bench-data with $(find bench-results -type f -name '*.json' | wc -l | tr -d ' ') JSON files"
