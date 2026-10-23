#!/usr/bin/env bash
set -euo pipefail

# The musl build rejects valid changelog templates on Linux. Use the verified
# GNU build on the Ubuntu publication runner, preserving parser stack checks.
archive=release-plz-x86_64-unknown-linux-gnu.tar.gz
destination=$(mktemp -d "${RUNNER_TEMP:?}/coordinode-release-plz.XXXXXX")
trap 'rm -f "$destination/$archive"' EXIT
curl --fail --silent --show-error --location \
  "https://github.com/release-plz/release-plz/releases/download/release-plz-v0.3.174/$archive" \
  --output "$destination/$archive"
printf '%s  %s\n' \
  411433a5e3bc833140dcab89f4ef482d6390298f72c6c15a958aef4e98003438 \
  "$destination/$archive" | sha256sum --check
tar -xzf "$destination/$archive" -C "$destination"
"$destination/release-plz" --version
printf '%s\n' "$destination" >> "${GITHUB_PATH:?}"
