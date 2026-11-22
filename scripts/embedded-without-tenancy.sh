#!/usr/bin/env bash
# The embedded library built without its `tenancy` feature compiles no grant
# catalog and no authorization step: nothing in its build turns on the
# storage engine's `tenancy` feature, on any target. A disabled feature is
# not compiled, so checking the resolved features is checking the build.
#
# Exits non-zero, naming the path that turned it on, when something does.
set -u

tree="$(cargo tree -p coordinode-embed --no-default-features -e features --target all -i coordinode-storage 2>/dev/null)"
if printf '%s\n' "$tree" | grep -q 'coordinode-storage feature "tenancy"'; then
  echo "coordinode-embed without \`tenancy\` builds coordinode-storage with \`tenancy\`:" >&2
  printf '%s\n' "$tree" >&2
  exit 1
fi
