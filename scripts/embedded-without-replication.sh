#!/usr/bin/env bash
# The embedded library built without its `replication` feature carries no
# consensus, no inter-node transport and no async runtime: none of them is in
# its dependency tree on any target. A disabled feature is not compiled, so
# checking the tree is checking the build.
#
# Exits non-zero, naming what got in and the path it took, when one of them
# reaches the default build.
set -u

absent=(openraft tonic hyper h2 tokio coordinode-raft coordinode-wire)
failed=0
for crate in "${absent[@]}"; do
  path="$(cargo tree -p coordinode-embed -e normal --target all -i "$crate" 2>/dev/null)"
  if [ -n "$path" ]; then
    echo "coordinode-embed without \`replication\` depends on $crate:" >&2
    echo "$path" >&2
    failed=1
  fi
done
exit "$failed"
