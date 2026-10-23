# Embedded library releases

`.config/registry-packages.toml` names the complete embedded library dependency
closure, including optional replication. These packages and the server share a
version. The remaining workspace packages are not registry libraries.

The `Embedded registry packages` workflow tests the actual archives before a PR
lands. `Release PLZ` publishes only a merged release PR after successful source
CI. It also supports a manual retry on main; that path requires successful source
CI on the same commit. Release-PR preparation runs separately on clean main.

## Preparation and qualification

Run `prepare.py` only in a disposable, clean checkout. It removes the workspace
feature-unification dependency, generates bindings from the pinned proto
submodule, and prepares manifests, build scripts and attribution files. Generated
bindings belong in archives, not in Git. The published build scripts copy those
bindings; they do not require protoc, a submodule or network generation.

`check_packages.py` inspects every archive and installs them into an independent
disk-backed Rust consumer. Cargo's local registry contains the unpublished
archives and their registry dependencies; there are no path patches. Both the
minimum toolchain and the source toolchain exercise the default and replication
feature configurations. The fixture checks transactions, schema publication,
uniqueness, CAS, rollback, reopen and reconciliation after a process exits without
closing the database. The replication-enabled fixture is not a multi-member
failure qualification, and process exit is not a power-loss simulation.

The release job runs the same qualification before upload. It selects the commit
release-plz will actually publish: an ancestor release-PR head is eligible only
when its complete Git tree matches the source-CI commit. A different tree stops
publication. The preparation report records both revisions.

The workflow pins its preparation tools in `.github/actions/prepare-registry`.
For local checks, use that same environment and run the Python unit tests before
`prepare.py` and `check_packages.py`. The latter accepts repeated
`--consumer-toolchain` arguments. Source builds and their full test suite are
separate gates; stripping source-only tests from an archive does not replace them.

## Publication credentials and recovery

For the first publication, provision the `crates-io` GitHub environment for main
and the temporary `CRATES_IO_BOOTSTRAP_TOKEN` secret. Restrict the token to the
listed package names, initial publication and Trusted Publisher registration.
The workflow publishes through release-plz, registers each crate's publisher for
`structured-world/coordinode`, `coordinode-release.yml`, environment `crates-io`,
and reads the binding back. Remove the bootstrap secret and revoke the token
after successful initial publication and publisher verification.

Later publications use release-plz's native OIDC flow with `id-token: write`.
GitHub App authentication is separate: the standalone CLI reads `GIT_TOKEN`.
Do not create tags, registry versions or GitHub releases manually.

An interrupted publication can be retried only with the same qualified archives.
`registry.py preflight` rejects an existing tag belonging to another revision,
an immutable registry version with another checksum, or a yanked version.
Successful uploads are not repeated; missing packages remain required. An old
server-only version is not a valid source revision for new library contents:
prepare a new coordinated release PR instead.

After upload, `registry.py verify` checks every published checksum and the
consumer runs again with a fresh Cargo home against crates.io itself. The
workflow retains its preparation report and consumer dependency graphs as
artifacts, including failure evidence. A registered publisher or a no-op OIDC
run alone does not prove a successful OIDC publication; verify an actual release
through that path before declaring the publication flow qualified.
