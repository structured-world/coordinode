# Embedded library releases

`.config/registry-packages.toml` names the complete embedded library dependency
closure, including optional replication. These packages and the server share a
version. The remaining workspace packages are not registry libraries.

The `Embedded registry packages` workflow tests the actual archives before a PR
lands. `Release PLZ` runs on a push to main and publishes only a merged release
PR after successful source CI on that exact commit. A manual retry on main
can select the merged release PR; the workflow resolves its immutable merge
commit, verifies that it belongs to main, and requires the same source-CI
evidence. Release-PR preparation runs separately on
clean main. Publication does not use `workflow_run`, which crates.io rejects for
Trusted Publishing.

`pr_config.py` derives an ignored configuration for release-PR calculation.
Before a library's first upload, it uses the existing coordinated Git baseline
with publishing disabled in that calculation only. A published library retains
its registry baseline and semantic-version checks. Registry failures stop the
calculation. Upload always uses the checked-in `.release-plz.toml`, never this
derived configuration; the server tag does not establish a library upload.

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

The workflow obtains a temporary credential through the official
`rust-lang/crates-io-auth-action` with `id-token: write` and passes it to
release-plz. The action revokes it when the job finishes, including on failure.
Authentication runs even when all versions were already uploaded by an earlier
attempt; reconciliation cannot silently skip checking the OIDC binding.
Provision the `crates-io` GitHub environment for main before publishing. Register
each crate's Trusted Publisher in crates.io for `structured-world/coordinode`,
workflow `coordinode-release.yml`, environment `crates-io`. CI consumes these
bindings; it never creates or changes them and has no bootstrap token path.

A previously unpublished crate requires an initial owner upload before its
publisher can be registered. Use an existing managed Cargo credential or a
short-lived crate-scoped token, outside CI. Use the exact qualified version and
prepared archive from the failed publication, verify the registry checksum,
and register its publisher in the browser. Revoke any temporary credential
created for setup. Observe crates.io's publishing limits and `Retry-After`.

To retry publication after main has advanced, dispatch `coordinode-release.yml`
on main with `release_pr` set to the original merged release PR number. This
uses the current publishing workflow with that approved source revision:

```sh
gh workflow run coordinode-release.yml --repo structured-world/coordinode \
  --ref main -f release_pr=PR_NUMBER
```

The workflow derives the commit from GitHub rather than accepting a branch or
an arbitrary SHA. It checks the existing source-suite result; do not rerun that
suite solely to retry registry authentication.

GitHub App authentication is separate: the standalone CLI reads `GIT_TOKEN`.
Git tags, GitHub releases and subsequent registry versions are created only by
release-plz.

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
