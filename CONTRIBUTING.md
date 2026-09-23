# Contributing to CoordiNode

We welcome contributions from everyone. This document explains how to get involved.

## Ways to Contribute

- **Bug reports** — found something broken? [Open an issue](https://github.com/structured-world/coordinode/issues/new?template=bug_report.yml)
- **Feature requests** — have an idea? [Suggest it](https://github.com/structured-world/coordinode/issues/new?template=feature_request.yml)
- **Code** — fix a bug or implement a feature (see below)
- **Documentation** — improve docs, fix typos, add examples

## Development Setup

```bash
# Clone
git clone https://github.com/structured-world/coordinode.git
cd coordinode

# Build
cargo build

# Run tests (the same selection CI runs, so local builds reuse its artefacts).
# The cluster tests time Raft elections; on a machine busy running the whole
# suite in parallel, the generous timeouts CI uses keep them from electing
# spuriously and timing out.
COORDINODE_TEST_RAFT_GENEROUS_TIMEOUTS=1 cargo nextest run --workspace --all-features
cargo test --doc --all-features

# Run with Clippy (must pass with zero warnings)
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Requires Rust 1.90+ (see `rust-toolchain.toml` for the pinned toolchain) and
[cargo-nextest](https://nexte.st).

**After adding, removing or changing a dependency**, regenerate the
workspace-hack crate and commit the result; CI fails when it is stale:

```bash
cargo install cargo-hakari --locked   # once
cargo hakari generate
cargo hakari manage-deps
```

`crates/coordinode-workspace-hack` pins the features of third-party
dependencies so that building one crate (`-p`) and building the workspace
compile the same artefacts instead of a second copy of each.

**Debugging.** The everyday `dev` profile keeps file and line information for
panics and backtraces but leaves out what a debugger needs. To step through
code, build with the `debugger` profile; it goes to `target/debugger` and does
not replace the everyday build:

```bash
cargo build --profile debugger
cargo nextest run --cargo-profile debugger -E 'test(name)'
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`feat/description` or `fix/description`)
3. Make your changes
4. Ensure all checks pass:
   - `cargo fmt --all -- --check`
   - `cargo clippy --workspace --all-targets --all-features -- -D warnings` (zero warnings)
   - `COORDINODE_TEST_RAFT_GENEROUS_TIMEOUTS=1 cargo nextest run --workspace --all-features` and `cargo test --doc --all-features` (all tests pass)
   - `cargo hakari generate --diff` and `cargo hakari manage-deps --dry-run` (workspace-hack up to date)
5. Write a clear commit message following [Conventional Commits](https://www.conventionalcommits.org/)
6. Open a pull request with a description of what changed and why

## Code Style

- **Rust** — follow `rustfmt` defaults and Clippy recommendations
- `pub(crate)` by default — explicit `pub` only for public API
- `Result<T, E>` everywhere — no `unwrap()` on I/O paths
- Tests for every function — happy path + error path at minimum

## Contributor License Agreement (CLA)

CoordiNode is dual-licensed. The Community Edition is AGPL-3.0-only, and the same code base is also distributed under a commercial licence as the Enterprise Edition. For that to stay possible, a contribution has to arrive with more than "the same licence as the project": the copyright holder needs the right to distribute it under both.

Before a first pull request can be merged, you sign the [Contributor License Agreement](CLA.md). Signing happens in the pull request itself: a bot posts the request, you reply with the sentence it asks for, and the signature is recorded in `signatures/` in this repository. It is a one-time step per GitHub account.

In short, the CLA says:

- You keep the copyright in your contribution.
- You grant the project's copyright holder (and any successor the copyright is assigned to) a perpetual, worldwide, non-exclusive, royalty-free, irrevocable licence to use, modify, distribute and sublicense your contribution under any terms, including AGPL-3.0-only and the commercial Enterprise Edition licence.
- You grant a patent licence covering your contribution to the same extent.
- You confirm you are entitled to make the grant: the work is yours, or your employer has authorised it.

And the project promises in return:

- Your contribution stays available under AGPL-3.0-only in the Community Edition. It is never withdrawn into a proprietary-only edition.
- You remain free to use, license and redistribute your own contribution however you like.

If your employer owns what you write, ask them to confirm they permit the contribution before you sign.

## Code of Conduct

Be respectful. We don't have a formal code of conduct document, but we expect professional behavior. Harassment, discrimination, and bad faith arguments are not tolerated.

## Questions?

Open a [question issue](https://github.com/structured-world/coordinode/issues/new?template=question.yml) or reach out at oss@sw.foundation.
