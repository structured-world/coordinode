---
applyTo: "**/*.rs"
---

# Rust Code Review Instructions

## Review Priority (HIGH → LOW)

Focus review effort on real bugs, not cosmetics. Stop after finding issues in higher tiers — do not pad reviews with low-priority nitpicks.

### Tier 1 — Logic Bugs and Correctness (MUST flag)
- Incorrect algorithm: wrong key format, wrong traversal order, off-by-one, TOCTOU
- Data corruption: wrong merge logic on posting lists, incorrect edge direction, dropped or duplicated entries
- MVCC/transaction correctness: snapshot visibility violations, conflict detection gaps, read-your-writes broken
- Missing validation: unchecked array index, unvalidated input from network/gRPC, unchecked segment metadata from disk
- Resource leaks: unclosed file handles, temporary files not cleaned up on error paths
- Concurrency: data races, lock ordering violations, shared mutable state without sync, missing synchronization on metadata updates
- Error swallowing: `let _ = fallible_call()` silently dropping errors that affect correctness or data integrity
- Integer overflow/truncation on sizes, lengths, node IDs, or block offsets
- Global mutable state: `lazy_static! { Mutex<HashMap> }` or equivalent — must use shared storage
- Incorrect merge semantics: tombstones not propagated, point deletes applied out of order, posting list merge violating sorted invariant

### Tier 2 — Safety and Crash Recovery (MUST flag)
- `unsafe` without `// SAFETY:` invariant explanation (should be zero — `#[forbid(unsafe_code)]`)
- `unwrap()`/`expect()` on I/O, network, or deserialization (must use `Result` propagation)
- Crash safety: write ordering that leaves data unrecoverable after power loss (e.g., updating index before data is fsynced)
- Partial write exposure: readers seeing data that is still being written
- fsync ordering: metadata (WAL, Raft log) must be durable before the operation it describes is considered committed
- Hardcoded secrets, credentials, or private URLs
- Information leaks: internal task or decision numbers, stage labels, paths to internal architecture documents, or local tool instruction files anywhere in code, comments, test names, error messages or docs
- Panic paths: any `unwrap()` on storage/network I/O (`#[deny(clippy::unwrap_used)]`)

### Tier 3 — API Design and Robustness (flag if clear improvement)
- Public API missing `#[must_use]` on builder-style methods returning `Self` or other non-`Result` types that callers might accidentally discard
- `pub` visibility where `pub(crate)` suffices
- Missing `Send + Sync` bounds on types used across threads
- `Clone` on large types (segment readers, block caches) where a reference would work
- Fallible operations returning `()` instead of `Result`
- Cross-crate dependency violations: storage internals leaking above `coordinode-storage`

### Tier 4 — Style (ONLY flag if misleading or confusing)
- Variable/function names that actively mislead about behavior
- Dead code (unused functions, unreachable branches)

## DO NOT Flag (Explicit Exclusions)

These are not actionable review findings. Do not raise them:

- **Comment wording vs code behavior**: If a comment describes intent and the code achieves it differently, the comment is documentation of intent — not a bug. Do not suggest rewording comments to match exact comparison operators.
- **Comment precision**: "returns X" when it technically returns `Result<X>` — the comment conveys meaning, not type signature.
- **Magic numbers with context**: A literal in `assert_eq!(status, 3, "expected SUSPENDED")` — the assertion message provides the context. Do not suggest importing a named constant when the value is used once in a test with an explanatory message.
- **Block sizes and configuration values**: Specific numeric values for cache sizes (e.g., `64 * 1024 * 1024`), HNSW parameters, or compression thresholds are domain constants, not magic numbers, when used in configuration or tests with surrounding context.
- **Minor naming preferences**: `opts` vs `options`, `adj_key` vs `adjacency_key`, `lvl` vs `level` — these are team style, not bugs.
- **Import ordering**: Import grouping or ordering style. Unused imports are NOT cosmetic — they cause `clippy -D warnings` failures and must be removed.
- **Test code style**: Tests prioritize readability and explicitness over DRY. Repeated setup code in tests is acceptable.
- **`#[allow(clippy::...)]` with justification comment**: Respect the author's suppression if explained. In test modules, `#[allow(clippy::unwrap_used)]` is standard. New code should prefer `#[expect(clippy::...)]`.
- **Temporary directory strategies in existing tests**: Existing tests using manual temp paths are not a finding. New tests should prefer `tempfile::tempdir()`.
- **Enum reverse mapping style**: Using `EnumType[value]` or match expressions for display — both are valid patterns.

## Scope Rules

- **Review ONLY code within the PR's diff.** Do not suggest inline fixes for unchanged lines.
- For issues **outside the diff**, suggest opening a separate issue instead of proposing code changes.
- **Read the PR description.** If it lists known limitations or deferred items, do not re-flag them.

## Rust-Specific Standards

- Prefer `#[expect(lint)]` over `#[allow(lint)]` — `#[expect]` warns when suppression becomes unnecessary
- `TryFrom`/`TryInto` for fallible conversions; `as` casts need `#[expect(clippy::cast_possible_truncation)]` with reason
- No `unwrap()` / `expect()` on I/O paths — use `?` propagation
- `expect()` is acceptable for programmer invariants (e.g., lock poisoning, `const` construction) with reason
- Code must pass `cargo clippy --all-features -- -D warnings`
- A comment states the rule or the reason itself; a comment that points at an internal document or names an internal task number instead is a leak (Tier 2)

## Testing Standards

- Test naming: `fn test_<what>_<condition>()` or `fn <descriptive_name>()`
- Use `tempfile::tempdir()` for test directories — ensures cleanup even on panic
- Integration tests that require infrastructure use `#[ignore = "reason"]`
- Prefer `assert_eq!` with message over bare `assert!` for better failure output
- Hardcoded values in tests are fine when accompanied by explanatory comments or assertion messages
- Gold standard tests (`compound_queries.rs`) must never be weakened — if they fail, fix the code, not the test

## CoordiNode-Specific Rules

- No C/C++ FFI in CE application code
- No `Box<dyn Any>` — use proper typing (enums or generics)
- Storage internals (LSM handles, column families) must not leak above `coordinode-storage`
- No crate may depend on `coordinode-server`
- Multi-instance safe: no in-process mutable state that would break with N replicas behind a load balancer
- SIMD: runtime detection only (`is_x86_feature_detected!`), never compile-time gate — same binary for all CPUs
