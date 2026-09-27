#!/usr/bin/env bash
# CoordiNode pre-push validation script.
#
# Runs the same checks as CI to catch issues before pushing.
# Exit on first failure.
#
# Usage:
#   ./scripts/validate.sh          # full validation
#   ./scripts/validate.sh --quick  # skip release build and audit

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

step() {
    echo -e "\n${YELLOW}━━━ $1 ━━━${NC}"
}

ok() {
    echo -e "${GREEN}✓ $1${NC}"
}

fail() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

QUICK=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=true
fi

step "Step 1/7: Format check"
cargo fmt --all -- --check || fail "Formatting issues found. Run: cargo fmt --all"
ok "Format clean"

step "Step 2/7: Clippy (zero warnings)"
cargo clippy --workspace --all-targets --all-features -- -D warnings || fail "Clippy warnings found"
ok "Clippy clean"

step "Step 3/7: Unit and integration tests"
# The integration tests start the server binary, so it is built first.
cargo build -p coordinode-server --all-features || fail "Server build failed"
cargo nextest run --workspace --all-features || fail "Tests failed"
ok "Tests passed"

step "Step 4/7: Doc tests"
cargo test --doc --workspace --all-features || fail "Doc tests failed"
ok "Doc tests passed"

step "Step 5/7: API docs (zero warnings)"
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace --all-features || fail "API docs have warnings"
ok "API docs clean"

if [[ "$QUICK" == "false" ]]; then
    step "Step 6/7: Release build"
    cargo build --release --all-features || fail "Release build failed"
    ok "Release build succeeded"

    step "Step 7/7: Security audit"
    if command -v cargo-audit &> /dev/null; then
        cargo audit || fail "Security vulnerabilities found"
        ok "Audit clean"
    else
        echo -e "${YELLOW}⚠ cargo-audit not installed, skipping. Install: cargo install cargo-audit${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipping release build and audit (--quick mode)${NC}"
fi

echo -e "\n${GREEN}━━━ All checks passed ━━━${NC}"
