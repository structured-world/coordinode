# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## 0.5.1 - 2026-08-29

#### Added

- *(session)* ORDERED transactions with nonce reorder and commit-drain timeout
- *(session)* abort an interactive transaction on its first failed statement
- *(session)* real interactive transactions over the session stream
- *(session)* SHOW SESSIONS / SHOW TRANSACTIONS introspection
- *(session)* multiplexed bidi session protocol with server-side cursors

#### Fixed

- *(session)* run blocking cursor work off the async worker pool

#### Testing

- *(session)* cover ordered first-failure abort; document nonce contract
