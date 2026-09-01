# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## 0.5.7 - 2026-09-01

#### Added

- *(session)* tell a client what its connection can do, and let it configure one

---

## 0.5.2 - 2026-08-30

#### Fixed

- *(ci)* teach the changelog splitter the current heading layout

---

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
