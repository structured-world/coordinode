# Changelog

All notable changes to this crate are documented in this file.

## v0.6.0 - 2026-09-19

### Fixed

- *(search)* index compound words in chinese text

## 0.5.2 - 2026-08-30

#### Added

- *(wire)* let the TLS crypto provider be chosen at startup

#### Fixed

- *(ci)* teach the changelog splitter the current heading layout

---

## 0.5.1 - 2026-08-29

#### Added

- *(wire)* encrypt outbound inter-node gRPC with client TLS
- *(wire)* TLS/mTLS config foundation with pure-Rust crypto provider

#### Fixed

- *(wire)* default to zstd level 3 to avoid the Fast-strategy panic path

#### Refactored

- *(wire)* migrate PEM parsing off unmaintained rustls-pemfile
- extract shared wire codec, compress segment transfer too

---

## 0.5.0 - 2026-06-27

#### Added

- *(wire)* encrypt outbound inter-node gRPC with client TLS
- *(wire)* TLS/mTLS config foundation with pure-Rust crypto provider

#### Fixed

- *(wire)* default to zstd level 3 to avoid the Fast-strategy panic path

#### Refactored

- *(wire)* migrate PEM parsing off unmaintained rustls-pemfile
- extract shared wire codec, compress segment transfer too
