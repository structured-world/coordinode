# Changelog

All notable changes to this crate are documented in this file.

## v0.7.3 - 2026-10-18

### Performance

- *(storage)* defer the log sync of volatile writes, not the write

## v0.7.1 - 2026-10-13

### Added

- *(search)* run encrypted index searches within the statement budget
- *(search)* run text index searches against the statement budget

### Fixed

- *(search)* keep text statistics to what readers see after a failed write
- *(search)* parse the documented text query language
- *(search)* highlight the words a fuzzy search matched
- *(search)* run a text prefix as a term range
- *(search)* refuse writes after a lost text-index writer
- *(search)* match every word a text prefix starts
- *(search)* carry no fictitious birth time in text documents
- *(query)* search temporal labels at their valid state

### Testing

- *(search)* keep final statistics through segment merge and reopen
- *(search)* cover refused and zero birth-time reads

## v0.7.0 - 2026-10-06

### Documentation

- drop internal ids from search, replicate, swarm, wire and timeseries

### Fixed

- *(search)* score full-text against the live documents only
- *(search)* answer unfolded writes exactly instead of waiting
- *(search)* maintain text indexes from committed writes

## v0.6.0 - 2026-09-23

### Fixed

- *(storage)* never let the watermark pass a snapshot just handed out
- *(search)* index compound words in chinese text

## 0.5.2 - 2026-08-30

#### Fixed

- *(ci)* teach the changelog splitter the current heading layout

---

## 0.5.1 - 2026-08-29

#### Added

- *(storage)* R156 + R157 - multi-endpoint storage placement
- *(search)* FTS MVCC snapshot filter via per-doc commit_ts + segment registry
- *(text-search)* implement TextService gRPC with fuzzy + language-aware search
- *(search)* external CJK dictionary loading from filesystem (G014)
- CoordiNode v0.1.0-alpha.1 - graph + vector + full-text engine

#### Fixed

- *(ci)* replace cargo test with nextest, fix CJK env var race

#### Refactored

- extract unit tests into sibling files (query, storage, vector, search)
- extract unit tests into sibling test files
- *(search)* thread Transaction through the SSE token index

#### Testing

- *(search)* direct unit tests for search_with_highlights_fuzzy and search_with_highlights_and_language
- *(text-search)* Ukrainian e2e + multi-property merge coverage

---

## 0.5.0 - 2026-06-27

#### Added

- *(storage)* R156 + R157 - multi-endpoint storage placement
- *(search)* FTS MVCC snapshot filter via per-doc commit_ts + segment registry
- *(text-search)* implement TextService gRPC with fuzzy + language-aware search
- *(search)* external CJK dictionary loading from filesystem (G014)
- CoordiNode v0.1.0-alpha.1 - graph + vector + full-text engine

#### Fixed

- *(ci)* replace cargo test with nextest, fix CJK env var race

#### Refactored

- extract unit tests into sibling files (query, storage, vector, search)
- extract unit tests into sibling test files
- *(search)* thread Transaction through the SSE token index

#### Testing

- *(search)* direct unit tests for search_with_highlights_fuzzy and search_with_highlights_and_language
- *(text-search)* Ukrainian e2e + multi-property merge coverage

---

## 0.4.1 - 2026-04-18

#### Added

- *(search)* FTS MVCC snapshot filter via per-doc commit_ts + segment registry

---

## 0.3.9 - 2026-04-13

#### Added

- *(text-search)* implement TextService gRPC with fuzzy + language-aware search

#### Testing

- *(search)* direct unit tests for search_with_highlights_fuzzy and search_with_highlights_and_language
- *(text-search)* Ukrainian e2e + multi-property merge coverage
