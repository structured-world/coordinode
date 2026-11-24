# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## v0.11.0 - 2026-11-24

### Added

- *(migrate)* give 0.10.1 records the version they can prove
- *(migrate)* bring a 0.10.1 data directory to this build
- *(core)* carry the acting principal in a log unit
- *(storage)* scope the change feed and fixed cuts to a namespace
- *(storage)* units name the namespace their data belongs to
- *(core)* carry the logical effect context in the unit frame
- *(schema)* give labels a catalog identity records store
- *(operation)* apply each client write once
- *(storage)* store index generations under local installations
- *(migrate)* add the node sizes report module
- *(migrate)* catalog record migrations, check and journal stats
- *(migrate)* offline tool to bring a data directory current

### Fixed

- *(ci)* public docs, bench lockfile, platform paths in a test
- *(migrate)* prove a version only by a write the apply took
- *(storage)* refuse a caller's condition on a lost revision
- *(migrate)* prove a version by its unit and refuse unstamped stores
- *(storage)* never copy a checkpoint over a file it holds
- *(migrate)* find every marked store, Raft snapshot captures too

### Performance

- *(query)* bound property delta chains with whole writes

### Testing

- *(migrate)* track the data directories of the receipt fixtures

## v0.10.1 - 2026-11-20

### Added

- *(core)* carry the logical effect context in the unit frame
- *(schema)* give labels a catalog identity records store
- *(operation)* apply each client write once
- *(storage)* store index generations under local installations
- *(migrate)* add the node sizes report module
- *(migrate)* catalog record migrations, check and journal stats
- *(migrate)* offline tool to bring a data directory current

### Fixed

- *(storage)* never copy a checkpoint over a file it holds
- *(migrate)* find every marked store, Raft snapshot captures too

### Performance

- *(query)* bound property delta chains with whole writes

## v0.10.0 - 2026-11-19

### Added

- *(core)* carry the logical effect context in the unit frame
- *(schema)* give labels a catalog identity records store
- *(operation)* apply each client write once
- *(storage)* store index generations under local installations
- *(migrate)* add the node sizes report module
- *(migrate)* catalog record migrations, check and journal stats
- *(migrate)* offline tool to bring a data directory current

### Fixed

- *(storage)* never copy a checkpoint over a file it holds
- *(migrate)* find every marked store, Raft snapshot captures too

### Performance

- *(query)* bound property delta chains with whole writes

## v0.9.0 - 2026-10-24

### Added

- *(operation)* apply each client write once
- *(storage)* store index generations under local installations
- *(migrate)* add the node sizes report module
- *(migrate)* catalog record migrations, check and journal stats
- *(migrate)* offline tool to bring a data directory current

### Fixed

- *(storage)* never copy a checkpoint over a file it holds
- *(migrate)* find every marked store, Raft snapshot captures too

### Performance

- *(query)* bound property delta chains with whole writes

## v0.8.0 - 2026-10-23

### Added

- *(operation)* apply each client write once
- *(storage)* store index generations under local installations
- *(migrate)* add the node sizes report module
- *(migrate)* catalog record migrations, check and journal stats
- *(migrate)* offline tool to bring a data directory current

### Fixed

- *(storage)* never copy a checkpoint over a file it holds
- *(migrate)* find every marked store, Raft snapshot captures too

### Performance

- *(query)* bound property delta chains with whole writes

## v0.7.3 - 2026-10-18

### Added

- *(operation)* apply each client write once
- *(storage)* store index generations under local installations
- *(migrate)* add the node sizes report module
- *(migrate)* catalog record migrations, check and journal stats
- *(migrate)* offline tool to bring a data directory current

### Fixed

- *(storage)* never copy a checkpoint over a file it holds
- *(migrate)* find every marked store, Raft snapshot captures too

### Performance

- *(query)* bound property delta chains with whole writes

## v0.7.2 - 2026-10-15

### Added

- *(storage)* store index generations under local installations
- *(migrate)* add the node sizes report module
- *(migrate)* catalog record migrations, check and journal stats
- *(migrate)* offline tool to bring a data directory current

### Fixed

- *(migrate)* find every marked store, Raft snapshot captures too

### Performance

- *(query)* bound property delta chains with whole writes

## v0.7.1 - 2026-10-13

### Added

- *(storage)* store index generations under local installations
- *(migrate)* add the node sizes report module
- *(migrate)* catalog record migrations, check and journal stats
- *(migrate)* offline tool to bring a data directory current

### Fixed

- *(migrate)* find every marked store, Raft snapshot captures too

### Performance

- *(query)* bound property delta chains with whole writes
