# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-26

### Added
- Optional `store` feature: `store::UpdatableIndex`, an updatable, durable
  character k-gram fuzzy-match index backed by
  [`segstore`](https://crates.io/crates/segstore) (write-ahead log, checkpoint,
  compaction, crash recovery). The gram size `k` is chosen at `open`. Opt-in; the
  default build does not depend on segstore.
- Added a candidate verification and ranking example.

## [0.1.1] - 2026-03-08

### Added
- Initial release.
- Doc-tests on six key public functions.
- SECURITY.md.
- Crates.io metadata and a quickstart.
