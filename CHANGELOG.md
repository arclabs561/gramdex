# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-06-27

### Changed

- `store::UpdatableIndex` now caches each segment's `GramDex` by the segment's
  stable `Arc` identity (via segstore 0.2), so a mutation rebuilds only the new
  or changed segments instead of the whole corpus on the next query.
- Requires `segstore` 0.2 (only affects the optional `store` feature; the on-disk
  store format changed, so a `store` index written by 0.2.x is not read by 0.3.0).

## [0.2.1] - 2026-06-26

### Fixed

- `store::UpdatableIndex` caches the per-segment `GramDex` indexes and rebuilds
  them only on mutation (add/delete/compact), instead of rebuilding every segment
  on every query.

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
