# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `store::UpdatableIndex` now keys its in-memory per-segment `GramDex` cache by
  segstore's stable segment ids instead of `Arc` pointers, and prunes stale cache
  entries when compaction/reclaim changes the segment set.
- `store::UpdatableIndex::candidates` now builds query k-grams once per search
  instead of once per sealed segment, reducing warm multi-segment query latency.
- Store writer searches now build the temporary writer-buffer index from the
  buffer slice instead of cloning buffered strings first.
- The `store` feature now requires `segstore = "0.4.1"` and `postcard`. This
  remains fully optional; default builds do not depend on the storage stack.

### Added

- Added a `store_reopen_diagnostics` example that measures the
  `store::SnapshotIndex` sidecar-first reader path versus rebuilding missing
  per-segment `GramDex` sidecars from source segments.
- Added `store::UpdatableIndex::plan_candidates`,
  `store::UpdatableIndex::candidates_bounded`,
  `store::SnapshotIndex::plan_candidates`, and
  `store::SnapshotIndex::candidates_bounded`, bringing the durable store path
  in line with the base `GramDex` broad-query bailout planner.
- Added `store::SnapshotIndex`, a read-only checkpoint view that opens
  segstore's manifest and queries persisted per-segment `GramDex` sidecars
  before falling back to one source segment decode on a sidecar miss.
- `examples/updatable_store` demonstrates the optional `store::UpdatableIndex`
  path: checkpoint, delete, reopen, `candidates_min_shared`, and caller-side
  verification of the returned candidate ids.
- Added `store::UpdatableIndex::candidates_min_shared`, exposing shared-gram
  candidate pruning through the durable store wrapper.
- The `store` feature now persists per-segment `GramDex` sidecars containing
  the built k-gram index plus live id set. Reopening an unchanged segment can
  load the sidecar instead of rebuilding its index; stale recipes, corrupt
  payloads, and stale live-id sets are rejected and rebuilt.
- Store benchmarks now measure cold sidecar loading separately from cold
  rebuilding when sidecars are missing or stale. On the synthetic 20k-doc
  benchmark (`FLUSH=2_000`), cold sidecar load measured ~24.9 ms versus ~90.7
  ms for rebuilding missing sidecars.

## [0.3.4] - 2026-06-28

### Added

- `store::UpdatableIndex::extend(docs)`: bulk ingest that syncs the write-ahead log
  once per batch instead of once per document. On a real filesystem this is ~4.7x
  faster than a loop of `add` for a corpus load (bench `ingest_fs`: 7.3ms vs 1.6ms
  for 4000 docs).

### Changed

- The `store` feature now requires `segstore = "0.3"`. The internal
  `merge_segments` impl takes `&[&Segment]` (segstore 0.3's by-reference signature,
  which drops a per-compaction clone). No public API change beyond `extend`.

## [0.3.3] - 2026-06-27

### Changed

- A `delete` now invalidates only the cached index of the segment that holds the
  id, not the whole cache, so one delete no longer forces every segment to
  rebuild on the next query.

## [0.3.2] - 2026-06-27

### Added

- `store::UpdatableIndex::compact_tiers()`: one round of size-tiered compaction
  (merge similarly-sized segments), keeping segment count bounded without a full
  `compact()`.

## [0.3.1] - 2026-06-27

### Added

- `store::UpdatableIndex::reclaim(min_live_ratio)` and `space_amplification()`
  (via the new `Store::live_len`): cheap tombstone reclamation, merging only the
  delete-heavy segments instead of a full compaction.

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
