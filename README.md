# gramdex

[![crates.io](https://img.shields.io/crates/v/gramdex.svg)](https://crates.io/crates/gramdex)
[![Documentation](https://docs.rs/gramdex/badge.svg)](https://docs.rs/gramdex)
[![CI](https://github.com/arclabs561/gramdex/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/gramdex/actions/workflows/ci.yml)

K-gram indexing for approximate string matching.

## Quickstart

```toml
[dependencies]
gramdex = "0.3.4"
```

```rust
use gramdex::{GramDex, trigram_jaccard};

let mut ix = GramDex::new();
ix.add_document_trigrams(1, "hello");
ix.add_document_trigrams(2, "yellow");

let candidates = ix.candidates_union_trigrams("mellow");
let mut verified: Vec<u32> = candidates
    .into_iter()
    .filter(|&doc| match doc {
        1 => trigram_jaccard("mellow", "hello") >= 0.2,
        2 => trigram_jaccard("mellow", "yellow") >= 0.2,
        _ => false,
    })
    .collect();
verified.sort_unstable();
assert_eq!(verified, vec![2]);
```

## Best starting points

- **Gram generation**: `char_kgrams` / `char_trigrams`
- **Candidate index**: `GramDex` (union candidates, scored candidates, bailout planning)
- **Verification**: `trigram_jaccard`

## Example

```sh
cargo run --example candidate_verify_rank
```

Output:

```text
query: "nearest neighbor"
doc 3: shared=13, jaccard=0.394, text="nearest neighbour spelling variant"
doc 0: shared=14, jaccard=0.341, text="vector quantization for nearest neighbor search"
```

## Design notes

- This crate focuses on **candidate generation**; you bring your own verification policy.
- Offsets/spans are naturally expressed in **Unicode scalar values** (`char` count), not bytes.

## Updatable index (`store` feature)

`store::UpdatableIndex` wraps the index in a durable, segmented store
([`segstore`](https://crates.io/crates/segstore)): incremental add/delete, a
write-ahead log, checkpoint, compaction, and crash recovery, with the gram size
`k` chosen at `open`. Per-segment indexes are cached and persisted as sidecars,
so unchanged segments can load their built `GramDex` blocks after a restart
instead of rebuilding them. Opt-in; the default build does not depend on
segstore. `store::SnapshotIndex` opens the last checkpoint manifest and queries
sidecars first, so source text batches are read only when a sidecar is missing
or unusable. `candidates_min_shared`, `plan_candidates`, and
`candidates_bounded` expose the same shared-gram pruning and broad-query bailout
as the in-memory index for durable stores.

```sh
cargo run --features store --example updatable_store
```

For measurement, `cargo run --release --features store --example store_reopen_diagnostics`
prints the first snapshot-query cost with persisted `GramDex` sidecars present
versus after deleting those sidecars and forcing source-segment rebuilds.

```text
documents: 1000, flush threshold: 200, k-grams: 3
sidecars loaded path: 5
sidecars rebuild path before/after delete: 5/0
first snapshot query with sidecars: 3140 us
first snapshot query after deleting sidecars: 7045 us
matching candidates: 1
query doc present: true
```

```text
before delete:
  candidates: [1, 2, 3]
  min_shared=3: [2, 3]
after reopen:
  candidates: [1, 3]
  min_shared=3: [3]
  verified doc 3: jaccard=1.000, text="mellow"
  verified doc 1: jaccard=0.400, text="hello"
```

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

at your option.
