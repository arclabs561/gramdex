# gramdex

[![crates.io](https://img.shields.io/crates/v/gramdex.svg)](https://crates.io/crates/gramdex)
[![Documentation](https://docs.rs/gramdex/badge.svg)](https://docs.rs/gramdex)

K-gram indexing for approximate string matching.

Index queries return candidates, not final matches. Verify them with an exact
or application-specific similarity check before presenting results.

## Quickstart

```toml
[dependencies]
gramdex = "0.6"
```

The default build supports Rust 1.80. The optional `store` feature requires
Rust 1.83 with current dependencies.

```rust
use gramdex::{GramDex, trigram_jaccard};

let mut ix = GramDex::new();
ix.add_document_trigrams(1, "hello");
ix.add_document_trigrams(2, "yellow");

let candidates = ix.candidates_union_trigrams("mellow");
let mut verified: Vec<u32> = candidates
    .into_iter()
    .filter(|&doc| match doc {
        1 => trigram_jaccard("mellow", "hello") >= 0.5,
        2 => trigram_jaccard("mellow", "yellow") >= 0.5,
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

Grams are built from Unicode scalar values (`char`), not bytes.

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

## Updatable index (`store` feature)

`store::UpdatableIndex` stores the index in
[`segstore`](https://crates.io/crates/segstore), with incremental adds and deletes,
a write-ahead log, checkpoints, compaction, and crash recovery. Choose the gram
size `k` when opening the store. Segment indexes are cached and saved in sidecar
files when possible. Valid sidecars can be reused after a restart instead of
rebuilding the indexes. The default build does not depend on segstore.

`store::SnapshotIndex` opens the last checkpoint and queries sidecars first.
It reads the source text only when a sidecar is missing or unusable.
`candidates_min_shared`, `plan_candidates`, and
`candidates_bounded` expose the same shared-gram pruning and broad-query bailout
as the in-memory index for durable stores.

```sh
cargo run --features store --example updatable_store
```

Output:

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

For measurement, `cargo run --release --features store --example store_reopen_diagnostics`
reopens a checkpointed store and prints the duration of its first snapshot
query. It does this twice: once with the persisted `GramDex` sidecars present,
and once after deleting those sidecars, which forces each segment to be
rebuilt from its source text. Timings depend on the machine and filesystem and
are omitted below.

```text
documents: 1000, flush threshold: 200, k-grams: 3
sidecars loaded path: 5
sidecars rebuild path before/after delete: 5/0
matching candidates: 1
query doc present: true
```

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

at your option.
