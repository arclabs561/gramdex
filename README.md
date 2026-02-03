# gramdex

`gramdex` provides small, dependency-light primitives for approximate string matching:

- Unicode-scalar (Rust `char`) \(k\)-gram generation
- A minimal grams → document-ids index (`GramDex`) for candidate generation
- An exact (verification) trigram Jaccard helper (`trigram_jaccard`)

## Quickstart

```toml
[dependencies]
gramdex = "0.1.0"
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

## Design notes

- This crate focuses on **candidate generation**; you bring your own verification policy.
- Offsets/spans are naturally expressed in **Unicode scalar values** (`char` count), not bytes.

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

at your option.
