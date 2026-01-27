## gramdex

`gramdex` provides small, dependency-light primitives for approximate string matching:

- Unicode-scalar (Rust `char`) \(k\)-gram generation
- A minimal grams → document-ids index (`GramDex`) for candidate generation
- An exact (verification) trigram Jaccard helper (`trigram_jaccard`)

### Design notes

- This crate focuses on **candidate generation**; you bring your own verification policy.
- Offsets/spans are naturally expressed in **Unicode scalar values** (`char` count), not bytes.

### License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

at your option.

