# gramdex examples

These examples are small demonstrations on synthetic strings. Their output,
including the timings printed by `store_reopen_diagnostics`, is not benchmark
evidence; use `cargo bench` for measurement.

## Where to start

| I want to... | Run |
|---|---|
| Find candidates by shared trigrams, then verify and rank them by Jaccard | `candidate_verify_rank` |
| Add, checkpoint, delete, and reopen a durable k-gram index | `updatable_store` |
| Compare first-query time after reopening with and without persisted sidecars | `store_reopen_diagnostics` |

## Default features

| Example | What it shows |
|---|---|
| `candidate_verify_rank` | Builds a `GramDex` over four strings, takes the union of trigram candidates for one query, keeps those with trigram Jaccard >= 0.10, and prints them ranked by Jaccard then shared-gram count |

```sh
cargo run --release --example candidate_verify_rank
```

## Feature-gated examples

| Feature | Example | What it shows |
|---|---|---|
| `store` | `updatable_store` | `store::UpdatableIndex` in memory: indexes four strings, checks candidates and `min_shared` candidates, deletes one document, reopens from the same directory, and checks the deletion survived |
| `store` | `store_reopen_diagnostics` | Builds a 1,000-document checkpointed store on disk twice, deletes the segment sidecars in one copy, and prints the first `SnapshotIndex` query time for each path; asserts both paths return the same candidates |

```sh
cargo run --release --features store --example updatable_store
cargo run --release --features store --example store_reopen_diagnostics
```
