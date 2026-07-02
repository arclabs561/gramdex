//! Benchmarks for the `store` feature (segstore-backed updatable k-gram index).
//!
//! Run: `cargo bench --features store --bench store`. Without the feature the
//! harness is an empty no-op so the target still compiles. Measures build
//! throughput, warm query latency (per-segment index cached), cold restart
//! latency with persisted sidecars, and the cold rebuild cost when sidecars are
//! missing or stale.

#[cfg(not(feature = "store"))]
fn main() {}

#[cfg(feature = "store")]
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};

#[cfg(feature = "store")]
const N: usize = 20_000;
#[cfg(feature = "store")]
const FLUSH: usize = 2_000; // ~10 segments
#[cfg(feature = "store")]
const K: usize = 3;

#[cfg(feature = "store")]
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[cfg(feature = "store")]
fn text(state: &mut u64) -> String {
    (0..40)
        .map(|_| (b'a' + (xorshift(state) % 26) as u8) as char)
        .collect()
}

#[cfg(feature = "store")]
fn fresh_store(
    warm: bool,
    checkpoint: bool,
) -> (
    std::sync::Arc<dyn durability::Directory>,
    gramdex::store::UpdatableIndex,
    String,
) {
    use durability::MemoryDirectory;
    let mut s = 0x1234_5678_9abc_def0u64;
    let dir = MemoryDirectory::arc();
    let mut store = gramdex::store::UpdatableIndex::open(dir.clone(), FLUSH, K).unwrap();
    for i in 0..N {
        store.add(i as u32, text(&mut s)).unwrap();
    }
    if checkpoint {
        store.checkpoint().unwrap();
    }
    let q = text(&mut s);
    if warm {
        let _ = store.candidates(&q);
    }
    (dir, store, q)
}

#[cfg(feature = "store")]
fn benches(c: &mut Criterion) {
    let mut g = c.benchmark_group("store");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("build", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = fresh_store(false, true);
            },
            BatchSize::SmallInput,
        )
    });

    let (_, warm, q) = fresh_store(true, true);
    g.bench_function("search_warm", |b| b.iter(|| warm.candidates(&q)));

    g.bench_function("search_cold_load_sidecars", |b| {
        b.iter_batched(
            || {
                let (dir, _, q) = fresh_store(false, true);
                let store = gramdex::store::UpdatableIndex::open(dir, FLUSH, K).unwrap();
                (store, q)
            },
            |(store, q)| store.candidates(&q),
            BatchSize::SmallInput,
        )
    });

    g.bench_function("search_cold_rebuild_missing_sidecars", |b| {
        b.iter_batched(
            || {
                let (_, store, q) = fresh_store(false, false);
                (store, q)
            },
            |(store, q)| store.candidates(&q),
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

#[cfg(feature = "store")]
fn ingest_fs(c: &mut Criterion) {
    // The extend() win is invisible on MemoryDirectory (flush is free); on a real
    // filesystem the per-item WAL flush is the cost extend amortizes into one batch
    // sync. add-per-item vs extend over the same documents.
    use durability::FsDirectory;
    let mut g = c.benchmark_group("ingest_fs");
    let n = 4_000usize; // fewer than N: real-fs writes are slower
    g.throughput(Throughput::Elements(n as u64));
    let mk = |tag: &str| {
        let mut p = std::env::temp_dir();
        p.push(format!("gramdex-bench-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    };
    g.bench_function("add", |b| {
        b.iter_batched(
            || mk("add"),
            |p| {
                let mut s = 0x1234_5678_9abc_def0u64;
                let mut store =
                    gramdex::store::UpdatableIndex::open(FsDirectory::arc(&p).unwrap(), FLUSH, K)
                        .unwrap();
                for i in 0..n {
                    store.add(i as u32, text(&mut s)).unwrap();
                }
                let _ = std::fs::remove_dir_all(&p);
            },
            BatchSize::PerIteration,
        )
    });
    g.bench_function("extend", |b| {
        b.iter_batched(
            || mk("extend"),
            |p| {
                let mut s = 0x1234_5678_9abc_def0u64;
                let mut store =
                    gramdex::store::UpdatableIndex::open(FsDirectory::arc(&p).unwrap(), FLUSH, K)
                        .unwrap();
                store
                    .extend((0..n).map(|i| (i as u32, text(&mut s))))
                    .unwrap();
                let _ = std::fs::remove_dir_all(&p);
            },
            BatchSize::PerIteration,
        )
    });
    g.finish();
}

#[cfg(feature = "store")]
criterion_group!(g, benches, ingest_fs);
#[cfg(feature = "store")]
criterion_main!(g);
