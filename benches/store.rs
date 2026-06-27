//! Benchmarks for the `store` feature (segstore-backed updatable k-gram index).
//!
//! Run: `cargo bench --features store --bench store`. Without the feature the
//! harness is an empty no-op so the target still compiles. Measures build
//! throughput, warm query latency (per-segment index cached), and the cold
//! "rebuild every segment" cost -- what a delete that clears the whole cache
//! pays, which the targeted-invalidation delete avoids (one segment instead).

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
fn fresh_store(warm: bool) -> (gramdex::store::UpdatableIndex, String) {
    use durability::MemoryDirectory;
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut store = gramdex::store::UpdatableIndex::open(MemoryDirectory::arc(), FLUSH, K).unwrap();
    for i in 0..N {
        store.add(i as u32, text(&mut s)).unwrap();
    }
    store.checkpoint().unwrap();
    let q = text(&mut s);
    if warm {
        let _ = store.candidates(&q);
    }
    (store, q)
}

#[cfg(feature = "store")]
fn benches(c: &mut Criterion) {
    let mut g = c.benchmark_group("store");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("build", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = fresh_store(false);
            },
            BatchSize::SmallInput,
        )
    });

    let (warm, q) = fresh_store(true);
    g.bench_function("search_warm", |b| b.iter(|| warm.candidates(&q)));

    g.bench_function("search_cold_rebuild_all", |b| {
        b.iter_batched(
            || fresh_store(false),
            |(store, q)| store.candidates(&q),
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

#[cfg(feature = "store")]
criterion_group!(g, benches);
#[cfg(feature = "store")]
criterion_main!(g);
