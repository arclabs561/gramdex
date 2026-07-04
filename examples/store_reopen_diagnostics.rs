//! Measure sidecar-first snapshot reopen vs rebuild for the segstore-backed store.
//!
//! Run:
//! `cargo run --release --features store --example store_reopen_diagnostics`

use std::sync::Arc;
use std::time::{Duration, Instant};

use durability::{Directory, MemoryDirectory};
use gramdex::store::{SnapshotIndex, UpdatableIndex};

const N: usize = 1_000;
const FLUSH: usize = 200;
const K: usize = 3;
const MIN_SHARED: u32 = 20;
const QUERY_ID: u32 = 377;

type DynError = Box<dyn std::error::Error>;
type StoreDir = Arc<dyn Directory>;
type Candidates = Vec<u32>;

fn main() -> Result<(), DynError> {
    let (load_dir, query) = build_checkpointed_dir()?;
    let load_sidecars = sidecar_count(&load_dir)?;
    let (load_elapsed, load_hits) = first_snapshot_query(load_dir.clone(), &query)?;

    let (rebuild_dir, rebuild_query) = build_checkpointed_dir()?;
    let sidecars_before_delete = sidecar_count(&rebuild_dir)?;
    delete_sidecars(&rebuild_dir)?;
    let sidecars_after_delete = sidecar_count(&rebuild_dir)?;
    let (rebuild_elapsed, rebuild_hits) = first_snapshot_query(rebuild_dir, &rebuild_query)?;

    assert_eq!(load_hits, rebuild_hits);
    assert!(load_hits.contains(&QUERY_ID));

    println!("documents: {N}, flush threshold: {FLUSH}, k-grams: {K}");
    println!("sidecars loaded path: {load_sidecars}");
    println!(
        "sidecars rebuild path before/after delete: {sidecars_before_delete}/{sidecars_after_delete}"
    );
    println!(
        "first snapshot query with sidecars: {}",
        micros(load_elapsed)
    );
    println!(
        "first snapshot query after deleting sidecars: {}",
        micros(rebuild_elapsed)
    );
    println!("matching candidates: {}", load_hits.len());
    println!("query doc present: {}", load_hits.contains(&QUERY_ID));

    Ok(())
}

fn build_checkpointed_dir() -> Result<(StoreDir, String), DynError> {
    let dir: StoreDir = MemoryDirectory::arc();
    let mut index = UpdatableIndex::open(dir.clone(), FLUSH, K)?;
    index.extend((0..N).map(|id| (id as u32, text(id as u32))))?;
    index.checkpoint()?;
    Ok((dir, text(QUERY_ID)))
}

fn first_snapshot_query(dir: StoreDir, query: &str) -> Result<(Duration, Candidates), DynError> {
    let snapshot = SnapshotIndex::open(dir, K)?;
    let start = Instant::now();
    let hits = snapshot.candidates_min_shared(query, MIN_SHARED)?;
    Ok((start.elapsed(), hits))
}

fn delete_sidecars(dir: &StoreDir) -> Result<(), DynError> {
    for name in dir.list_dir("")? {
        if name.starts_with("segstore.idx.") {
            dir.delete(&name)?;
        }
    }
    Ok(())
}

fn sidecar_count(dir: &StoreDir) -> Result<usize, DynError> {
    Ok(dir
        .list_dir("")?
        .into_iter()
        .filter(|name| name.starts_with("segstore.idx."))
        .count())
}

fn text(id: u32) -> String {
    let mut state = 0x9abc_def0_1234_5678u64 ^ id as u64;
    format!(
        "record-{id:04}-{}-{}-{}-{}",
        token(&mut state),
        token(&mut state),
        token(&mut state),
        token(&mut state)
    )
}

fn token(state: &mut u64) -> String {
    (0..12)
        .map(|_| (b'a' + (xorshift(state) % 26) as u8) as char)
        .collect()
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn micros(duration: Duration) -> String {
    format!("{} us", duration.as_micros())
}
