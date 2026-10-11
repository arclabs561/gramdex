//! segstore's shared conformance suite against gramdex's updatable index.
#![cfg(feature = "store")]

use std::sync::Arc;

use durability::{Directory, PersistenceResult};
use gramdex::store::UpdatableIndex;
use segstore::conformance::{run_all, ConformanceStore};

const K: usize = 3;

struct Adapter(UpdatableIndex);

/// 48 pseudo-random lowercase letters: about 46 distinct trigrams, so two
/// documents for different seeds never share all of them.
fn text(seed: u64) -> String {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..48)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            char::from(b'a' + (x % 26) as u8)
        })
        .collect()
}

impl ConformanceStore for Adapter {
    type Doc = String;

    fn open(dir: Arc<dyn Directory>) -> PersistenceResult<Self> {
        UpdatableIndex::open(dir, 3, K).map(Self)
    }
    fn add(&mut self, id: u32, doc: String) -> PersistenceResult<()> {
        self.0.add(id, doc)
    }
    fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.0.delete(id)
    }
    fn compact(&mut self) -> PersistenceResult<()> {
        self.0.compact()
    }
    fn compact_tiers(&mut self) -> PersistenceResult<()> {
        self.0.compact_tiers()
    }
    fn reclaim(&mut self, min_live_ratio: f64) -> PersistenceResult<()> {
        self.0.reclaim(min_live_ratio)
    }
    fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.0.checkpoint()
    }
    fn doc(seed: u64) -> String {
        text(seed)
    }
    /// Ids sharing every distinct trigram of `doc`.
    fn matching(&self, doc: &String) -> Vec<u32> {
        let chars: Vec<char> = doc.chars().collect();
        let mut grams: Vec<String> = chars.windows(K).map(|w| w.iter().collect()).collect();
        grams.sort();
        grams.dedup();
        self.0.candidates_min_shared(doc, grams.len() as u32)
    }
}

#[test]
fn updatable_index_meets_segstore_conformance() {
    run_all::<Adapter>();
}
