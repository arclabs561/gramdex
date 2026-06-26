//! Updatable, durable fuzzy string-match index (character k-grams) via `segstore`.
//!
//! Enabled by the optional `store` feature. The base [`GramDex`] is build-once;
//! this wraps it in a segstore `SegmentedStore` so documents can be added and
//! deleted incrementally with a write-ahead log + checkpoint + compaction, and
//! the index survives a restart.
//!
//! Each segment stores the source `(id, text)` pairs; a real `GramDex` over the
//! live documents of each segment answers a query, and the candidate id sets are
//! unioned. The gram size `k` is chosen at [`UpdatableIndex::open`] (`k = 3` for
//! trigrams, larger for stricter matching). gramdex is a candidate generator, so
//! the result is an unranked id set (verify with [`crate::trigram_jaccard`] or
//! the bounded planners as usual).

use std::sync::Arc;

use durability::{Directory, PersistenceResult};
use segstore::{SegmentedStore, Store};

use crate::GramDex;

/// segstore payload: items are document texts, a segment is a batch of source
/// texts (the GramDex is rebuilt from them per query).
struct GramBacking;

impl Store for GramBacking {
    type Id = u32;
    type Item = String;
    type Segment = Vec<(u32, String)>;

    fn build_segment(&self, batch: &[(u32, String)]) -> Vec<(u32, String)> {
        batch.to_vec()
    }

    fn merge_segments(
        &self,
        segs: &[Vec<(u32, String)>],
        live: &dyn Fn(&u32) -> bool,
    ) -> Vec<(u32, String)> {
        segs.iter()
            .flatten()
            .filter(|(id, _)| live(id))
            .cloned()
            .collect()
    }
}

/// An updatable, durable character k-gram fuzzy-match index.
pub struct UpdatableIndex {
    inner: SegmentedStore<GramBacking>,
    k: usize,
}

impl UpdatableIndex {
    /// Open (or recover) an index under `dir` that matches on character `k`-grams
    /// (`k = 3` for trigrams). Up to `flush_threshold` documents are buffered
    /// before a new immutable segment is sealed.
    pub fn open(
        dir: Arc<dyn Directory>,
        flush_threshold: usize,
        k: usize,
    ) -> PersistenceResult<Self> {
        Ok(Self {
            inner: SegmentedStore::open(dir, GramBacking, flush_threshold)?,
            k,
        })
    }

    /// The character k-gram size this index matches on.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Add (or re-add) a document by id.
    pub fn add(&mut self, id: u32, text: impl Into<String>) -> PersistenceResult<()> {
        self.inner.add(id, text.into())
    }

    /// Tombstone a document.
    pub fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.inner.delete(id)
    }

    /// Merge segments (dropping tombstoned docs) and persist a checkpoint.
    pub fn compact(&mut self) -> PersistenceResult<()> {
        self.inner.compact()
    }

    /// Persist a checkpoint without merging.
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.inner.checkpoint()
    }

    /// Candidate document ids whose character `k`-grams overlap `text`, unioned
    /// over every live document (`k` is the value passed to [`Self::open`]).
    pub fn candidates(&self, text: &str) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for seg in self.inner.segments() {
            out.extend(self.candidates_in(seg, text));
        }
        let buffered = self.inner.buffer().to_vec();
        out.extend(self.candidates_in(&buffered, text));
        out.sort_unstable();
        out.dedup();
        out
    }

    fn candidates_in(&self, batch: &[(u32, String)], text: &str) -> Vec<u32> {
        let mut ix = GramDex::new();
        let mut any = false;
        for (id, doc) in batch {
            // Skip docs too short for `k` grams (add_document_char_kgrams errors).
            if self.inner.is_live(id) && ix.add_document_char_kgrams(*id, doc, self.k).is_ok() {
                any = true;
            }
        }
        if !any {
            return Vec::new();
        }
        ix.candidates_union_char_kgrams(text, self.k)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use durability::MemoryDirectory;

    #[test]
    fn add_delete_compact_recover_with_configurable_k() {
        let dir = MemoryDirectory::arc();
        {
            // k = 3 (trigrams).
            let mut store = UpdatableIndex::open(dir.clone(), 2, 3).unwrap();
            store.add(1, "hello").unwrap();
            store.add(2, "yellow").unwrap(); // flush
            store.add(3, "mellow").unwrap(); // buffered

            let c = store.candidates("mellow");
            assert!(c.contains(&3), "exact match is a candidate");
            assert!(c.contains(&2), "yellow shares trigrams with mellow");

            store.delete(2).unwrap();
            let c = store.candidates("mellow");
            assert!(!c.contains(&2), "deleted doc drops out of candidates");
            assert!(c.contains(&3));

            store.compact().unwrap();
            let c = store.candidates("mellow");
            assert!(
                c.contains(&3) && !c.contains(&2),
                "compaction preserves the result"
            );
        }
        let store = UpdatableIndex::open(dir, 2, 3).unwrap();
        let c = store.candidates("mellow");
        assert!(
            c.contains(&3) && !c.contains(&2),
            "recovery preserves the result"
        );
    }

    #[test]
    fn larger_k_is_stricter() {
        let dir = MemoryDirectory::arc();
        // k = 5: "mellow" and "yellow" share the 5-gram "ellow", "hello" does not.
        let mut store = UpdatableIndex::open(dir, 4, 5).unwrap();
        store.add(1, "hello").unwrap();
        store.add(2, "yellow").unwrap();
        let c = store.candidates("mellow");
        assert!(c.contains(&2), "yellow shares the 5-gram ellow");
        assert!(!c.contains(&1), "hello shares no 5-gram with mellow at k=5");
    }
}
