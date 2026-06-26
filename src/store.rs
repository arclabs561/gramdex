//! Updatable, durable fuzzy string-match index (character k-grams) via `segstore`.
//!
//! Enabled by the optional `store` feature. The base [`GramDex`] is build-once;
//! this wraps it in a segstore `SegmentedStore` so documents can be added and
//! deleted incrementally with a write-ahead log + checkpoint + compaction, and
//! the index survives a restart.
//!
//! Each segment stores the source `(id, text)` pairs; a real `GramDex` over the
//! live documents of each segment is built and **cached**, rebuilt only when the
//! index is mutated (an add that seals a segment, a delete, or a compaction), not
//! on every query. The small unflushed buffer is built per query. The gram size
//! `k` is chosen at [`UpdatableIndex::open`] (`k = 3` for trigrams). gramdex is a
//! candidate generator, so the result is an unranked id set (verify with
//! [`crate::trigram_jaccard`] or the bounded planners as usual).

use std::cell::RefCell;
use std::sync::Arc;

use durability::{Directory, PersistenceResult};
use segstore::{SegmentedStore, Store};

use crate::GramDex;

/// segstore payload: items are document texts, a segment is a batch of source
/// texts (a `GramDex` is built + cached from them).
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

/// Cached per-segment indexes, valid for a given mutation generation.
struct Cache {
    generation: u64,
    segments: Vec<Option<GramDex>>,
}

/// An updatable, durable character k-gram fuzzy-match index.
pub struct UpdatableIndex {
    inner: SegmentedStore<GramBacking>,
    k: usize,
    generation: u64,
    cache: RefCell<Cache>,
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
            generation: 0,
            cache: RefCell::new(Cache {
                generation: u64::MAX,
                segments: Vec::new(),
            }),
        })
    }

    /// The character k-gram size this index matches on.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Add (or re-add) a document by id.
    pub fn add(&mut self, id: u32, text: impl Into<String>) -> PersistenceResult<()> {
        self.inner.add(id, text.into())?;
        self.generation += 1;
        Ok(())
    }

    /// Tombstone a document.
    pub fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.inner.delete(id)?;
        self.generation += 1;
        Ok(())
    }

    /// Merge segments (dropping tombstoned docs) and persist a checkpoint.
    pub fn compact(&mut self) -> PersistenceResult<()> {
        self.inner.compact()?;
        self.generation += 1;
        Ok(())
    }

    /// Persist a checkpoint without merging.
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.inner.checkpoint()
    }

    /// Candidate document ids whose character `k`-grams overlap `text`, unioned
    /// over every live document.
    pub fn candidates(&self, text: &str) -> Vec<u32> {
        self.refresh_cache();
        let mut out: Vec<u32> = Vec::new();
        {
            let cache = self.cache.borrow();
            for ix in cache.segments.iter().flatten() {
                out.extend(
                    ix.candidates_union_char_kgrams(text, self.k)
                        .unwrap_or_default(),
                );
            }
        }
        let buffered: Vec<(u32, String)> = self.inner.buffer().to_vec();
        if let Some(ix) = self.build_live_index(&buffered) {
            out.extend(
                ix.candidates_union_char_kgrams(text, self.k)
                    .unwrap_or_default(),
            );
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    fn refresh_cache(&self) {
        let mut cache = self.cache.borrow_mut();
        if cache.generation == self.generation {
            return;
        }
        cache.segments.clear();
        for seg in self.inner.segments() {
            cache.segments.push(self.build_live_index(seg));
        }
        cache.generation = self.generation;
    }

    fn build_live_index(&self, items: &[(u32, String)]) -> Option<GramDex> {
        let mut ix = GramDex::new();
        let mut any = false;
        for (id, doc) in items {
            // Skip docs too short for `k` grams (add_document_char_kgrams errors).
            if self.inner.is_live(id) && ix.add_document_char_kgrams(*id, doc, self.k).is_ok() {
                any = true;
            }
        }
        if !any {
            return None;
        }
        Some(ix)
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
            let mut store = UpdatableIndex::open(dir.clone(), 2, 3).unwrap();
            store.add(1, "hello").unwrap();
            store.add(2, "yellow").unwrap();
            store.add(3, "mellow").unwrap();

            let c = store.candidates("mellow");
            assert!(c.contains(&3) && c.contains(&2));
            assert_eq!(store.candidates("mellow"), c, "cached query is stable");

            store.delete(2).unwrap();
            assert!(
                !store.candidates("mellow").contains(&2),
                "delete invalidates the cache"
            );

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
        let mut store = UpdatableIndex::open(dir, 4, 5).unwrap();
        store.add(1, "hello").unwrap();
        store.add(2, "yellow").unwrap();
        let c = store.candidates("mellow");
        assert!(c.contains(&2), "yellow shares the 5-gram ellow");
        assert!(!c.contains(&1), "hello shares no 5-gram with mellow at k=5");
    }
}
