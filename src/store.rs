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
use std::collections::HashMap;
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

    fn segment_len(&self, seg: &Vec<(u32, String)>) -> usize {
        seg.len()
    }

    fn live_len(&self, seg: &Vec<(u32, String)>, live: &dyn Fn(&u32) -> bool) -> Option<usize> {
        Some(seg.iter().filter(|(id, _)| live(id)).count())
    }
}

/// Per-segment indexes keyed by the segment's stable `Arc` identity. Because
/// segstore keeps an unchanged segment's `Arc` across mutations, a sealed add
/// only builds the one new segment's index (the rest are reused) instead of
/// rebuilding the whole corpus -- the dominant cost in an add-then-query loop.
struct Cache {
    by_ptr: HashMap<usize, Option<GramDex>>,
}

/// An updatable, durable character k-gram fuzzy-match index.
pub struct UpdatableIndex {
    inner: SegmentedStore<GramBacking>,
    k: usize,
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
            cache: RefCell::new(Cache {
                by_ptr: HashMap::new(),
            }),
        })
    }

    /// The character k-gram size this index matches on.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Add (or re-add) a document by id.
    pub fn add(&mut self, id: u32, text: impl Into<String>) -> PersistenceResult<()> {
        // A sealed add introduces a new segment (a new Arc identity); existing
        // segments keep theirs, so the cache reuses them and builds only the new one.
        self.inner.add(id, text.into())?;
        Ok(())
    }

    /// Tombstone a document.
    pub fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.inner.delete(id)?;
        // A tombstone changes the live filter used to build every segment's index,
        // so invalidate the cache (deletes are far rarer than adds).
        self.cache.borrow_mut().by_ptr.clear();
        Ok(())
    }

    /// Merge segments (dropping tombstoned docs) and persist a checkpoint.
    pub fn compact(&mut self) -> PersistenceResult<()> {
        self.inner.compact()?;
        Ok(())
    }

    /// Persist a checkpoint without merging.
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.inner.checkpoint()
    }

    /// Run one round of size-tiered compaction, merging similarly-sized segments
    /// so the segment count stays bounded without a full [`compact`](Self::compact).
    pub fn compact_tiers(&mut self) -> PersistenceResult<()> {
        self.inner.compact_tiers()?;
        Ok(())
    }

    /// Merge only the segments whose live ratio is below `min_live_ratio`,
    /// reclaiming tombstoned documents -- the cheap alternative to a full
    /// [`compact`](Self::compact) when a few segments are delete-heavy.
    pub fn reclaim(&mut self, min_live_ratio: f64) -> PersistenceResult<()> {
        self.inner.reclaim_tombstones(min_live_ratio)?;
        Ok(())
    }

    /// Storage amplification: stored documents divided by live documents (`1.0`
    /// with no tombstones, higher as deletes accumulate).
    pub fn space_amplification(&self) -> Option<f64> {
        self.inner.space_amplification()
    }

    /// Candidate document ids whose character `k`-grams overlap `text`, unioned
    /// over every live document.
    pub fn candidates(&self, text: &str) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        {
            let segs = self.inner.segments();
            let mut cache = self.cache.borrow_mut();
            // Drop cached indexes for segments no longer present (post-compaction).
            let current: std::collections::HashSet<usize> =
                segs.iter().map(|a| Arc::as_ptr(a) as usize).collect();
            cache.by_ptr.retain(|key, _| current.contains(key));
            // Build only segments not already cached (i.e. new ones).
            for seg in segs {
                let key = Arc::as_ptr(seg) as usize;
                cache
                    .by_ptr
                    .entry(key)
                    .or_insert_with(|| self.build_live_index(&seg[..]));
            }
            for ix in cache.by_ptr.values().flatten() {
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
