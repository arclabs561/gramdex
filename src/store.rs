//! Updatable, durable fuzzy string-match index (character k-grams) via `segstore`.
//!
//! Enabled by the optional `store` feature. The base [`GramDex`] is build-once;
//! this wraps it in a segstore `SegmentedStore` so documents can be added and
//! deleted incrementally with a write-ahead log + checkpoint + compaction, and
//! the index survives a restart.
//!
//! Each segment stores the source `(id, text)` pairs; a real `GramDex` over the
//! live documents of each segment is built, cached, and persisted as a
//! per-segment sidecar. [`crate::store::UpdatableIndex`] keeps the writer path
//! simple and loads source segments on open; [`crate::store::SnapshotIndex`] is
//! the read-only restart path that opens only the segstore manifest and uses
//! sidecars first, decoding a source segment only when its sidecar is missing or
//! unusable. The gram size `k` is chosen at open (`k = 3` for trigrams).
//! gramdex is a candidate generator, so the result is an unranked id set (verify
//! with [`crate::trigram_jaccard`] or the bounded planners as usual).

use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::sync::Arc;

use durability::{Directory, PersistenceResult};
use segstore::{SegmentCatalog, SegmentedStore, Store};

use crate::{char_kgrams, GramDex};

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
        segs: &[&Vec<(u32, String)>],
        live: &dyn Fn(&u32) -> bool,
    ) -> Vec<(u32, String)> {
        segs.iter()
            .flat_map(|s| s.iter())
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

/// Per-segment indexes keyed by segstore's stable segment id. A sealed add
/// creates one new segment id, so cached indexes for existing segments are
/// reused instead of rebuilding the whole corpus on the next query.
struct Cache {
    by_segment_id: HashMap<u64, Option<GramDex>>,
}

/// The `kind` tag for a persisted per-segment GramDex sidecar.
const INDEX_KIND: &str = "gramdex";
const SIDECAR_MAGIC: &[u8; 8] = b"GRDXIDX1";
const SIDECAR_VERSION: u32 = 1;

#[derive(serde::Serialize)]
struct GramSidecarRef<'a> {
    index: &'a GramDex,
    ids: Vec<u32>,
}

#[derive(serde::Deserialize)]
struct GramSidecar {
    index: GramDex,
    ids: Vec<u32>,
}

fn make_sidecar_recipe(k: usize) -> String {
    format!("gramdex-store-v1;codec=postcard-gramdex-v1;k={k}")
}

fn encode_sidecar(recipe: &str, index: &[u8]) -> Option<Vec<u8>> {
    let recipe = recipe.as_bytes();
    let recipe_len = u32::try_from(recipe.len()).ok()?;
    let mut bytes = Vec::with_capacity(16 + recipe.len() + index.len());
    bytes.extend_from_slice(SIDECAR_MAGIC);
    bytes.extend_from_slice(&SIDECAR_VERSION.to_le_bytes());
    bytes.extend_from_slice(&recipe_len.to_le_bytes());
    bytes.extend_from_slice(recipe);
    bytes.extend_from_slice(index);
    Some(bytes)
}

fn decode_sidecar<'a>(recipe: &str, bytes: &'a [u8]) -> Option<&'a [u8]> {
    if bytes.len() < 16 {
        return None;
    }
    if &bytes[..8] != SIDECAR_MAGIC {
        return None;
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
    if version != SIDECAR_VERSION {
        return None;
    }
    let recipe_len = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
    let recipe_start = 16usize;
    let recipe_end = recipe_start.checked_add(recipe_len)?;
    if bytes.len() < recipe_end {
        return None;
    }
    if &bytes[recipe_start..recipe_end] != recipe.as_bytes() {
        return None;
    }
    Some(&bytes[recipe_end..])
}

fn build_index_from_items(
    items: &[(u32, String)],
    k: usize,
    is_live: impl Fn(&u32) -> bool,
) -> Option<GramDex> {
    let mut ix = GramDex::new();
    let mut any = false;
    for (id, doc) in items {
        // Skip docs too short for `k` grams (add_document_char_kgrams errors).
        if is_live(id) && ix.add_document_char_kgrams(*id, doc, k).is_ok() {
            any = true;
        }
    }
    if !any {
        return None;
    }
    Some(ix)
}

/// An updatable, durable character k-gram fuzzy-match index.
pub struct UpdatableIndex {
    inner: SegmentedStore<GramBacking>,
    k: usize,
    sidecar_recipe: String,
    cache: RefCell<Cache>,
    /// Segment ids whose on-disk GramDex sidecar was validated or written in
    /// this process, so checkpoint persistence stays O(new segments).
    persisted: RefCell<HashSet<u64>>,
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
            sidecar_recipe: make_sidecar_recipe(k),
            cache: RefCell::new(Cache {
                by_segment_id: HashMap::new(),
            }),
            persisted: RefCell::new(HashSet::new()),
        })
    }

    /// The character k-gram size this index matches on.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Add (or re-add) a document by id.
    pub fn add(&mut self, id: u32, text: impl Into<String>) -> PersistenceResult<()> {
        // A sealed add introduces a new segment id; existing segment ids stay
        // stable, so the cache reuses them and builds only the new one.
        self.inner.add(id, text.into())?;
        Ok(())
    }

    /// Add (or re-add) many documents, syncing the write-ahead log once for the
    /// whole batch instead of once per document. This is the bulk-ingest path (the
    /// corpus-load phase): per-item WAL sync is the dominant cost on a real disk, so
    /// one sync per batch is several times faster than a loop of [`Self::add`]. A
    /// crash mid-batch recovers a consistent prefix (each document is an
    /// independently CRC-checked WAL record).
    pub fn extend(
        &mut self,
        docs: impl IntoIterator<Item = (u32, String)>,
    ) -> PersistenceResult<()> {
        self.inner.extend(docs)?;
        Ok(())
    }

    /// Tombstone a document.
    pub fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.inner.delete(id)?;
        // A tombstone only changes the live-set of the segment that holds `id`, so
        // invalidate just that segment's cached index -- not the whole cache.
        let mut cache = self.cache.borrow_mut();
        for (seg_idx, seg) in self.inner.segments().iter().enumerate() {
            if seg.iter().any(|(sid, _)| *sid == id) {
                let seg_id = self.inner.segment_ids()[seg_idx];
                cache.by_segment_id.remove(&seg_id);
                self.persisted.borrow_mut().remove(&seg_id);
                let _ = self
                    .inner
                    .dir()
                    .delete(&self.inner.index_name(seg_id, INDEX_KIND));
            }
        }
        Ok(())
    }

    /// Merge segments (dropping tombstoned docs) and persist a checkpoint.
    pub fn compact(&mut self) -> PersistenceResult<()> {
        self.inner.compact()?;
        self.prune_cache_to_current_segments();
        self.persist_new_segments();
        Ok(())
    }

    /// Persist a checkpoint without merging.
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.inner.checkpoint()?;
        self.persist_new_segments();
        Ok(())
    }

    /// Run one round of size-tiered compaction, merging similarly-sized segments
    /// so the segment count stays bounded without a full [`compact`](Self::compact).
    pub fn compact_tiers(&mut self) -> PersistenceResult<()> {
        let stats = self.inner.compact_tiers()?;
        if stats.merges > 0 {
            self.prune_cache_to_current_segments();
            self.persist_new_segments();
        }
        Ok(())
    }

    /// Merge only the segments whose live ratio is below `min_live_ratio`,
    /// reclaiming tombstoned documents -- the cheap alternative to a full
    /// [`compact`](Self::compact) when a few segments are delete-heavy.
    pub fn reclaim(&mut self, min_live_ratio: f64) -> PersistenceResult<()> {
        let stats = self.inner.reclaim_tombstones(min_live_ratio)?;
        if stats.merges > 0 {
            self.prune_cache_to_current_segments();
            self.persist_new_segments();
        }
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
        let query_grams = char_kgrams(text, self.k).unwrap_or_default();
        self.candidates_from_grams(&query_grams, |ix, grams| ix.candidates_union(grams))
    }

    /// Candidate document ids that share at least `min_shared` distinct
    /// character `k`-grams with `text`.
    ///
    /// This is a cheaper pre-verification filter than the plain union path for
    /// broad fuzzy-match queries. `min_shared <= 1` is equivalent to
    /// [`Self::candidates`].
    pub fn candidates_min_shared(&self, text: &str, min_shared: u32) -> Vec<u32> {
        let query_grams = char_kgrams(text, self.k).unwrap_or_default();
        self.candidates_from_grams(&query_grams, |ix, grams| {
            ix.candidates_union_min_shared(grams, min_shared)
        })
    }

    fn candidates_from_grams(
        &self,
        query_grams: &[String],
        mut per_segment: impl FnMut(&GramDex, &[String]) -> Vec<u32>,
    ) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        {
            let segs = self.inner.segments();
            let mut cache = self.cache.borrow_mut();
            // Build only current segments not already cached, loading a persisted
            // sidecar first when one matches the current recipe and live id set.
            let ids = self.inner.segment_ids();
            for (i, seg) in segs.iter().enumerate() {
                let seg_id = ids[i];
                let index = cache
                    .by_segment_id
                    .entry(seg_id)
                    .or_insert_with(|| self.build_or_load(&seg[..], seg_id));
                if let Some(ix) = index {
                    out.extend(per_segment(ix, query_grams));
                }
            }
        }
        let buffered = self.inner.buffer();
        if let Some(ix) = self.build_live_index(buffered) {
            out.extend(per_segment(&ix, query_grams));
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    fn prune_cache_to_current_segments(&self) {
        let current: HashSet<u64> = self.inner.segment_ids().iter().copied().collect();
        self.cache
            .borrow_mut()
            .by_segment_id
            .retain(|id, _| current.contains(id));
    }

    fn build_live_index(&self, items: &[(u32, String)]) -> Option<GramDex> {
        build_index_from_items(items, self.k, |id| self.inner.is_live(id))
    }

    /// Load segment `seg_id`'s persisted GramDex sidecar, or build it over the
    /// segment's live documents and persist it for the next restart.
    fn build_or_load(&self, seg: &[(u32, String)], seg_id: u64) -> Option<GramDex> {
        if let Some(index) = self.load_sidecar(seg, seg_id) {
            self.persisted.borrow_mut().insert(seg_id);
            return Some(index);
        }
        let index = self.build_live_index(seg)?;
        self.persist_sidecar(&index, seg, seg_id);
        Some(index)
    }

    /// Load a sidecar only if its recipe and live id set match the current
    /// segment. A stale sidecar can never serve a tombstoned document.
    fn load_sidecar(&self, seg: &[(u32, String)], seg_id: u64) -> Option<GramDex> {
        let name = self.inner.index_name(seg_id, INDEX_KIND);
        if !self.inner.dir().exists(&name) {
            return None;
        }
        let mut bytes = Vec::new();
        self.inner
            .dir()
            .open_file(&name)
            .ok()?
            .read_to_end(&mut bytes)
            .ok()?;
        let index_bytes = self.decode_sidecar(&bytes)?;
        let mut sidecar: GramSidecar = postcard::from_bytes(index_bytes).ok()?;
        sidecar.ids.sort_unstable();
        sidecar.ids.dedup();
        if sidecar.ids == self.live_ids_vec(seg) {
            Some(sidecar.index)
        } else {
            None
        }
    }

    /// Persist a built per-segment GramDex as its sidecar. Best-effort: a failed
    /// write leaves the in-memory index usable and simply rebuilds later.
    fn persist_sidecar(&self, index: &GramDex, seg: &[(u32, String)], seg_id: u64) {
        let sidecar = GramSidecarRef {
            index,
            ids: self.live_ids_vec(seg),
        };
        if let Ok(index_bytes) = postcard::to_allocvec(&sidecar) {
            let Some(bytes) = self.encode_sidecar(&index_bytes) else {
                return;
            };
            if self
                .inner
                .dir()
                .atomic_write(&self.inner.index_name(seg_id, INDEX_KIND), &bytes)
                .is_ok()
            {
                self.persisted.borrow_mut().insert(seg_id);
            }
        }
    }

    fn live_ids_vec(&self, seg: &[(u32, String)]) -> Vec<u32> {
        let mut ids: Vec<u32> = seg
            .iter()
            .filter_map(|(id, _)| self.inner.is_live(id).then_some(*id))
            .collect();
        ids.sort_unstable();
        ids
    }

    fn encode_sidecar(&self, index: &[u8]) -> Option<Vec<u8>> {
        encode_sidecar(&self.sidecar_recipe, index)
    }

    fn decode_sidecar<'a>(&self, bytes: &'a [u8]) -> Option<&'a [u8]> {
        decode_sidecar(&self.sidecar_recipe, bytes)
    }

    /// Persist sidecars for sealed segments that lack a current one. This is
    /// incremental: already validated/written segment ids are skipped.
    fn persist_new_segments(&self) {
        let ids = self.inner.segment_ids();
        let id_set: HashSet<u64> = ids.iter().copied().collect();
        self.persisted.borrow_mut().retain(|id| id_set.contains(id));
        for (i, seg) in self.inner.segments().iter().enumerate() {
            let seg_id = ids[i];
            if self.persisted.borrow().contains(&seg_id) {
                continue;
            }
            if self.load_sidecar(&seg[..], seg_id).is_some() {
                self.persisted.borrow_mut().insert(seg_id);
                continue;
            }
            if let Some(index) = self.build_live_index(&seg[..]) {
                self.persist_sidecar(&index, &seg[..], seg_id);
            }
        }
    }
}

/// A read-only checkpoint view that loads per-segment `GramDex` sidecars before
/// falling back to source segment payloads.
///
/// This is the restart/query path for larger stores whose built gram indexes
/// have already been persisted by [`UpdatableIndex::checkpoint`]. It opens the
/// segstore manifest without decoding source segments, then applies catalog
/// tombstones to sidecar candidates at query time. If a sidecar is missing,
/// stale by recipe, or not decodable, only that one source segment is decoded to
/// rebuild the sidecar.
pub struct SnapshotIndex {
    catalog: SegmentCatalog<u32>,
    k: usize,
    sidecar_recipe: String,
    cache: RefCell<Cache>,
}

impl SnapshotIndex {
    /// Open the last checkpoint under `dir` as a read-only search snapshot.
    ///
    /// WAL records after the last checkpoint are intentionally not visible;
    /// checkpoint before opening a snapshot when newly added documents must be
    /// searchable through this path.
    pub fn open(dir: Arc<dyn Directory>, k: usize) -> PersistenceResult<Self> {
        Ok(Self {
            catalog: SegmentCatalog::open(dir)?,
            k,
            sidecar_recipe: make_sidecar_recipe(k),
            cache: RefCell::new(Cache {
                by_segment_id: HashMap::new(),
            }),
        })
    }

    /// The character k-gram size this snapshot matches on.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Number of checkpointed immutable segments in this snapshot.
    pub fn segment_count(&self) -> usize {
        self.catalog.segment_count()
    }

    /// Number of tombstoned document ids in this snapshot.
    pub fn tombstone_count(&self) -> usize {
        self.catalog.tombstone_count()
    }

    /// Candidate document ids whose character `k`-grams overlap `text`, unioned
    /// over every live checkpointed document.
    pub fn candidates(&self, text: &str) -> PersistenceResult<Vec<u32>> {
        let query_grams = char_kgrams(text, self.k).unwrap_or_default();
        self.candidates_from_grams(&query_grams, |ix, grams| ix.candidates_union(grams))
    }

    /// Candidate document ids that share at least `min_shared` distinct
    /// character `k`-grams with `text`.
    pub fn candidates_min_shared(
        &self,
        text: &str,
        min_shared: u32,
    ) -> PersistenceResult<Vec<u32>> {
        let query_grams = char_kgrams(text, self.k).unwrap_or_default();
        self.candidates_from_grams(&query_grams, |ix, grams| {
            ix.candidates_union_min_shared(grams, min_shared)
        })
    }

    fn candidates_from_grams(
        &self,
        query_grams: &[String],
        mut per_segment: impl FnMut(&GramDex, &[String]) -> Vec<u32>,
    ) -> PersistenceResult<Vec<u32>> {
        let mut out: Vec<u32> = Vec::new();
        let mut cache = self.cache.borrow_mut();
        for &seg_id in self.catalog.segment_ids() {
            if let Entry::Vacant(entry) = cache.by_segment_id.entry(seg_id) {
                let index = self.build_or_load(seg_id)?;
                entry.insert(index);
            }
            if let Some(Some(ix)) = cache.by_segment_id.get(&seg_id) {
                out.extend(per_segment(ix, query_grams));
            }
        }
        out.retain(|id| self.catalog.is_live(id));
        out.sort_unstable();
        out.dedup();
        Ok(out)
    }

    fn build_or_load(&self, seg_id: u64) -> PersistenceResult<Option<GramDex>> {
        if let Some(index) = self.load_sidecar(seg_id) {
            return Ok(Some(index));
        }
        let segment: Vec<(u32, String)> = self.catalog.read_segment(seg_id)?;
        let index = self.build_live_index(&segment);
        if let Some(index) = &index {
            self.persist_sidecar(index, &segment, seg_id);
        }
        Ok(index)
    }

    fn load_sidecar(&self, seg_id: u64) -> Option<GramDex> {
        let name = self.catalog.index_name(seg_id, INDEX_KIND);
        if !self.catalog.dir().exists(&name) {
            return None;
        }
        let mut bytes = Vec::new();
        self.catalog
            .dir()
            .open_file(&name)
            .ok()?
            .read_to_end(&mut bytes)
            .ok()?;
        let index_bytes = decode_sidecar(&self.sidecar_recipe, &bytes)?;
        let sidecar: GramSidecar = postcard::from_bytes(index_bytes).ok()?;
        Some(sidecar.index)
    }

    fn build_live_index(&self, items: &[(u32, String)]) -> Option<GramDex> {
        build_index_from_items(items, self.k, |id| self.catalog.is_live(id))
    }

    fn persist_sidecar(&self, index: &GramDex, segment: &[(u32, String)], seg_id: u64) {
        let sidecar = GramSidecarRef {
            index,
            ids: self.live_ids_vec(segment),
        };
        if let Ok(index_bytes) = postcard::to_allocvec(&sidecar) {
            let Some(bytes) = encode_sidecar(&self.sidecar_recipe, &index_bytes) else {
                return;
            };
            let _ = self
                .catalog
                .dir()
                .atomic_write(&self.catalog.index_name(seg_id, INDEX_KIND), &bytes);
        }
    }

    fn live_ids_vec(&self, segment: &[(u32, String)]) -> Vec<u32> {
        let mut ids: Vec<u32> = segment
            .iter()
            .filter_map(|(id, _)| self.catalog.is_live(id).then_some(*id))
            .collect();
        ids.sort_unstable();
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use durability::MemoryDirectory;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::Mutex;

    const A: &str = "hello";
    const B: &str = "yellow";
    const C: &str = "mellow";
    const D: &str = "a separate unrelated document";

    struct RecordingDirectory {
        inner: Arc<dyn Directory>,
        opened: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingDirectory {
        fn wrap(inner: Arc<dyn Directory>) -> (Arc<dyn Directory>, Arc<Mutex<Vec<String>>>) {
            let opened = Arc::new(Mutex::new(Vec::new()));
            (
                Arc::new(Self {
                    inner,
                    opened: opened.clone(),
                }),
                opened,
            )
        }
    }

    impl Directory for RecordingDirectory {
        fn create_file(&self, path: &str) -> PersistenceResult<Box<dyn Write + Send>> {
            self.inner.create_file(path)
        }

        fn open_file(&self, path: &str) -> PersistenceResult<Box<dyn Read + Send>> {
            self.opened.lock().unwrap().push(path.to_string());
            self.inner.open_file(path)
        }

        fn exists(&self, path: &str) -> bool {
            self.inner.exists(path)
        }

        fn delete(&self, path: &str) -> PersistenceResult<()> {
            self.inner.delete(path)
        }

        fn atomic_rename(&self, from: &str, to: &str) -> PersistenceResult<()> {
            self.inner.atomic_rename(from, to)
        }

        fn create_dir_all(&self, path: &str) -> PersistenceResult<()> {
            self.inner.create_dir_all(path)
        }

        fn list_dir(&self, path: &str) -> PersistenceResult<Vec<String>> {
            self.inner.list_dir(path)
        }

        fn append_file(&self, path: &str) -> PersistenceResult<Box<dyn Write + Send>> {
            self.inner.append_file(path)
        }

        fn atomic_write(&self, path: &str, data: &[u8]) -> PersistenceResult<()> {
            self.inner.atomic_write(path, data)
        }

        fn file_path(&self, path: &str) -> Option<PathBuf> {
            self.inner.file_path(path)
        }
    }

    fn read_file(dir: &Arc<dyn Directory>, name: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        dir.open_file(name)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        bytes
    }

    fn checkpointed_store(dir: Arc<dyn Directory>, k: usize) -> (String, Vec<u8>) {
        let mut store = UpdatableIndex::open(dir, 2, k).unwrap();
        store.add(1, A).unwrap();
        store.add(2, B).unwrap();
        store.add(3, C).unwrap();
        store.add(4, D).unwrap();
        store.checkpoint().unwrap();
        let seg_id = store.inner.segment_ids()[0];
        let name = store.inner.index_name(seg_id, INDEX_KIND);
        let bytes = read_file(store.inner.dir(), &name);
        (name, bytes)
    }

    #[test]
    fn add_delete_compact_recover_with_configurable_k() {
        let dir = MemoryDirectory::arc();
        {
            let mut store = UpdatableIndex::open(dir.clone(), 2, 3).unwrap();
            store.add(1, A).unwrap();
            store.add(2, B).unwrap();
            store.add(3, C).unwrap();

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
        store.add(1, A).unwrap();
        store.add(2, B).unwrap();
        let c = store.candidates("mellow");
        assert!(c.contains(&2), "yellow shares the 5-gram ellow");
        assert!(!c.contains(&1), "hello shares no 5-gram with mellow at k=5");
    }

    #[test]
    fn candidates_min_shared_filters_weak_store_candidates() {
        let dir = MemoryDirectory::arc();
        {
            let mut store = UpdatableIndex::open(dir.clone(), 2, 3).unwrap();
            store.add(1, A).unwrap();
            store.add(2, B).unwrap();
            store.add(3, C).unwrap();
            store.add(4, D).unwrap();

            assert_eq!(store.candidates("mellow"), vec![1, 2, 3]);
            assert_eq!(store.candidates_min_shared("mellow", 1), vec![1, 2, 3]);
            assert_eq!(store.candidates_min_shared("mellow", 3), vec![2, 3]);

            store.delete(2).unwrap();
            store.checkpoint().unwrap();
        }

        let store = UpdatableIndex::open(dir, 2, 3).unwrap();
        assert_eq!(store.candidates_min_shared("mellow", 3), vec![3]);
    }

    #[test]
    fn checkpoint_persists_sidecars_and_reopen_loads_them() {
        let dir = MemoryDirectory::arc();
        {
            let mut store = UpdatableIndex::open(dir.clone(), 2, 3).unwrap();
            store.add(1, A).unwrap();
            store.add(2, B).unwrap();
            store.add(3, C).unwrap();
            store.add(4, D).unwrap();
            store.checkpoint().unwrap();

            let ids: Vec<u64> = store.inner.segment_ids().to_vec();
            assert!(
                !ids.is_empty(),
                "4 docs at flush 2 seal at least one segment"
            );
            for id in &ids {
                assert!(
                    store
                        .inner
                        .dir()
                        .exists(&store.inner.index_name(*id, INDEX_KIND)),
                    "segment {id} must have a persisted sidecar after checkpoint"
                );
            }
        }

        let store = UpdatableIndex::open(dir, 2, 3).unwrap();
        let c = store.candidates(C);
        assert!(
            c.contains(&2) && c.contains(&3),
            "search over loaded sidecars returns k-gram candidates"
        );
    }

    #[test]
    fn compact_prunes_cached_segment_indexes() {
        let dir = MemoryDirectory::arc();
        let mut store = UpdatableIndex::open(dir, 2, 3).unwrap();
        store.add(1, A).unwrap();
        store.add(2, B).unwrap();
        store.add(3, C).unwrap();
        store.add(4, D).unwrap();

        let before_ids = store.inner.segment_ids().to_vec();
        assert!(
            before_ids.len() >= 2,
            "test setup should create multiple sealed segments"
        );
        let _ = store.candidates(C);
        assert_eq!(
            store.cache.borrow().by_segment_id.len(),
            before_ids.len(),
            "warm query should cache each sealed segment"
        );

        store.compact().unwrap();

        let after_ids = store.inner.segment_ids().to_vec();
        assert_eq!(
            after_ids.len(),
            1,
            "compact should merge the sealed segments"
        );
        assert!(
            store
                .cache
                .borrow()
                .by_segment_id
                .keys()
                .all(|id| after_ids.contains(id)),
            "cache should not retain indexes for compacted-away segment ids"
        );
    }

    #[test]
    fn gramdex_sidecar_recipe_mismatch_rebuilds() {
        let dir = MemoryDirectory::arc();
        let (name, before) = checkpointed_store(dir.clone(), 3);
        assert_eq!(
            &before[..SIDECAR_MAGIC.len()],
            SIDECAR_MAGIC,
            "new sidecars carry the gramdex envelope"
        );

        let store = UpdatableIndex::open(dir.clone(), 2, 4).unwrap();
        let seg_id = store.inner.segment_ids()[0];
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_none(),
            "sidecar built with k=3 must not load under k=4"
        );
        assert!(
            !store.candidates(C).is_empty(),
            "mismatched sidecar falls back to rebuild"
        );

        let after = read_file(store.inner.dir(), &name);
        assert_ne!(before, after, "rebuild overwrites the stale-recipe sidecar");
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_some(),
            "rebuilt sidecar now matches the current recipe"
        );
    }

    #[test]
    fn gramdex_sidecar_envelope_rejects_corrupt_headers() {
        let store = UpdatableIndex::open(MemoryDirectory::arc(), 2, 3).unwrap();
        let index = b"index-bytes";
        let bytes = store.encode_sidecar(index).unwrap();
        assert_eq!(store.decode_sidecar(&bytes), Some(index.as_slice()));

        assert!(store.decode_sidecar(&bytes[..8]).is_none());

        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 0xFF;
        assert!(store.decode_sidecar(&bad_magic).is_none());

        let mut bad_version = bytes.clone();
        bad_version[8..12].copy_from_slice(&(SIDECAR_VERSION + 1).to_le_bytes());
        assert!(store.decode_sidecar(&bad_version).is_none());

        let mut bad_recipe_len = bytes.clone();
        bad_recipe_len[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(store.decode_sidecar(&bad_recipe_len).is_none());

        let mut bad_recipe = bytes.clone();
        bad_recipe[16] ^= 0x01;
        assert!(store.decode_sidecar(&bad_recipe).is_none());
    }

    #[test]
    fn gramdex_sidecar_invalid_payload_rebuilds() {
        let dir = MemoryDirectory::arc();
        let (name, _) = checkpointed_store(dir.clone(), 3);
        {
            let store = UpdatableIndex::open(dir.clone(), 2, 3).unwrap();
            let corrupt = store.encode_sidecar(b"not-a-postcard-gramdex").unwrap();
            store.inner.dir().atomic_write(&name, &corrupt).unwrap();
        }

        let store = UpdatableIndex::open(dir.clone(), 2, 3).unwrap();
        let seg_id = store.inner.segment_ids()[0];
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_none(),
            "valid envelope with invalid GramDex payload is rejected"
        );
        assert!(
            !store.candidates(C).is_empty(),
            "invalid payload falls back to rebuild"
        );
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_some(),
            "rebuilt sidecar loads after the fallback"
        );
    }

    #[test]
    fn deleted_id_does_not_resurface_through_a_stale_sidecar() {
        let dir = MemoryDirectory::arc();
        let (name, stale_sidecar) = checkpointed_store(dir.clone(), 3);
        {
            let mut store = UpdatableIndex::open(dir.clone(), 2, 3).unwrap();
            store.delete(2).unwrap();
            store.checkpoint().unwrap();
            store
                .inner
                .dir()
                .atomic_write(&name, &stale_sidecar)
                .unwrap();
        }

        let store = UpdatableIndex::open(dir, 2, 3).unwrap();
        let seg_id = store.inner.segment_ids()[0];
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_none(),
            "stale sidecar id set is rejected"
        );
        let c = store.candidates(C);
        assert!(
            !c.contains(&2),
            "deleted id 2 must not resurface from a stale sidecar"
        );
        assert!(c.contains(&3), "live candidate should remain searchable");
    }

    #[test]
    fn snapshot_index_queries_sidecars_without_opening_segment_payloads() {
        let dir = MemoryDirectory::arc();
        {
            let mut store = UpdatableIndex::open(dir.clone(), 2, 3).unwrap();
            store.add(1, A).unwrap();
            store.add(2, B).unwrap();
            store.add(3, C).unwrap();
            store.add(4, D).unwrap();
            store.checkpoint().unwrap();
        }

        let (watched, opened) = RecordingDirectory::wrap(dir);
        let snapshot = SnapshotIndex::open(watched, 3).unwrap();
        assert_eq!(snapshot.segment_count(), 2);
        assert_eq!(snapshot.tombstone_count(), 0);
        assert_eq!(snapshot.candidates_min_shared(C, 3).unwrap(), vec![2, 3]);

        let opened = opened.lock().unwrap().clone();
        assert!(
            opened.iter().any(|path| path.starts_with("segstore.idx.")),
            "snapshot should open persisted sidecars: {opened:?}"
        );
        assert!(
            !opened.iter().any(|path| path.starts_with("segstore.seg.")),
            "valid sidecars should avoid source segment payload reads: {opened:?}"
        );
    }

    #[test]
    fn snapshot_index_filters_tombstones_from_stale_sidecar_without_source_read() {
        let dir = MemoryDirectory::arc();
        let (name, stale_sidecar) = checkpointed_store(dir.clone(), 3);
        {
            let mut store = UpdatableIndex::open(dir.clone(), 2, 3).unwrap();
            store.delete(2).unwrap();
            store.checkpoint().unwrap();
            store
                .inner
                .dir()
                .atomic_write(&name, &stale_sidecar)
                .unwrap();
        }

        let (watched, opened) = RecordingDirectory::wrap(dir);
        let snapshot = SnapshotIndex::open(watched, 3).unwrap();
        assert_eq!(snapshot.tombstone_count(), 1);
        assert_eq!(snapshot.candidates_min_shared(C, 3).unwrap(), vec![3]);

        let opened = opened.lock().unwrap().clone();
        assert!(
            opened.iter().any(|path| path.starts_with("segstore.idx.")),
            "snapshot should use the stale sidecar before applying tombstones: {opened:?}"
        );
        assert!(
            !opened.iter().any(|path| path.starts_with("segstore.seg.")),
            "tombstone filtering should not require source segment payload reads: {opened:?}"
        );
    }

    #[test]
    fn snapshot_index_rebuilds_missing_sidecar_from_one_segment() {
        let dir = MemoryDirectory::arc();
        let (name, _) = checkpointed_store(dir.clone(), 3);
        dir.delete(&name).unwrap();

        let (watched, opened) = RecordingDirectory::wrap(dir.clone());
        let snapshot = SnapshotIndex::open(watched, 3).unwrap();
        assert_eq!(snapshot.candidates_min_shared(C, 3).unwrap(), vec![2, 3]);
        assert!(
            dir.exists(&name),
            "snapshot fallback should persist the rebuilt sidecar"
        );

        let opened = opened.lock().unwrap().clone();
        assert!(
            opened.iter().any(|path| path.starts_with("segstore.seg.")),
            "missing sidecar should fall back to one source segment read: {opened:?}"
        );
    }
}
