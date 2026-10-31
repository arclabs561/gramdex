//! `SnapshotIndex` must not lose checkpointed documents to a sidecar that the
//! writer persisted for a later, uncheckpointed live set.
#![cfg(feature = "store")]

use durability::{Directory, MemoryDirectory};
use gramdex::store::{SnapshotIndex, UpdatableIndex};

/// `SnapshotIndex` documents that it shows the *last checkpoint* and that
/// WAL records after it are not visible. `UpdatableIndex` persists a
/// per-segment sidecar at query time (before any checkpoint), so a
/// post-checkpoint delete must not leak into the snapshot as a missing
/// document: a false negative from a candidate generator that promises none.
#[test]
fn g1_snapshot_misses_checkpointed_doc_after_uncheckpointed_delete() {
    let dir: std::sync::Arc<dyn Directory> = MemoryDirectory::arc();
    {
        let mut store = UpdatableIndex::open(dir.clone(), 2, 3).unwrap();
        store.add(1, "hello").unwrap();
        store.add(2, "yellow").unwrap();
        store.checkpoint().unwrap();

        // Baseline: snapshot of this checkpoint finds doc 1.
        let snap = SnapshotIndex::open(dir.clone(), 3).unwrap();
        assert!(snap.candidates("hello").unwrap().contains(&1));

        store.delete(1).unwrap(); // WAL only, no checkpoint
        let _ = store.candidates("hello"); // rebuilds + persists sidecar without doc 1
    } // "crash": no checkpoint after the delete

    let snap = SnapshotIndex::open(dir, 3).unwrap();
    assert_eq!(snap.tombstone_count(), 0, "delete is not in the checkpoint");
    let c = snap.candidates("hello").unwrap();
    assert!(
        c.contains(&1),
        "snapshot of last checkpoint lost live doc 1 (tombstone_count=0): candidates={c:?}"
    );
}
