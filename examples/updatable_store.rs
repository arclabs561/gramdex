//! Updatable k-gram store: add, checkpoint, delete, reopen, and verify.
//!
//! Run: cargo run --features store --example updatable_store

use std::collections::BTreeMap;

use durability::MemoryDirectory;
use gramdex::{store::UpdatableIndex, trigram_jaccard};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let docs = BTreeMap::from([
        (1, "hello"),
        (2, "yellow"),
        (3, "mellow"),
        (4, "a separate unrelated document"),
    ]);
    let dir = MemoryDirectory::arc();

    {
        let mut index = UpdatableIndex::open(dir.clone(), 2, 3)?;
        index.extend(docs.iter().map(|(id, text)| (*id, (*text).to_string())))?;
        index.checkpoint()?;

        let candidates = index.candidates("mellow");
        let strong = index.candidates_min_shared("mellow", 3);

        println!("before delete:");
        println!("  candidates: {candidates:?}");
        println!("  min_shared=3: {strong:?}");

        assert_eq!(candidates, vec![1, 2, 3]);
        assert_eq!(strong, vec![2, 3]);

        index.delete(2)?;
        index.checkpoint()?;
    }

    let recovered = UpdatableIndex::open(dir, 2, 3)?;
    let candidates = recovered.candidates("mellow");
    let strong = recovered.candidates_min_shared("mellow", 3);
    let verified = verify("mellow", &candidates, &docs, 0.3);

    println!("after reopen:");
    println!("  candidates: {candidates:?}");
    println!("  min_shared=3: {strong:?}");
    for (doc_id, score, text) in &verified {
        println!("  verified doc {doc_id}: jaccard={score:.3}, text={text:?}");
    }

    assert_eq!(candidates, vec![1, 3]);
    assert_eq!(strong, vec![3]);
    assert_eq!(
        verified.iter().map(|(id, _, _)| *id).collect::<Vec<_>>(),
        vec![3, 1]
    );

    Ok(())
}

fn verify<'a>(
    query: &str,
    candidates: &[u32],
    docs: &'a BTreeMap<u32, &'a str>,
    min_score: f32,
) -> Vec<(u32, f32, &'a str)> {
    let mut verified: Vec<_> = candidates
        .iter()
        .filter_map(|doc_id| {
            let text = docs.get(doc_id)?;
            let score = trigram_jaccard(query, text);
            (score >= min_score).then_some((*doc_id, score, *text))
        })
        .collect();
    verified.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    verified
}
