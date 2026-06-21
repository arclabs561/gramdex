use gramdex::{trigram_jaccard, GramDex};
use std::collections::BTreeMap;

fn main() {
    let docs = BTreeMap::from([
        (0, "vector quantization for nearest neighbor search"),
        (1, "sparse vectors for learned retrieval"),
        (2, "graph walks for retrieval planning"),
        (3, "nearest neighbour spelling variant"),
    ]);

    let mut index = GramDex::new();
    for (doc_id, text) in &docs {
        index.add_document_trigrams(*doc_id, text);
    }

    let query = "nearest neighbor";
    let query_grams = gramdex::char_trigrams(query);

    let mut ranked: Vec<_> = index
        .candidates_union_scored(&query_grams)
        .into_iter()
        .filter_map(|(doc_id, shared)| {
            let text = docs.get(&doc_id)?;
            let verified = trigram_jaccard(query, text);
            (verified >= 0.10).then_some((doc_id, shared, verified, *text))
        })
        .collect();

    ranked.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| a.0.cmp(&b.0))
    });

    println!("query: {query:?}");
    for (doc_id, shared, verified, text) in ranked {
        println!("doc {doc_id}: shared={shared}, jaccard={verified:.3}, text={text:?}");
    }
}
