//! Semantic parity between indexed candidate paths and a naive trigram oracle.

use std::collections::HashSet;

use gramdex::{char_trigrams, DocId, GramDex};

const CORPUS: &[&str] = &[
    "kitten",
    "mitten",
    "kitchen",
    "banana",
    "aaaaa",
    "café",
    "咖啡店",
    "👩‍💻rust",
    "hi",
    "",
];

fn distinct_trigrams(text: &str) -> HashSet<String> {
    char_trigrams(text).into_iter().collect()
}

fn naive_scores(query: &str) -> Vec<(DocId, u32)> {
    let query_grams = distinct_trigrams(query);
    let mut scores: Vec<_> = CORPUS
        .iter()
        .enumerate()
        .filter_map(|(doc_id, text)| {
            let shared = query_grams.intersection(&distinct_trigrams(text)).count() as u32;
            (shared > 0).then_some((doc_id as DocId, shared))
        })
        .collect();
    scores.sort_unstable_by(|(a_id, a_score), (b_id, b_score)| {
        b_score.cmp(a_score).then_with(|| a_id.cmp(b_id))
    });
    scores
}

fn build_index() -> GramDex {
    let mut index = GramDex::new();
    for (doc_id, text) in CORPUS.iter().enumerate() {
        index.add_document_trigrams(doc_id as DocId, text);
    }
    index
}

#[test]
fn indexed_candidate_paths_match_naive_distinct_trigram_oracle() {
    let index = build_index();
    let queries = [
        "kitten",
        "smitten",
        "aaaaaa",
        "café",
        "咖啡馆",
        "👩‍💻rustacean",
        "xy",
        "",
    ];

    for query in queries {
        let grams = char_trigrams(query);
        let expected_scores = naive_scores(query);
        let expected_ids: Vec<_> = expected_scores.iter().map(|(id, _)| *id).collect();

        assert_eq!(
            index.candidates_union_scored(&grams),
            expected_scores,
            "scored candidates diverged for {query:?}"
        );

        let mut expected_union = expected_ids.clone();
        expected_union.sort_unstable();
        assert_eq!(
            index.candidates_union(&grams),
            expected_union,
            "union candidates diverged for {query:?}"
        );

        for min_shared in 0..=4 {
            let mut expected_pruned: Vec<_> = expected_scores
                .iter()
                .filter_map(|(id, shared)| (*shared >= min_shared.max(1)).then_some(*id))
                .collect();
            expected_pruned.sort_unstable();
            assert_eq!(
                index.candidates_union_min_shared(&grams, min_shared),
                expected_pruned,
                "min_shared={min_shared} diverged for {query:?}"
            );
        }
    }
}
