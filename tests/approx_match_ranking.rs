//! Integration test: approximate-match retrieval and ranking by k-gram overlap.
//!
//! Invariant under test: for a query string, `GramDex` retrieves exactly the
//! documents that share at least one trigram with the query, and
//! `candidates_union_scored` ranks them by descending count of distinct shared
//! trigrams. A document with zero trigram overlap is excluded from the
//! candidate set entirely (the negative case), and the `trigram_jaccard`
//! verifier orders survivors consistently with that overlap.
//!
//! The corpus is hand-built so the answer is unambiguous. Query "kitten"
//! against {"kitten", "mitten", "kitchen", "banana"} has these char-trigram
//! sets (verified by hand):
//!   query "kitten"  -> {kit, itt, tte, ten}
//!   "kitten" (exact)-> {kit, itt, tte, ten}            shared=4, jaccard=1.000
//!   "mitten"        -> {mit, itt, tte, ten}            shared=3, jaccard=0.600
//!   "kitchen"       -> {kit, itc, tch, che, hen}       shared=1, jaccard=0.125
//!   "banana"        -> {ban, ana, nan}                 shared=0, jaccard=0.000
//! so the strict ranking is kitten > mitten > kitchen, with banana excluded.

use gramdex::{trigram_jaccard, GramDex};

const KITTEN: gramdex::DocId = 0;
const MITTEN: gramdex::DocId = 1;
const KITCHEN: gramdex::DocId = 2;
const BANANA: gramdex::DocId = 3;

fn build_index() -> GramDex {
    let mut ix = GramDex::new();
    ix.add_document_trigrams(KITTEN, "kitten");
    ix.add_document_trigrams(MITTEN, "mitten");
    ix.add_document_trigrams(KITCHEN, "kitchen");
    ix.add_document_trigrams(BANANA, "banana");
    ix
}

#[test]
fn retrieval_includes_overlapping_excludes_disjoint() {
    let ix = build_index();
    let cands = ix.candidates_union_trigrams("kitten");

    // Every document sharing >=1 trigram is retrieved.
    assert!(cands.contains(&KITTEN), "exact match must be a candidate");
    assert!(
        cands.contains(&MITTEN),
        "one-typo match must be a candidate"
    );
    assert!(
        cands.contains(&KITCHEN),
        "prefix-sharing match must be a candidate"
    );

    // Negative case: zero trigram overlap => excluded from the candidate set.
    assert!(
        !cands.contains(&BANANA),
        "banana shares no trigram with 'kitten' and must not be a candidate"
    );
    assert_eq!(cands.len(), 3, "exactly the three overlapping docs");
}

#[test]
fn ranking_is_by_distinct_shared_trigram_count() {
    let ix = build_index();
    let grams = gramdex::char_trigrams("kitten");
    let scored = ix.candidates_union_scored(&grams);

    // Full, exact ranking on the hand-built case: descending shared-gram count.
    // banana (shared=0) is absent, not merely ranked last.
    assert_eq!(
        scored,
        vec![(KITTEN, 4), (MITTEN, 3), (KITCHEN, 1)],
        "ranking must be kitten(4) > mitten(3) > kitchen(1), banana excluded"
    );
}

#[test]
fn min_shared_prunes_weak_overlap() {
    let ix = build_index();

    // Requiring >=2 shared trigrams keeps the close matches, drops kitchen
    // (shared=1) and banana (shared=0).
    let pruned = ix.candidates_union_trigrams_min_shared("kitten", 2);
    assert!(pruned.contains(&KITTEN));
    assert!(pruned.contains(&MITTEN));
    assert!(!pruned.contains(&KITCHEN), "kitchen shares only 1 trigram");
    assert!(!pruned.contains(&BANANA), "banana shares 0 trigrams");
}

#[test]
fn jaccard_verifier_orders_candidates_consistently() {
    // The verification primitive ranks survivors in the same order as the
    // k-gram overlap, and reports exactly 0.0 for the disjoint negative case.
    let exact = trigram_jaccard("kitten", "kitten");
    let typo = trigram_jaccard("kitten", "mitten");
    let prefix = trigram_jaccard("kitten", "kitchen");
    let disjoint = trigram_jaccard("kitten", "banana");

    assert!((exact - 1.0).abs() < 1e-6, "exact match is similarity 1.0");
    assert!(
        exact > typo && typo > prefix && prefix > disjoint,
        "jaccard ordering must match overlap ordering: 1.0 > {typo} > {prefix} > {disjoint}"
    );
    assert_eq!(disjoint, 0.0, "zero-overlap pair has jaccard 0.0");
}
