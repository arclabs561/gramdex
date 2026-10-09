//! Union candidates never miss a document that shares a trigram with the query.
//!
//! Callers verify candidates afterwards, so a false positive costs time but a
//! false negative silently drops a match. This checks the contract on many
//! random short strings over a small alphabet, where shared and repeated
//! trigrams are common, against a brute-force scan of every document.

use std::collections::HashSet;

use gramdex::{char_trigrams, DocId, GramDex};

/// Small deterministic generator so the test needs no extra dependency.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn string(&mut self, max_len: u64) -> String {
        const ALPHABET: &[char] = &['a', 'b', 'c', 'é'];
        let len = self.next() % (max_len + 1);
        (0..len)
            .map(|_| ALPHABET[(self.next() % ALPHABET.len() as u64) as usize])
            .collect()
    }
}

fn shared_trigrams(a: &str, b: &str) -> usize {
    let a: HashSet<String> = char_trigrams(a).into_iter().collect();
    let b: HashSet<String> = char_trigrams(b).into_iter().collect();
    a.intersection(&b).count()
}

#[test]
fn union_and_min_shared_candidates_contain_every_true_match() {
    let mut rng = Lcg(0x5eed);
    for _round in 0..50 {
        let docs: Vec<String> = (0..40).map(|_| rng.string(8)).collect();
        let mut index = GramDex::new();
        for (id, text) in docs.iter().enumerate() {
            index.add_document_trigrams(id as DocId, text);
        }

        for _query in 0..20 {
            let query = rng.string(8);
            let union: HashSet<DocId> = index
                .candidates_union_trigrams(&query)
                .into_iter()
                .collect();
            for min_shared in 1..=3u32 {
                let pruned: HashSet<DocId> = index
                    .candidates_union_trigrams_min_shared(&query, min_shared)
                    .into_iter()
                    .collect();
                for (id, text) in docs.iter().enumerate() {
                    let shared = shared_trigrams(&query, text);
                    let id = id as DocId;
                    if shared >= 1 {
                        assert!(
                            union.contains(&id),
                            "union missed doc {id} {text:?} for query {query:?}"
                        );
                    }
                    if shared >= min_shared as usize {
                        assert!(
                            pruned.contains(&id),
                            "min_shared={min_shared} missed doc {id} {text:?} \
                             ({shared} shared) for query {query:?}"
                        );
                    }
                }
            }
        }
    }
}
