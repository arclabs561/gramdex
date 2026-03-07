//! `gramdex`: k-gram indexing primitives for approximate string matching.
//!
//! This crate is about **candidate generation** for fuzzy matching:
//! - build an index mapping grams -> candidate document ids (or string ids)
//! - query by grams to get a bounded candidate set
//! - verify candidates with an exact checker (edit distance / substring / etc.)
//!
//! Tokenization policy for grams matters. This crate provides a **Unicode-scalar**
//! (Rust `char`) k-gram helper as a safe default. Callers can supply their own
//! gram stream if they need byte-grams or grapheme clusters.

#![warn(missing_docs)]

use std::collections::{HashMap, HashSet};

/// Document id type.
pub type DocId = u32;

/// Errors for gram indexing.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// k must be >= 1.
    #[error("k must be >= 1")]
    InvalidK,
}

/// Planner output for candidate generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidatePlan {
    /// Use the returned candidates as a search set.
    Candidates(Vec<DocId>),
    /// Bail out: query is too broad, caller should scan all documents.
    ScanAll,
}

/// Configuration for candidate planning / bailout.
#[derive(Debug, Clone, Copy)]
pub struct PlannerConfig {
    /// If an upper bound on candidates exceeds this ratio of the corpus, bail out.
    pub max_candidate_ratio: f32,
    /// If an upper bound on candidates exceeds this absolute count, bail out.
    pub max_candidates: u32,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            max_candidate_ratio: 0.6,
            max_candidates: 200_000,
        }
    }
}

/// Produce Unicode-scalar k-grams (sliding window over Rust `char`).
///
/// # Examples
///
/// ```
/// let grams = gramdex::char_kgrams("hello", 3).unwrap();
/// assert_eq!(grams, vec!["hel", "ell", "llo"]);
///
/// // Short input produces no grams
/// assert!(gramdex::char_kgrams("hi", 3).unwrap().is_empty());
/// ```
pub fn char_kgrams(text: &str, k: usize) -> Result<Vec<String>, Error> {
    if k == 0 {
        return Err(Error::InvalidK);
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < k {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(chars.len().saturating_sub(k) + 1);
    for w in chars.windows(k) {
        out.push(w.iter().collect::<String>());
    }
    Ok(out)
}

/// Produce Unicode-scalar trigrams (a convenience wrapper over [`char_kgrams`]).
///
/// # Examples
///
/// ```
/// let tris = gramdex::char_trigrams("test");
/// assert_eq!(tris, vec!["tes", "est"]);
/// ```
pub fn char_trigrams(text: &str) -> Vec<String> {
    // k=3 is always valid; avoid exposing Result in the common case.
    char_kgrams(text, 3).expect("k=3 is valid")
}

fn char_trigram_tuples(text: &str) -> Vec<[char; 3]> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 3 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(chars.len() - 2);
    for w in chars.windows(3) {
        out.push([w[0], w[1], w[2]]);
    }
    out
}

fn set_sizes_inter_union<T: Ord>(mut a: Vec<T>, mut b: Vec<T>) -> (usize, usize) {
    // Treat the inputs as sets (sort + dedup), then compute |A ∩ B| and |A ∪ B|.
    a.sort_unstable();
    a.dedup();
    b.sort_unstable();
    b.dedup();

    let mut i = 0usize;
    let mut j = 0usize;
    let mut inter = 0usize;
    let mut union = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                union += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                union += 1;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                inter += 1;
                union += 1;
                i += 1;
                j += 1;
            }
        }
    }
    union += (a.len() - i) + (b.len() - j);
    (inter, union)
}

/// Exact trigram Jaccard similarity over Unicode-scalar trigrams.
///
/// This is intended as a lightweight **verification** primitive for candidates
/// produced by [`GramDex`]. It is **not** a tokenizer; callers should normalize
/// case/whitespace/etc. before calling if they need that behavior.
///
/// Convention:
/// - If both inputs have zero trigrams (length < 3), returns 1.0.
/// - If only one input has trigrams, returns 0.0.
///
/// # Examples
///
/// ```
/// let sim = gramdex::trigram_jaccard("hello", "hello");
/// assert!((sim - 1.0).abs() < 1e-6);
///
/// let sim = gramdex::trigram_jaccard("hello", "world");
/// assert!(sim < 0.5);
/// ```
pub fn trigram_jaccard(a: &str, b: &str) -> f32 {
    let a_tris = char_trigram_tuples(a);
    let b_tris = char_trigram_tuples(b);

    if a_tris.is_empty() && b_tris.is_empty() {
        return 1.0;
    }
    if a_tris.is_empty() || b_tris.is_empty() {
        return 0.0;
    }

    let (inter, union) = set_sizes_inter_union(a_tris, b_tris);
    // union can be 0 only if both sets are empty, which we handled above.
    (inter as f32) / (union as f32)
}

/// A minimal grams->docs candidate index.
#[derive(Debug, Default)]
pub struct GramDex {
    // gram -> docs containing gram
    grams: HashMap<String, HashSet<DocId>>,
    // all indexed docs (for ScanAll fallback)
    docs: HashSet<DocId>,
}

impl GramDex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert grams for a document id.
    pub fn add_document(&mut self, doc_id: DocId, grams: &[String]) {
        self.docs.insert(doc_id);
        for g in grams {
            self.grams.entry(g.clone()).or_default().insert(doc_id);
        }
    }

    /// Insert Unicode-scalar k-grams for a document id.
    pub fn add_document_char_kgrams(
        &mut self,
        doc_id: DocId,
        text: &str,
        k: usize,
    ) -> Result<(), Error> {
        let grams = char_kgrams(text, k)?;
        self.add_document(doc_id, &grams);
        Ok(())
    }

    /// Insert Unicode-scalar trigrams for a document id.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut ix = gramdex::GramDex::new();
    /// ix.add_document_trigrams(0, "hello");
    /// ix.add_document_trigrams(1, "world");
    /// assert_eq!(ix.num_docs(), 2);
    /// ```
    pub fn add_document_trigrams(&mut self, doc_id: DocId, text: &str) {
        let grams = char_trigrams(text);
        self.add_document(doc_id, &grams);
    }

    /// Number of indexed documents.
    pub fn num_docs(&self) -> u32 {
        self.docs.len() as u32
    }

    /// Document frequency for a gram (number of docs containing it).
    pub fn df(&self, gram: &str) -> u32 {
        self.grams.get(gram).map(|ds| ds.len() as u32).unwrap_or(0)
    }

    /// Iterate all known document ids.
    pub fn document_ids(&self) -> impl Iterator<Item = DocId> + '_ {
        self.docs.iter().copied()
    }

    /// Candidate docs: union of docs that share at least one query gram.
    ///
    /// This is intentionally permissive (no false negatives for the grams set),
    /// but may include many false positives; callers should verify.
    pub fn candidates_union(&self, query_grams: &[String]) -> Vec<DocId> {
        let mut out: HashSet<DocId> = HashSet::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for g in query_grams {
            if !seen.insert(g.as_str()) {
                continue;
            }
            if let Some(ds) = self.grams.get(g) {
                out.extend(ds.iter().copied());
            }
        }
        let mut v: Vec<DocId> = out.into_iter().collect();
        v.sort_unstable();
        v
    }

    /// Convenience: union candidates for Unicode-scalar k-grams of `text`.
    pub fn candidates_union_char_kgrams(&self, text: &str, k: usize) -> Result<Vec<DocId>, Error> {
        let grams = char_kgrams(text, k)?;
        Ok(self.candidates_union(&grams))
    }

    /// Convenience: bounded union candidates for Unicode-scalar k-grams of `text`.
    pub fn candidates_union_char_kgrams_bounded(
        &self,
        text: &str,
        k: usize,
        cfg: PlannerConfig,
    ) -> Result<Vec<DocId>, Error> {
        let grams = char_kgrams(text, k)?;
        Ok(self.candidates_union_bounded(&grams, cfg))
    }

    /// Convenience: union candidates for Unicode-scalar trigrams of `text`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut ix = gramdex::GramDex::new();
    /// ix.add_document_trigrams(0, "hello");
    /// ix.add_document_trigrams(1, "yellow");
    ///
    /// let cands = ix.candidates_union_trigrams("mellow");
    /// assert!(cands.contains(&1)); // "yellow" shares trigrams with "mellow"
    /// ```
    pub fn candidates_union_trigrams(&self, text: &str) -> Vec<DocId> {
        let grams = char_trigrams(text);
        self.candidates_union(&grams)
    }

    /// Convenience: bounded union candidates for Unicode-scalar trigrams of `text`.
    pub fn candidates_union_trigrams_bounded(&self, text: &str, cfg: PlannerConfig) -> Vec<DocId> {
        let grams = char_trigrams(text);
        self.candidates_union_bounded(&grams, cfg)
    }

    /// Candidate docs with an overlap count: number of **distinct** query grams
    /// that appear in each document.
    ///
    /// This is useful for cheap pruning before expensive verification:
    /// `min_shared = 2` often removes many candidates vs plain union.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut ix = gramdex::GramDex::new();
    /// ix.add_document_trigrams(0, "abcdef");
    /// ix.add_document_trigrams(1, "abcxyz");
    ///
    /// let grams = gramdex::char_trigrams("abcde");
    /// let scored = ix.candidates_union_scored(&grams);
    /// // doc 0 shares more trigrams than doc 1
    /// assert!(scored[0].0 == 0 && scored[0].1 >= 2);
    /// ```
    pub fn candidates_union_scored(&self, query_grams: &[String]) -> Vec<(DocId, u32)> {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut counts: HashMap<DocId, u32> = HashMap::new();
        for g in query_grams {
            if !seen.insert(g.as_str()) {
                continue;
            }
            if let Some(ds) = self.grams.get(g) {
                for &doc in ds {
                    *counts.entry(doc).or_insert(0) += 1;
                }
            }
        }
        let mut v: Vec<(DocId, u32)> = counts.into_iter().collect();
        // Highest shared-grams first; tie-break by doc_id for determinism.
        v.sort_unstable_by(|(a_id, a_c), (b_id, b_c)| b_c.cmp(a_c).then_with(|| a_id.cmp(b_id)));
        v
    }

    /// Candidate docs that share at least `min_shared` distinct query grams.
    pub fn candidates_union_min_shared(
        &self,
        query_grams: &[String],
        min_shared: u32,
    ) -> Vec<DocId> {
        if min_shared <= 1 {
            return self.candidates_union(query_grams);
        }
        let mut v: Vec<DocId> = self
            .candidates_union_scored(query_grams)
            .into_iter()
            .filter_map(|(doc_id, shared)| (shared >= min_shared).then_some(doc_id))
            .collect();
        v.sort_unstable();
        v
    }

    /// Convenience: candidates that share at least `min_shared` trigrams with `text`.
    pub fn candidates_union_trigrams_min_shared(&self, text: &str, min_shared: u32) -> Vec<DocId> {
        let grams = char_trigrams(text);
        self.candidates_union_min_shared(&grams, min_shared)
    }

    /// Plan candidate generation with bailout thresholds.
    ///
    /// This uses the **actual** union size (not the loose upper bound
    /// \( \sum_g df(g) \)), which avoids false `ScanAll` decisions when grams
    /// overlap heavily. It is slightly more work than a pure `df`-based bound,
    /// but it reuses the work needed to produce the final candidate set.
    pub fn plan_candidates_union(
        &self,
        query_grams: &[String],
        cfg: PlannerConfig,
    ) -> CandidatePlan {
        if query_grams.is_empty() {
            return CandidatePlan::Candidates(Vec::new());
        }
        let n = self.num_docs();
        if n == 0 {
            return CandidatePlan::Candidates(Vec::new());
        }

        let mut seen: HashSet<&str> = HashSet::new();
        let mut out: HashSet<DocId> = HashSet::new();
        for g in query_grams {
            if !seen.insert(g.as_str()) {
                continue;
            }
            if let Some(ds) = self.grams.get(g) {
                out.extend(ds.iter().copied());

                if out.len() >= cfg.max_candidates as usize {
                    return CandidatePlan::ScanAll;
                }

                let ratio = (out.len() as f32) / (n as f32);
                if ratio > cfg.max_candidate_ratio {
                    return CandidatePlan::ScanAll;
                }
            }
        }

        let mut v: Vec<DocId> = out.into_iter().collect();
        v.sort_unstable();
        CandidatePlan::Candidates(v)
    }

    /// Candidates with a fallback to scanning all docs when too broad.
    pub fn candidates_union_bounded(
        &self,
        query_grams: &[String],
        cfg: PlannerConfig,
    ) -> Vec<DocId> {
        match self.plan_candidates_union(query_grams, cfg) {
            CandidatePlan::Candidates(c) => c,
            CandidatePlan::ScanAll => {
                let mut v: Vec<DocId> = self.document_ids().collect();
                v.sort_unstable();
                v
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kgrams_are_unicode_safe() {
        // "é" can be one char here; we just want to ensure no byte slicing.
        let grams = char_kgrams("café", 2).unwrap();
        assert!(!grams.is_empty());
    }

    #[test]
    fn trigram_jaccard_conventions() {
        assert_eq!(trigram_jaccard("hi", "yo"), 1.0);
        assert_eq!(trigram_jaccard("hi", "hiya"), 0.0);
        assert!((trigram_jaccard("test", "test") - 1.0).abs() < 1e-6);
        assert!(trigram_jaccard("abcd", "abce") > 0.0);
    }

    #[test]
    fn gramdex_candidates_union_works() {
        let mut ix = GramDex::new();
        ix.add_document(1, &char_kgrams("hello", 3).unwrap());
        ix.add_document(2, &char_kgrams("yellow", 3).unwrap());

        let qs = char_kgrams("mellow", 3).unwrap();
        let cands = ix.candidates_union(&qs);
        assert!(cands.contains(&2));
    }

    #[test]
    fn query_helpers_work() {
        let mut ix = GramDex::new();
        ix.add_document_trigrams(1, "hello");
        ix.add_document_trigrams(2, "yellow");

        let cands = ix.candidates_union_trigrams("mellow");
        assert!(cands.contains(&2));

        let cands2 = ix.candidates_union_char_kgrams("mellow", 3).unwrap();
        assert!(cands2.contains(&2));
    }

    #[test]
    fn gramdex_candidates_union_scored_prunes() {
        let mut ix = GramDex::new();
        ix.add_document(1, &char_kgrams("abcdefgh", 3).unwrap());
        ix.add_document(2, &char_kgrams("abcxyzhh", 3).unwrap());
        ix.add_document(3, &char_kgrams("zzzabcqq", 3).unwrap());

        let qs = char_kgrams("abcde", 3).unwrap(); // "abc","bcd","cde"
        let scored = ix.candidates_union_scored(&qs);
        assert!(!scored.is_empty());

        // doc 1 shares at least "abc","bcd","cde" => should be top-ish
        assert_eq!(scored[0].0, 1);
        assert!(scored[0].1 >= 2);

        let pruned = ix.candidates_union_min_shared(&qs, 2);
        assert!(pruned.contains(&1));
        // doc 3 likely only shares "abc"
        assert!(!pruned.contains(&3));
    }

    #[test]
    fn gramdex_can_bail_out() {
        let mut ix = GramDex::new();
        // Make a very common gram across many docs.
        for i in 0..100u32 {
            ix.add_document(i, &["aaa".to_string(), format!("u{i}")]);
        }
        let plan = ix.plan_candidates_union(
            &[String::from("aaa")],
            PlannerConfig {
                max_candidate_ratio: 0.2,
                max_candidates: 10,
            },
        );
        assert_eq!(plan, CandidatePlan::ScanAll);
    }

    #[test]
    fn plan_candidates_union_avoids_df_sum_false_bailout() {
        let mut ix = GramDex::new();
        let grams: Vec<String> = (0..100).map(|i| format!("g{i}")).collect();
        ix.add_document(1, &grams);

        // Old df-sum bound would be 100 (>= 10) and would bail. Actual union is 1.
        let plan = ix.plan_candidates_union(
            &grams,
            PlannerConfig {
                max_candidate_ratio: 1.0,
                max_candidates: 10,
            },
        );
        assert_eq!(plan, CandidatePlan::Candidates(vec![1]));
    }

    #[test]
    fn add_document_trigrams_matches_manual() {
        let mut ix1 = GramDex::new();
        ix1.add_document_trigrams(1, "hello");

        let mut ix2 = GramDex::new();
        ix2.add_document(1, &char_trigrams("hello"));

        assert_eq!(
            ix1.candidates_union(&char_trigrams("hello")),
            ix2.candidates_union(&char_trigrams("hello"))
        );
        assert_eq!(ix1.df("hel"), 1);
    }
}
