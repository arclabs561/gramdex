use gramdex::{trigram_jaccard, GramDex};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let sample = "vector quantization for nearest neighbor search\nsparse vectors for learned retrieval\ngraph walks for retrieval planning\nnearest neighbour spelling variant";
    let input;
    let (query, corpus) = match args.as_slice() {
        [] => ("nearest neighbor", sample),
        [query, path] => {
            input = std::fs::read_to_string(path).map_err(|e| format!("opening {path:?}: {e}"))?;
            (query.as_str(), input.as_str())
        }
        _ => return Err("usage: candidate_verify_rank [QUERY CORPUS.txt]".into()),
    };
    let docs: Vec<_> = corpus.lines().collect();
    let mut index = GramDex::new();
    for (id, text) in docs.iter().enumerate() {
        index.add_document_trigrams(id.try_into()?, text);
    }
    let query_grams = gramdex::char_trigrams(query);

    let mut ranked: Vec<_> = index
        .candidates_union_scored(&query_grams)
        .into_iter()
        .filter_map(|(doc_id, shared)| {
            let text = docs.get(doc_id as usize)?;
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
    Ok(())
}
