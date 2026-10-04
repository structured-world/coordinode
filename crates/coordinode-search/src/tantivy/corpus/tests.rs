use super::*;

fn toks(words: &[&str]) -> Vec<String> {
    words.iter().map(|w| w.to_string()).collect()
}

/// A document counts once toward each distinct term it holds and by its
/// position count toward the tokens; removing it undoes exactly that.
#[test]
fn add_and_remove_are_exact_inverses() {
    let mut corpus = Corpus::default();
    corpus.add(&toks(&["rust", "graph", "rust"]));
    corpus.add(&toks(&["graph"]));
    assert_eq!((corpus.docs, corpus.tokens), (2, 4));
    assert_eq!(corpus.df.get("rust"), Some(&1));
    assert_eq!(corpus.df.get("graph"), Some(&2));

    corpus.remove(&toks(&["rust", "graph", "rust"]));
    assert_eq!((corpus.docs, corpus.tokens), (1, 1));
    assert_eq!(
        corpus.df.get("rust"),
        None,
        "a term no document holds is gone"
    );
    assert_eq!(corpus.df.get("graph"), Some(&1));
}

/// A change applies to a base statistic and refuses one it would take below
/// zero.
#[test]
fn a_change_applies_and_refuses_underflow() {
    assert_eq!(CorpusChange::apply(10, 2, 3), Some(9));
    assert_eq!(CorpusChange::apply(1, 0, 2), None);
}
