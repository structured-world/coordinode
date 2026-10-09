//! The query language of a text search, analyzed as the documents were.
//!
//! - words, any of them matching (`raft consensus`);
//! - `"exact phrases"`, with `"…"~N` allowing N positions of slop;
//! - `AND`, `OR` and `NOT` (upper case), grouped with parentheses; `AND`
//!   binds tighter than `OR`, and words side by side are alternatives;
//! - `word*` for every word starting with `word`;
//! - `word~N` for words within N edits (insertion, deletion, substitution or
//!   a swap of neighbours) of `word`, N up to 2, `word~` meaning 2;
//! - `^B` after a word or phrase to weigh its score by B.
//!
//! Every word and phrase goes through the language pipeline the documents
//! were indexed with, so a query word meets the stemmed form it was stored
//! under.

use tantivy::Term;
use tantivy::query::{
    AllQuery, BooleanQuery, BoostQuery, FuzzyTermQuery, Occur, PhraseQuery, Query, TermQuery,
};
use tantivy::schema::{Field, IndexRecordOption};

use super::tokenize;

/// The largest edit distance a fuzzy word accepts.
const MAX_FUZZY_DISTANCE: u8 = 2;

/// A malformed query, with what is wrong in it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("text query syntax: {0}")]
pub struct QuerySyntaxError(pub String);

/// One parsed query.
#[derive(Debug, Clone, PartialEq)]
enum Node {
    Word {
        text: String,
        prefix: bool,
        fuzzy: Option<u8>,
        boost: Option<f32>,
    },
    Phrase {
        text: String,
        slop: u32,
        boost: Option<f32>,
    },
    And(Vec<Node>),
    Or(Vec<Node>),
    Not(Box<Node>),
}

#[derive(Debug, Clone, PartialEq)]
enum Lexeme {
    Open,
    Close,
    And,
    Or,
    Not,
    Word(String),
    Phrase(String),
}

/// The query for `text` over `field`, its words analyzed by `language`;
/// `prefix` builds a query of the words starting with a stem. `None` when
/// the query leaves nothing to search for (only stop words, say).
///
/// # Errors
///
/// An unbalanced parenthesis or quote, an operator with nothing to apply
/// to, or a malformed `~` or `^` modifier.
pub(crate) fn build(
    text: &str,
    field: Field,
    language: &str,
    prefix: &dyn Fn(&str) -> Box<dyn Query>,
) -> Result<Option<Box<dyn Query>>, QuerySyntaxError> {
    let lexemes = lex(text)?;
    if lexemes.is_empty() {
        return Ok(None);
    }
    let mut parser = Parser { lexemes, at: 0 };
    let node = parser.or()?;
    if parser.at < parser.lexemes.len() {
        return Err(QuerySyntaxError(format!(
            "unexpected {:?} after the query",
            parser.lexemes[parser.at]
        )));
    }
    let builder = Builder {
        field,
        language,
        prefix,
    };
    Ok(match builder.query(&node)? {
        Some(Built::Positive(query)) => Some(query),
        Some(Built::Negative(query)) => Some(all_but(query)),
        None => None,
    })
}

/// Split `text` into parentheses, operators, words and phrases.
fn lex(text: &str) -> Result<Vec<Lexeme>, QuerySyntaxError> {
    let mut out = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c == '(' {
            chars.next();
            out.push(Lexeme::Open);
        } else if c == ')' {
            chars.next();
            out.push(Lexeme::Close);
        } else if c == '"' {
            chars.next();
            let mut phrase = String::new();
            loop {
                match chars.next() {
                    Some('"') => break,
                    Some(c) => phrase.push(c),
                    None => return Err(QuerySyntaxError("a phrase is not closed".into())),
                }
            }
            // A modifier follows the closing quote directly.
            let mut modifiers = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() || c == '(' || c == ')' || c == '"' {
                    break;
                }
                modifiers.push(c);
                chars.next();
            }
            out.push(Lexeme::Phrase(format!("\"{phrase}\"{modifiers}")));
        } else {
            let mut word = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() || c == '(' || c == ')' || c == '"' {
                    break;
                }
                word.push(c);
                chars.next();
            }
            out.push(match word.as_str() {
                "AND" | "&&" => Lexeme::And,
                "OR" | "||" => Lexeme::Or,
                "NOT" | "!" => Lexeme::Not,
                _ => Lexeme::Word(word),
            });
        }
    }
    Ok(out)
}

struct Parser {
    lexemes: Vec<Lexeme>,
    at: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Lexeme> {
        self.lexemes.get(self.at)
    }

    /// Alternatives: `AND` groups joined by `OR` or by nothing at all.
    fn or(&mut self) -> Result<Node, QuerySyntaxError> {
        let mut parts = vec![self.and()?];
        loop {
            match self.peek() {
                Some(Lexeme::Or) => {
                    self.at += 1;
                    parts.push(self.and()?);
                }
                Some(Lexeme::Close) | None => break,
                Some(_) => parts.push(self.and()?),
            }
        }
        Ok(if parts.len() == 1 {
            parts.remove(0)
        } else {
            Node::Or(parts)
        })
    }

    /// Requirements: units joined by `AND`.
    fn and(&mut self) -> Result<Node, QuerySyntaxError> {
        let mut parts = vec![self.unary()?];
        while self.peek() == Some(&Lexeme::And) {
            self.at += 1;
            parts.push(self.unary()?);
        }
        Ok(if parts.len() == 1 {
            parts.remove(0)
        } else {
            Node::And(parts)
        })
    }

    fn unary(&mut self) -> Result<Node, QuerySyntaxError> {
        match self.peek().cloned() {
            Some(Lexeme::Not) => {
                self.at += 1;
                Ok(Node::Not(Box::new(self.unary()?)))
            }
            Some(Lexeme::Open) => {
                self.at += 1;
                let inner = self.or()?;
                if self.peek() != Some(&Lexeme::Close) {
                    return Err(QuerySyntaxError("a parenthesis is not closed".into()));
                }
                self.at += 1;
                Ok(inner)
            }
            Some(Lexeme::Word(word)) => {
                self.at += 1;
                word_node(&word)
            }
            Some(Lexeme::Phrase(phrase)) => {
                self.at += 1;
                phrase_node(&phrase)
            }
            Some(other) => Err(QuerySyntaxError(format!(
                "{other:?} has nothing to apply to"
            ))),
            None => Err(QuerySyntaxError(
                "the query ends where a word is expected".into(),
            )),
        }
    }
}

/// `text` with a trailing `^B` taken off, and B.
fn take_boost(text: &str) -> Result<(&str, Option<f32>), QuerySyntaxError> {
    let Some((head, boost)) = text.rsplit_once('^') else {
        return Ok((text, None));
    };
    let boost: f32 = boost
        .parse()
        .ok()
        .filter(|b: &f32| b.is_finite() && *b >= 0.0)
        .ok_or_else(|| QuerySyntaxError(format!("`^{boost}` is not a weight")))?;
    Ok((head, Some(boost)))
}

fn word_node(word: &str) -> Result<Node, QuerySyntaxError> {
    let (word, boost) = take_boost(word)?;
    let (word, fuzzy) = match word.rsplit_once('~') {
        Some((head, "")) => (head, Some(MAX_FUZZY_DISTANCE)),
        Some((head, distance)) => {
            let distance: u8 = distance
                .parse()
                .ok()
                .filter(|d| *d <= MAX_FUZZY_DISTANCE)
                .ok_or_else(|| {
                    QuerySyntaxError(format!(
                        "`~{distance}`: a fuzzy word takes a distance from 0 to {MAX_FUZZY_DISTANCE}"
                    ))
                })?;
            (head, Some(distance))
        }
        None => (word, None),
    };
    let (word, prefix) = match word.strip_suffix('*') {
        Some(stem) if !stem.is_empty() => (stem, true),
        _ => (word, false),
    };
    if prefix && fuzzy.is_some() {
        return Err(QuerySyntaxError(format!(
            "`{word}`: a word is either a prefix or fuzzy"
        )));
    }
    Ok(Node::Word {
        text: word.to_string(),
        prefix,
        fuzzy,
        boost,
    })
}

fn phrase_node(phrase: &str) -> Result<Node, QuerySyntaxError> {
    // `"text"` followed by its modifiers, as the lexer kept it.
    let (text, modifiers) = phrase[1..]
        .rsplit_once('"')
        .ok_or_else(|| QuerySyntaxError("a phrase is not closed".into()))?;
    let (modifiers, boost) = take_boost(modifiers)?;
    let slop = match modifiers.strip_prefix('~') {
        Some(slop) => slop
            .parse()
            .map_err(|_| QuerySyntaxError(format!("`~{slop}` is not a phrase slop")))?,
        None if modifiers.is_empty() => 0,
        None => {
            return Err(QuerySyntaxError(format!(
                "`{modifiers}` after a phrase is not a modifier"
            )));
        }
    };
    Ok(Node::Phrase {
        text: text.to_string(),
        slop,
        boost,
    })
}

/// A query built from a node: one that matches, or one that excludes.
enum Built {
    Positive(Box<dyn Query>),
    Negative(Box<dyn Query>),
}

struct Builder<'a> {
    field: Field,
    language: &'a str,
    prefix: &'a dyn Fn(&str) -> Box<dyn Query>,
}

impl Builder<'_> {
    fn query(&self, node: &Node) -> Result<Option<Built>, QuerySyntaxError> {
        Ok(match node {
            Node::Word {
                text,
                prefix,
                fuzzy,
                boost,
            } => self
                .word(text, *prefix, *fuzzy)
                .map(|q| Built::Positive(boosted(q, *boost))),
            Node::Phrase { text, slop, boost } => self
                .phrase(text, *slop)
                .map(|q| Built::Positive(boosted(q, *boost))),
            Node::Not(inner) => match self.query(inner)? {
                Some(Built::Positive(q)) => Some(Built::Negative(q)),
                // Excluding an exclusion keeps what it excluded.
                Some(Built::Negative(q)) => Some(Built::Positive(q)),
                None => None,
            },
            Node::And(parts) => self.group(parts, Occur::Must)?,
            Node::Or(parts) => self.group(parts, Occur::Should)?,
        })
    }

    /// `parts` combined with `occur`; an excluded part excludes from the
    /// whole group, as `a AND NOT b` and `a NOT b` both mean.
    fn group(&self, parts: &[Node], occur: Occur) -> Result<Option<Built>, QuerySyntaxError> {
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        let mut positive = 0usize;
        for part in parts {
            match self.query(part)? {
                Some(Built::Positive(q)) => {
                    positive += 1;
                    clauses.push((occur, q));
                }
                Some(Built::Negative(q)) => clauses.push((Occur::MustNot, q)),
                None => {}
            }
        }
        if clauses.is_empty() {
            return Ok(None);
        }
        if positive == 0 {
            // Only exclusions: they exclude from everything.
            let excluded: Vec<(Occur, Box<dyn Query>)> = clauses
                .into_iter()
                .map(|(_, q)| (Occur::Should, q))
                .collect();
            return Ok(Some(Built::Negative(Box::new(BooleanQuery::new(excluded)))));
        }
        if clauses.len() == 1 {
            return Ok(clauses.pop().map(|(_, q)| Built::Positive(q)));
        }
        Ok(Some(Built::Positive(Box::new(BooleanQuery::new(clauses)))))
    }

    fn word(&self, text: &str, prefix: bool, fuzzy: Option<u8>) -> Option<Box<dyn Query>> {
        if prefix {
            return Some((self.prefix)(&text.to_lowercase()));
        }
        // A fuzzy word is lower-cased, not stemmed, as Lucene and
        // Elasticsearch leave multi-term queries unanalyzed: a stemmer cuts a
        // misspelled word where it would not cut the right one, which would
        // move it further from the stored form than the typo did.
        let tokens = match fuzzy {
            Some(_) => tokenize::tokenize_text(text, "none"),
            None => tokenize::tokenize_text(text, self.language),
        };
        match (fuzzy, tokens.as_slice()) {
            (_, []) => None,
            (Some(distance), tokens) => {
                let fuzzies: Vec<(Occur, Box<dyn Query>)> = tokens
                    .iter()
                    .map(|token| {
                        let term = Term::from_field_text(self.field, &token.text);
                        let query: Box<dyn Query> =
                            Box::new(FuzzyTermQuery::new(term, distance, true));
                        (Occur::Must, query)
                    })
                    .collect();
                Some(single_or_all(fuzzies))
            }
            // A word the analyzer splits (a compound, a CJK run) is the
            // sequence of its parts.
            (None, tokens) => Some(self.sequence(tokens, 0)),
        }
    }

    fn phrase(&self, text: &str, slop: u32) -> Option<Box<dyn Query>> {
        let tokens = tokenize::tokenize_text(text, self.language);
        if tokens.is_empty() {
            return None;
        }
        Some(self.sequence(&tokens, slop))
    }

    /// The tokens in their order, `slop` positions apart at most; one token
    /// is a term.
    fn sequence(&self, tokens: &[tantivy::tokenizer::Token], slop: u32) -> Box<dyn Query> {
        if let [token] = tokens {
            let term = Term::from_field_text(self.field, &token.text);
            return Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs));
        }
        // Positions grow along the text, so none is below the first.
        let first = tokens[0].position;
        let terms: Vec<(usize, Term)> = tokens
            .iter()
            .map(|t| {
                (
                    t.position - first,
                    Term::from_field_text(self.field, &t.text),
                )
            })
            .collect();
        let mut phrase = PhraseQuery::new_with_offset(terms);
        phrase.set_slop(slop);
        Box::new(phrase)
    }
}

fn boosted(query: Box<dyn Query>, boost: Option<f32>) -> Box<dyn Query> {
    match boost {
        Some(boost) => Box::new(BoostQuery::new(query, boost)),
        None => query,
    }
}

fn single_or_all(mut clauses: Vec<(Occur, Box<dyn Query>)>) -> Box<dyn Query> {
    if clauses.len() == 1 {
        return clauses.remove(0).1;
    }
    Box::new(BooleanQuery::new(clauses))
}

/// Every document except those `excluded` matches.
fn all_but(excluded: Box<dyn Query>) -> Box<dyn Query> {
    Box::new(BooleanQuery::new(vec![
        (Occur::Must, Box::new(AllQuery) as Box<dyn Query>),
        (Occur::MustNot, excluded),
    ]))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
