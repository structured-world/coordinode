use super::*;

fn parse(text: &str) -> Result<Node, QuerySyntaxError> {
    let mut parser = Parser {
        lexemes: lex(text)?,
        at: 0,
    };
    let node = parser.or()?;
    if parser.at < parser.lexemes.len() {
        return Err(QuerySyntaxError("trailing".into()));
    }
    Ok(node)
}

fn word(text: &str) -> Node {
    Node::Word {
        text: text.into(),
        prefix: false,
        fuzzy: None,
        boost: None,
    }
}

/// `AND` binds tighter than `OR`, words side by side are alternatives,
/// `NOT` applies to the unit after it and parentheses group.
#[test]
fn operators_group_by_precedence() {
    assert_eq!(
        parse("raft AND (consensus OR paxos) NOT zookeeper").unwrap(),
        Node::Or(vec![
            Node::And(vec![
                word("raft"),
                Node::Or(vec![word("consensus"), word("paxos")]),
            ]),
            Node::Not(Box::new(word("zookeeper"))),
        ])
    );
    assert_eq!(
        parse("a b AND c").unwrap(),
        Node::Or(vec![word("a"), Node::And(vec![word("b"), word("c")])])
    );
}

/// Modifiers attach to their word or phrase: prefix, fuzzy distance (2 when
/// left out), phrase slop and boost.
#[test]
fn modifiers_attach_to_their_unit() {
    assert_eq!(
        parse("konsensus~2^3").unwrap(),
        Node::Word {
            text: "konsensus".into(),
            prefix: false,
            fuzzy: Some(2),
            boost: Some(3.0),
        }
    );
    assert_eq!(
        parse("konsensus~").unwrap(),
        Node::Word {
            text: "konsensus".into(),
            prefix: false,
            fuzzy: Some(2),
            boost: None,
        }
    );
    assert_eq!(
        parse("distribut*").unwrap(),
        Node::Word {
            text: "distribut".into(),
            prefix: true,
            fuzzy: None,
            boost: None,
        }
    );
    assert_eq!(
        parse("\"raft consensus\"~2^1.5").unwrap(),
        Node::Phrase {
            text: "raft consensus".into(),
            slop: 2,
            boost: Some(1.5),
        }
    );
}

/// Malformed queries are refused with a reason, not searched as words.
#[test]
fn malformed_queries_are_refused() {
    for bad in [
        "(raft",
        "raft)",
        "\"raft consensus",
        "raft AND",
        "NOT",
        "konsensus~3",
        "konsensus~x",
        "raft^x",
        "data*~1",
        "\"raft\"x",
    ] {
        assert!(parse(bad).is_err(), "{bad:?} was accepted");
    }
}
