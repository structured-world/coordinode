use super::*;

// -- Utility --

fn parse_ok(input: &str) -> Query {
    parse(input).unwrap_or_else(|e| panic!("parse failed for {input:?}: {e}"))
}

fn parse_err(input: &str) -> ParseError {
    parse(input).unwrap_err()
}

// -- Basic MATCH --

#[test]
fn simple_match_return() {
    let q = parse_ok("MATCH (n) RETURN n");
    assert_eq!(q.clauses.len(), 2);
    assert!(matches!(q.clauses[0], Clause::Match(_)));
    assert!(matches!(q.clauses[1], Clause::Return(_)));
}

#[test]
fn match_with_label() {
    let q = parse_ok("MATCH (n:User) RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        if let PatternElement::Node(ref np) = m.patterns[0].elements[0] {
            assert_eq!(np.labels, vec!["User"]);
            assert_eq!(np.variable, Some("n".to_string()));
        } else {
            panic!("expected node pattern");
        }
    } else {
        panic!("expected MATCH clause");
    }
}

#[test]
fn match_with_multiple_labels() {
    let q = parse_ok("MATCH (n:User:Admin) RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        if let PatternElement::Node(ref np) = m.patterns[0].elements[0] {
            assert_eq!(np.labels, vec!["User", "Admin"]);
        } else {
            panic!("expected node pattern");
        }
    } else {
        panic!("expected MATCH clause");
    }
}

#[test]
fn match_with_properties() {
    let q = parse_ok("MATCH (n:User {name: 'Alice', age: 30}) RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        if let PatternElement::Node(ref np) = m.patterns[0].elements[0] {
            assert_eq!(np.properties.len(), 2);
            assert_eq!(np.properties[0].0, "name");
            assert_eq!(np.properties[1].0, "age");
        } else {
            panic!("expected node pattern");
        }
    } else {
        panic!("expected MATCH clause");
    }
}

// -- Relationships --

#[test]
fn match_outgoing_relationship() {
    let q = parse_ok("MATCH (a)-[r:KNOWS]->(b) RETURN a, b");
    if let Clause::Match(ref m) = q.clauses[0] {
        assert_eq!(m.patterns[0].elements.len(), 3);
        if let PatternElement::Relationship(ref rp) = m.patterns[0].elements[1] {
            assert_eq!(rp.direction, Direction::Outgoing);
            assert_eq!(rp.rel_types, vec!["KNOWS"]);
            assert_eq!(rp.variable, Some("r".to_string()));
        } else {
            panic!("expected relationship pattern");
        }
    } else {
        panic!("expected MATCH clause");
    }
}

#[test]
fn match_incoming_relationship() {
    let q = parse_ok("MATCH (a)<-[:FOLLOWS]-(b) RETURN a");
    if let Clause::Match(ref m) = q.clauses[0] {
        if let PatternElement::Relationship(ref rp) = m.patterns[0].elements[1] {
            assert_eq!(rp.direction, Direction::Incoming);
            assert_eq!(rp.rel_types, vec!["FOLLOWS"]);
        } else {
            panic!("expected relationship pattern");
        }
    } else {
        panic!("expected MATCH clause");
    }
}

#[test]
fn match_undirected_relationship() {
    let q = parse_ok("MATCH (a)-[:KNOWS]-(b) RETURN a");
    if let Clause::Match(ref m) = q.clauses[0] {
        if let PatternElement::Relationship(ref rp) = m.patterns[0].elements[1] {
            assert_eq!(rp.direction, Direction::Both);
        } else {
            panic!("expected relationship pattern");
        }
    } else {
        panic!("expected MATCH clause");
    }
}

#[test]
fn match_variable_length_path() {
    let q = parse_ok("MATCH (a)-[:KNOWS*2..5]->(b) RETURN a, b");
    if let Clause::Match(ref m) = q.clauses[0] {
        if let PatternElement::Relationship(ref rp) = m.patterns[0].elements[1] {
            let lb = rp.length.unwrap();
            assert_eq!(lb.min, Some(2));
            assert_eq!(lb.max, Some(5));
        } else {
            panic!("expected relationship pattern");
        }
    } else {
        panic!("expected MATCH clause");
    }
}

// -- WHERE --

#[test]
fn match_where() {
    let q = parse_ok("MATCH (n:User) WHERE n.age > 25 RETURN n");
    // WHERE is folded into MATCH
    if let Clause::Match(ref m) = q.clauses[0] {
        assert!(m.where_clause.is_some());
    } else {
        panic!("expected MATCH clause");
    }
}

#[test]
fn where_and_or() {
    let q = parse_ok("MATCH (n) WHERE n.age > 25 AND n.name = 'Alice' RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        assert!(matches!(
            w,
            Expr::BinaryOp {
                op: BinaryOperator::And,
                ..
            }
        ));
    } else {
        panic!("expected MATCH");
    }
}

#[test]
fn where_starts_with() {
    let q = parse_ok("MATCH (n) WHERE n.name STARTS WITH 'A' RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        assert!(matches!(
            w,
            Expr::StringMatch {
                op: StringOp::StartsWith,
                ..
            }
        ));
    } else {
        panic!("expected MATCH");
    }
}

#[test]
fn where_is_null() {
    let q = parse_ok("MATCH (n) WHERE n.email IS NULL RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        assert!(matches!(w, Expr::IsNull { negated: false, .. }));
    } else {
        panic!("expected MATCH");
    }
}

#[test]
fn where_is_not_null() {
    let q = parse_ok("MATCH (n) WHERE n.email IS NOT NULL RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        assert!(matches!(w, Expr::IsNull { negated: true, .. }));
    } else {
        panic!("expected MATCH");
    }
}

#[test]
fn where_in_list() {
    let q = parse_ok("MATCH (n) WHERE n.status IN ['active', 'pending'] RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        assert!(matches!(w, Expr::In { .. }));
    } else {
        panic!("expected MATCH");
    }
}

// -- RETURN --

#[test]
fn return_with_alias() {
    let q = parse_ok("MATCH (n) RETURN n.name AS username");
    if let Clause::Return(ref rc) = q.clauses[1] {
        assert_eq!(rc.items.len(), 1);
        assert_eq!(rc.items[0].alias, Some("username".to_string()));
    } else {
        panic!("expected RETURN clause");
    }
}

#[test]
fn return_star() {
    let q = parse_ok("MATCH (n) RETURN *");
    if let Clause::Return(ref rc) = q.clauses[1] {
        assert!(matches!(rc.items[0].expr, Expr::Star));
    } else {
        panic!("expected RETURN clause");
    }
}

#[test]
fn return_distinct() {
    let q = parse_ok("MATCH (n) RETURN DISTINCT n.name");
    if let Clause::Return(ref rc) = q.clauses[1] {
        assert!(rc.distinct);
    } else {
        panic!("expected RETURN clause");
    }
}

#[test]
fn return_multiple() {
    let q = parse_ok("MATCH (n) RETURN n.name, n.age, n.email");
    if let Clause::Return(ref rc) = q.clauses[1] {
        assert_eq!(rc.items.len(), 3);
    } else {
        panic!("expected RETURN clause");
    }
}

// -- Aggregation --

#[test]
fn count_star() {
    let q = parse_ok("MATCH (n:User) RETURN count(*)");
    if let Clause::Return(ref rc) = q.clauses[1] {
        if let Expr::FunctionCall {
            ref name, ref args, ..
        } = rc.items[0].expr
        {
            assert_eq!(name, "count");
            assert!(matches!(args[0], Expr::Star));
        } else {
            panic!("expected function call");
        }
    } else {
        panic!("expected RETURN clause");
    }
}

#[test]
fn aggregation_with_grouping() {
    let q =
        parse_ok("MATCH (n:User) RETURN n.city AS city, count(*) AS cnt, avg(n.age) AS avg_age");
    if let Clause::Return(ref rc) = q.clauses[1] {
        assert_eq!(rc.items.len(), 3);
        assert_eq!(rc.items[0].alias, Some("city".to_string()));
        assert_eq!(rc.items[1].alias, Some("cnt".to_string()));
        assert_eq!(rc.items[2].alias, Some("avg_age".to_string()));
    } else {
        panic!("expected RETURN clause");
    }
}

// -- WITH --

#[test]
fn with_clause() {
    let q = parse_ok("MATCH (n:User) WITH n, count(*) AS cnt WHERE cnt > 5 RETURN n.name");
    assert!(q.clauses.iter().any(|c| matches!(c, Clause::With(_))));
}

// -- UNWIND --

#[test]
fn unwind_clause() {
    let q = parse_ok("UNWIND [1, 2, 3] AS x RETURN x");
    if let Clause::Unwind(ref uc) = q.clauses[0] {
        assert_eq!(uc.variable, "x");
        assert!(matches!(uc.expr, Expr::List(_)));
    } else {
        panic!("expected UNWIND clause");
    }
}

// -- Scientific notation in float literals --

#[test]
fn float_scientific_with_dot() {
    // 1.5e3 = 1500.0
    let q = parse_ok("RETURN 1.5e3 AS x");
    if let Clause::Return(ref rc) = q.clauses[0] {
        assert!(
            matches!(rc.items[0].expr, Expr::Literal(Value::Float(f)) if (f - 1500.0).abs() < 0.001),
            "expected 1500.0, got {:?}",
            rc.items[0].expr
        );
    }
}

#[test]
fn float_scientific_no_dot() {
    // 7e-05 = 0.00007
    let q = parse_ok("RETURN 7e-05 AS x");
    if let Clause::Return(ref rc) = q.clauses[0] {
        assert!(
            matches!(rc.items[0].expr, Expr::Literal(Value::Float(f)) if (f - 0.00007).abs() < 1e-10),
            "expected 0.00007, got {:?}",
            rc.items[0].expr
        );
    }
}

#[test]
fn float_scientific_positive_exponent() {
    // 3E+2 = 300.0
    let q = parse_ok("RETURN 3E+2 AS x");
    if let Clause::Return(ref rc) = q.clauses[0] {
        assert!(
            matches!(rc.items[0].expr, Expr::Literal(Value::Float(f)) if (f - 300.0).abs() < 0.001),
            "expected 300.0, got {:?}",
            rc.items[0].expr
        );
    }
}

#[test]
fn float_scientific_in_where() {
    // Scientific notation usable in WHERE predicates
    let q = parse_ok("MATCH (n) WHERE n.val > 1.5e-3 RETURN n");
    assert!(!q.clauses.is_empty());
}

// -- ORDER BY / SKIP / LIMIT (embedded in RETURN) --

#[test]
fn order_by_skip_limit() {
    let q = parse_ok("MATCH (n) RETURN n.name ORDER BY n.name DESC SKIP 10 LIMIT 25");
    assert!(q.clauses.iter().any(|c| matches!(c, Clause::OrderBy(_))));
    assert!(q.clauses.iter().any(|c| matches!(c, Clause::Skip(_))));
    assert!(q.clauses.iter().any(|c| matches!(c, Clause::Limit(_))));

    if let Some(Clause::OrderBy(ref items)) =
        q.clauses.iter().find(|c| matches!(c, Clause::OrderBy(_)))
    {
        assert_eq!(items.len(), 1);
        assert!(!items[0].ascending);
    }
}

// -- OPTIONAL MATCH --

#[test]
fn optional_match() {
    let q = parse_ok("MATCH (n:User) OPTIONAL MATCH (n)-[:KNOWS]->(m) RETURN n, m");
    assert!(matches!(q.clauses[0], Clause::Match(_)));
    assert!(matches!(q.clauses[1], Clause::OptionalMatch(_)));
}

// -- Parameters --

#[test]
fn parameter_in_where() {
    let q = parse_ok("MATCH (n) WHERE n.id = $userId RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        if let Expr::BinaryOp { right, .. } = w {
            assert!(matches!(**right, Expr::Parameter(ref s) if s == "userId"));
        } else {
            panic!("expected binary op");
        }
    }
}

#[test]
fn parameter_in_properties() {
    let q = parse_ok("MATCH (n:User {email: $email}) RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        if let PatternElement::Node(ref np) = m.patterns[0].elements[0] {
            assert!(matches!(np.properties[0].1, Expr::Parameter(ref s) if s == "email"));
        }
    }
}

// -- Vector functions --

#[test]
fn vector_distance_function() {
    let q =
        parse_ok("MATCH (m:Movie) WHERE vector_distance(m.embedding, $query_vec) < 0.3 RETURN m");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        if let Expr::BinaryOp { left, op, .. } = w {
            assert_eq!(*op, BinaryOperator::Lt);
            if let Expr::FunctionCall { name, args, .. } = left.as_ref() {
                assert_eq!(name, "vector_distance");
                assert_eq!(args.len(), 2);
            } else {
                panic!("expected function call");
            }
        } else {
            panic!("expected binary op");
        }
    }
}

// -- AS OF TIMESTAMP --

#[test]
fn as_of_timestamp() {
    let q = parse_ok("MATCH (n:User) RETURN n AS OF TIMESTAMP '2025-06-15T10:00:00Z'");
    assert!(q
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::AsOfTimestamp(_))));
}

// -- Complex queries --

#[test]
fn graph_plus_vector_query() {
    let q = parse_ok(
        "MATCH (u:User {id: $me})-[:LIKES]->(m:Movie) \
             WHERE vector_distance(m.embedding, $query_vec) < 0.3 \
             WITH m.genre AS genre, count(*) AS cnt \
             ORDER BY cnt DESC \
             LIMIT 10 \
             RETURN genre, cnt",
    );
    assert!(q.clauses.len() >= 4);
}

#[test]
fn multiple_patterns() {
    let q = parse_ok("MATCH (a:User), (b:Movie) WHERE a.id = 1 RETURN a, b");
    if let Clause::Match(ref m) = q.clauses[0] {
        assert_eq!(m.patterns.len(), 2);
    }
}

#[test]
fn shortest_path_named_pattern_parses() {
    let q = parse_ok("MATCH p = shortestPath((a:Person)-[:KNOWS*]->(b:Person)) RETURN length(p)");
    if let Clause::Match(ref m) = q.clauses[0] {
        assert_eq!(m.patterns.len(), 1);
        let p = &m.patterns[0];
        assert!(p.shortest_path, "pattern must be flagged shortest_path");
        assert_eq!(p.path_variable.as_deref(), Some("p"));
        // (a)-[:KNOWS*]->(b): node, relationship, node.
        assert_eq!(p.elements.len(), 3);
    } else {
        panic!("expected MATCH clause");
    }
}

#[test]
fn named_path_without_shortest_path_parses() {
    // `p = (a)-[:KNOWS]->(b)` binds a path variable on a plain pattern.
    let q = parse_ok("MATCH p = (a)-[:KNOWS]->(b) RETURN p");
    if let Clause::Match(ref m) = q.clauses[0] {
        let p = &m.patterns[0];
        assert!(!p.shortest_path);
        assert_eq!(p.path_variable.as_deref(), Some("p"));
    } else {
        panic!("expected MATCH clause");
    }
}

// -- Arithmetic --

#[test]
fn arithmetic_expression() {
    let q = parse_ok("MATCH (n) RETURN n.price * 1.1 + 5 AS total");
    if let Clause::Return(ref rc) = q.clauses[1] {
        assert_eq!(rc.items[0].alias, Some("total".to_string()));
    }
}

// -- Literals --

#[test]
fn string_escape() {
    let q = parse_ok("MATCH (n) WHERE n.name = 'it\\'s' RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        if let Expr::BinaryOp { right, .. } = w {
            assert_eq!(**right, Expr::Literal(Value::String("it's".to_string())));
        }
    }
}

#[test]
fn string_literal_preserves_edge_whitespace() {
    // A whitespace-padded (or whitespace-only) string literal must keep its
    // spaces — the grammar's compound-atomic rule stops implicit WHITESPACE
    // from stripping them. Regression: ' ' previously parsed to "".
    let q = parse_ok("MATCH (n) WHERE n.name = '  hi ' RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        if let Expr::BinaryOp { right, .. } = w {
            assert_eq!(**right, Expr::Literal(Value::String("  hi ".to_string())));
        } else {
            panic!("expected binary op");
        }
    } else {
        panic!("expected match clause");
    }

    let q = parse_ok("MATCH (n) WHERE n.sep = ' ' RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        if let Expr::BinaryOp { right, .. } = w {
            assert_eq!(**right, Expr::Literal(Value::String(" ".to_string())));
        } else {
            panic!("expected binary op");
        }
    } else {
        panic!("expected match clause");
    }
}

#[test]
fn boolean_literals() {
    let q = parse_ok("MATCH (n) WHERE n.active = true RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        if let Expr::BinaryOp { right, .. } = w {
            assert_eq!(**right, Expr::Literal(Value::Bool(true)));
        }
    }
}

#[test]
fn null_literal() {
    let q = parse_ok("MATCH (n) WHERE n.deleted = null RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        if let Expr::BinaryOp { right, .. } = w {
            assert_eq!(**right, Expr::Literal(Value::Null));
        }
    }
}

// -- CASE --

#[test]
fn case_expression() {
    let q =
        parse_ok("MATCH (n) RETURN CASE WHEN n.age < 18 THEN 'minor' ELSE 'adult' END AS category");
    if let Clause::Return(ref rc) = q.clauses[1] {
        assert!(matches!(rc.items[0].expr, Expr::Case { .. }));
    }
}

// -- Error cases --

#[test]
fn empty_query_fails() {
    let err = parse_err("");
    assert!(matches!(err, ParseError::Syntax { .. }));
}

#[test]
fn invalid_syntax_fails() {
    let err = parse_err("MATCH RETURN n");
    assert!(matches!(
        err,
        ParseError::Syntax { .. } | ParseError::Invalid(_)
    ));
}

#[test]
fn backtick_identifier() {
    let q = parse_ok("MATCH (n:`My Label`) RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        if let PatternElement::Node(ref np) = m.patterns[0].elements[0] {
            assert_eq!(np.labels, vec!["My Label"]);
        }
    }
}

// -- NOT expression --

#[test]
fn not_expression() {
    let q = parse_ok("MATCH (n) WHERE NOT n.active RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        assert!(matches!(
            w,
            Expr::UnaryOp {
                op: UnaryOperator::Not,
                ..
            }
        ));
    }
}

// -- Multiple relationship types --

#[test]
fn multiple_rel_types() {
    let q = parse_ok("MATCH (a)-[:KNOWS|:FOLLOWS]->(b) RETURN a, b");
    if let Clause::Match(ref m) = q.clauses[0] {
        if let PatternElement::Relationship(ref rp) = m.patterns[0].elements[1] {
            assert_eq!(rp.rel_types, vec!["KNOWS", "FOLLOWS"]);
        }
    }
}

// -- count(DISTINCT x) --

#[test]
fn count_distinct() {
    let q = parse_ok("MATCH (n) RETURN count(DISTINCT n.city) AS cities");
    if let Clause::Return(ref rc) = q.clauses[1] {
        if let Expr::FunctionCall {
            ref name, distinct, ..
        } = rc.items[0].expr
        {
            assert_eq!(name, "count");
            assert!(distinct);
        }
    }
}

// -- Negative number --

#[test]
fn negative_number() {
    let q = parse_ok("MATCH (n) WHERE n.temp > -10 RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        if let Expr::BinaryOp { right, .. } = w {
            assert!(matches!(
                **right,
                Expr::UnaryOp {
                    op: UnaryOperator::Neg,
                    ..
                }
            ));
        }
    }
}

// -- Long chain of traversals --

#[test]
fn long_traversal_chain() {
    let q = parse_ok("MATCH (a)-[:KNOWS]->(b)-[:WORKS_AT]->(c) RETURN a, b, c");
    if let Clause::Match(ref m) = q.clauses[0] {
        // a, KNOWS, b, WORKS_AT, c
        assert_eq!(m.patterns[0].elements.len(), 5);
    }
}

// -- ENDS WITH / CONTAINS --

#[test]
fn where_ends_with() {
    let q = parse_ok("MATCH (n) WHERE n.email ENDS WITH '.com' RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        assert!(matches!(
            w,
            Expr::StringMatch {
                op: StringOp::EndsWith,
                ..
            }
        ));
    }
}

#[test]
fn where_contains() {
    let q = parse_ok("MATCH (n) WHERE n.bio CONTAINS 'rust' RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().unwrap();
        assert!(matches!(
            w,
            Expr::StringMatch {
                op: StringOp::Contains,
                ..
            }
        ));
    }
}

// -- Case insensitive keywords --

#[test]
fn case_insensitive_keywords() {
    let q = parse_ok("match (n:User) where n.age > 18 return n.name");
    assert!(matches!(q.clauses[0], Clause::Match(_)));
    assert!(matches!(q.clauses[1], Clause::Return(_)));
}

// -- Empty node pattern --

#[test]
fn empty_node() {
    let q = parse_ok("MATCH () RETURN *");
    if let Clause::Match(ref m) = q.clauses[0] {
        if let PatternElement::Node(ref np) = m.patterns[0].elements[0] {
            assert!(np.variable.is_none());
            assert!(np.labels.is_empty());
        }
    }
}

// -- Map literal in RETURN --

#[test]
fn map_literal_return() {
    let q = parse_ok("MATCH (n) RETURN {name: n.name, age: n.age} AS props");
    if let Clause::Return(ref rc) = q.clauses[1] {
        assert!(matches!(rc.items[0].expr, Expr::MapLiteral(_)));
    }
}

// ====== Write operations ======

// -- CREATE --

#[test]
fn create_node() {
    let q = parse_ok("CREATE (n:User {name: 'Alice', age: 30})");
    if let Clause::Create(ref cc) = q.clauses[0] {
        assert_eq!(cc.patterns.len(), 1);
        if let PatternElement::Node(ref np) = cc.patterns[0].elements[0] {
            assert_eq!(np.labels, vec!["User"]);
            assert_eq!(np.properties.len(), 2);
        } else {
            panic!("expected node pattern");
        }
    } else {
        panic!("expected CREATE clause");
    }
}

#[test]
fn create_node_and_relationship() {
    let q = parse_ok("CREATE (a:User {name: 'Alice'})-[:KNOWS]->(b:User {name: 'Bob'})");
    if let Clause::Create(ref cc) = q.clauses[0] {
        assert_eq!(cc.patterns[0].elements.len(), 3); // a, KNOWS, b
    } else {
        panic!("expected CREATE clause");
    }
}

#[test]
fn create_multiple_patterns() {
    let q = parse_ok("CREATE (a:User), (b:Movie)");
    if let Clause::Create(ref cc) = q.clauses[0] {
        assert_eq!(cc.patterns.len(), 2);
    } else {
        panic!("expected CREATE clause");
    }
}

#[test]
fn match_create() {
    let q =
        parse_ok("MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'}) CREATE (a)-[:KNOWS]->(b)");
    assert!(matches!(q.clauses[0], Clause::Match(_)));
    assert!(matches!(q.clauses[1], Clause::Create(_)));
}

// -- MERGE --

#[test]
fn merge_simple() {
    let q = parse_ok("MERGE (n:User {email: 'alice@example.com'})");
    if let Clause::Merge(ref mc) = q.clauses[0] {
        if let PatternElement::Node(ref np) = mc.pattern.elements[0] {
            assert_eq!(np.labels, vec!["User"]);
            assert_eq!(np.properties.len(), 1);
        } else {
            panic!("expected node pattern");
        }
        assert!(mc.on_match.is_empty());
        assert!(mc.on_create.is_empty());
    } else {
        panic!("expected MERGE clause");
    }
}

#[test]
fn merge_on_match_on_create() {
    let q = parse_ok(
        "MERGE (n:User {email: $email}) \
             ON MATCH SET n.login_count = n.login_count + 1 \
             ON CREATE SET n.login_count = 1, n.created = $now",
    );
    if let Clause::Merge(ref mc) = q.clauses[0] {
        assert_eq!(mc.on_match.len(), 1);
        assert_eq!(mc.on_create.len(), 2);
    } else {
        panic!("expected MERGE clause");
    }
}

#[test]
fn merge_on_create_only() {
    let q = parse_ok("MERGE (n:User {email: $email}) ON CREATE SET n.created = $now");
    if let Clause::Merge(ref mc) = q.clauses[0] {
        assert!(mc.on_match.is_empty());
        assert_eq!(mc.on_create.len(), 1);
    } else {
        panic!("expected MERGE clause");
    }
}

// -- DELETE --

#[test]
fn delete_single() {
    let q = parse_ok("MATCH (n:User {id: 42}) DELETE n");
    if let Clause::Delete(ref dc) = q.clauses[1] {
        assert!(!dc.detach);
        assert_eq!(dc.exprs.len(), 1);
    } else {
        panic!("expected DELETE clause");
    }
}

#[test]
fn detach_delete() {
    let q = parse_ok("MATCH (n:User {id: 42}) DETACH DELETE n");
    if let Clause::Delete(ref dc) = q.clauses[1] {
        assert!(dc.detach);
        assert_eq!(dc.exprs.len(), 1);
    } else {
        panic!("expected DELETE clause");
    }
}

#[test]
fn delete_multiple() {
    let q = parse_ok("MATCH (a)-[r]->(b) DELETE a, r, b");
    if let Clause::Delete(ref dc) = q.clauses[1] {
        assert_eq!(dc.exprs.len(), 3);
    } else {
        panic!("expected DELETE clause");
    }
}

// -- MERGE NODES (R180) --

#[test]
fn merge_nodes_default_keep_first() {
    let q = parse_ok("MATCH (a:User {id: 1}), (b:User {id: 2}) MERGE NODES (a, b) INTO a");
    let mn = match &q.clauses[1] {
        Clause::MergeNodes(mn) => mn,
        other => panic!("expected MergeNodes, got {other:?}"),
    };
    assert_eq!(mn.source_a, "a");
    assert_eq!(mn.source_b, "b");
    assert_eq!(mn.target, "a");
    assert_eq!(mn.conflict, MergeNodesConflictStrategy::KeepFirst);
    assert!(mn.transfer_edges.is_none());
    assert_eq!(mn.duplicate, MergeNodesDuplicateStrategy::KeepBoth);
    assert!(
        mn.transfer_edge_properties,
        "edge properties transfer is on by default per arch spec"
    );
}

#[test]
fn merge_nodes_into_b_target() {
    let q = parse_ok("MATCH (a), (b) MERGE NODES (a, b) INTO b");
    let mn = match &q.clauses[1] {
        Clause::MergeNodes(mn) => mn,
        _ => panic!("expected MergeNodes"),
    };
    assert_eq!(mn.target, "b");
}

#[test]
fn merge_nodes_all_conflict_strategies() {
    for (cypher, expected) in [
        (
            "MATCH (a), (b) MERGE NODES (a, b) INTO a ON CONFLICT KEEP FIRST",
            MergeNodesConflictStrategy::KeepFirst,
        ),
        (
            "MATCH (a), (b) MERGE NODES (a, b) INTO a ON CONFLICT KEEP LAST",
            MergeNodesConflictStrategy::KeepLast,
        ),
        (
            "MATCH (a), (b) MERGE NODES (a, b) INTO a ON CONFLICT COALESCE",
            MergeNodesConflictStrategy::Coalesce,
        ),
    ] {
        let q = parse_ok(cypher);
        let mn = match &q.clauses[1] {
            Clause::MergeNodes(mn) => mn,
            _ => panic!("expected MergeNodes"),
        };
        assert_eq!(mn.conflict, expected, "for input {cypher}");
    }
}

#[test]
fn merge_nodes_on_conflict_set_expressions() {
    let q = parse_ok(
        "MATCH (a), (b) MERGE NODES (a, b) INTO a \
             ON CONFLICT SET a.name = b.name, a.tags = a.tags",
    );
    let mn = match &q.clauses[1] {
        Clause::MergeNodes(mn) => mn,
        _ => panic!("expected MergeNodes"),
    };
    match &mn.conflict {
        MergeNodesConflictStrategy::SetExpressions(items) => {
            assert_eq!(items.len(), 2);
        }
        other => panic!("expected SetExpressions, got {other:?}"),
    }
}

#[test]
fn merge_nodes_with_transfer_and_duplicate() {
    let q = parse_ok(
        "MATCH (a), (b) MERGE NODES (a, b) INTO a \
             TRANSFER EDGES FROM b TO a \
             ON DUPLICATE MERGE PROPERTIES \
             TRANSFER EDGE PROPERTIES",
    );
    let mn = match &q.clauses[1] {
        Clause::MergeNodes(mn) => mn,
        _ => panic!("expected MergeNodes"),
    };
    let t = mn.transfer_edges.as_ref().expect("transfer set");
    assert_eq!(t.src, "b");
    assert_eq!(t.dst, "a");
    assert_eq!(mn.duplicate, MergeNodesDuplicateStrategy::MergeProperties);
    assert!(mn.transfer_edge_properties);
}

#[test]
fn merge_nodes_all_duplicate_strategies() {
    for (cypher, expected) in [
            (
                "MATCH (a), (b) MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a ON DUPLICATE KEEP BOTH",
                MergeNodesDuplicateStrategy::KeepBoth,
            ),
            (
                "MATCH (a), (b) MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a ON DUPLICATE MERGE PROPERTIES",
                MergeNodesDuplicateStrategy::MergeProperties,
            ),
            (
                "MATCH (a), (b) MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a ON DUPLICATE KEEP TARGET",
                MergeNodesDuplicateStrategy::KeepTarget,
            ),
        ] {
            let q = parse_ok(cypher);
            let mn = match &q.clauses[1] {
                Clause::MergeNodes(mn) => mn,
                _ => panic!("expected MergeNodes"),
            };
            assert_eq!(mn.duplicate, expected, "for input {cypher}");
        }
}

#[test]
fn merge_nodes_rejects_unknown_target() {
    let err = parse_err("MATCH (a), (b) MERGE NODES (a, b) INTO c");
    let msg = format!("{err}");
    assert!(
        msg.contains("INTO `c`") && msg.contains("must be one of"),
        "expected target validation error, got: {msg}"
    );
}

#[test]
fn merge_nodes_rejects_same_source_variables() {
    let err = parse_err("MATCH (a) MERGE NODES (a, a) INTO a");
    let msg = format!("{err}");
    assert!(
        msg.contains("distinct source variables"),
        "expected distinctness error, got: {msg}"
    );
}

#[test]
fn merge_nodes_rejects_transfer_dst_not_target() {
    let err = parse_err("MATCH (a), (b) MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO b");
    let msg = format!("{err}");
    assert!(
        msg.contains("TRANSFER EDGES TO `b` must match INTO target `a`"),
        "expected dst==target error, got: {msg}"
    );
}

#[test]
fn merge_nodes_rejects_transfer_src_eq_target() {
    let err = parse_err("MATCH (a), (b) MERGE NODES (a, b) INTO a TRANSFER EDGES FROM a TO a");
    let msg = format!("{err}");
    assert!(
        msg.contains("must be the non-surviving source"),
        "expected non-surviving error, got: {msg}"
    );
}

#[test]
fn merge_nodes_rejects_duplicate_without_transfer() {
    let err = parse_err("MATCH (a), (b) MERGE NODES (a, b) INTO a ON DUPLICATE KEEP BOTH");
    let msg = format!("{err}");
    assert!(
        msg.contains("ON DUPLICATE requires a TRANSFER EDGES clause"),
        "expected requires-transfer error, got: {msg}"
    );
}

// -- CLONE NODE (R182) --

#[test]
fn clone_node_minimal_defaults_copy_properties() {
    let q = parse_ok("MATCH (a:User {id: 1}) CLONE NODE a AS b");
    let cn = match &q.clauses[1] {
        Clause::CloneNode(cn) => cn,
        other => panic!("expected CloneNode, got {other:?}"),
    };
    assert_eq!(cn.source, "a");
    assert_eq!(cn.target, "b");
    assert!(!cn.with_edges, "WITH EDGES is off by default");
    assert!(
        cn.with_properties,
        "properties are copied by default per arch spec"
    );
    assert!(cn.set_items.is_empty());
}

#[test]
fn clone_node_with_edges_and_properties() {
    let q = parse_ok("MATCH (a) CLONE NODE a AS b WITH EDGES WITH PROPERTIES");
    let cn = match &q.clauses[1] {
        Clause::CloneNode(cn) => cn,
        _ => panic!("expected CloneNode"),
    };
    assert!(cn.with_edges);
    assert!(cn.with_properties);
}

#[test]
fn clone_node_with_set_override() {
    let q = parse_ok("MATCH (a) CLONE NODE a AS b SET b.name = 'copy'");
    let cn = match &q.clauses[1] {
        Clause::CloneNode(cn) => cn,
        _ => panic!("expected CloneNode"),
    };
    assert_eq!(cn.set_items.len(), 1, "one SET item parsed");
    match &cn.set_items[0] {
        SetItem::Property {
            variable, property, ..
        } => {
            assert_eq!(variable, "b");
            assert_eq!(property, "name");
        }
        other => panic!("expected Property set item, got {other:?}"),
    }
}

#[test]
fn clone_node_same_source_and_target_rejected() {
    let err = parse_err("MATCH (a) CLONE NODE a AS a");
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("differ"),
        "expected distinct-variable error, got: {msg}"
    );
}

// -- REDIRECT EDGES (R183) --

#[test]
fn redirect_edges_minimal_defaults_both_all_types() {
    let q = parse_ok("MATCH (a), (b) REDIRECT EDGES FROM a TO b");
    let re = match &q.clauses[1] {
        Clause::RedirectEdges(re) => re,
        other => panic!("expected RedirectEdges, got {other:?}"),
    };
    assert_eq!(re.source, "a");
    assert_eq!(re.target, "b");
    assert!(re.edge_types.is_none(), "no filter = all edge types");
    assert_eq!(re.direction, RedirectDirection::Both);
}

#[test]
fn redirect_edges_with_type_filter() {
    let q =
        parse_ok("MATCH (a), (b) REDIRECT EDGES FROM a TO b WHERE type(r) IN ['KNOWS', 'FOLLOWS']");
    let re = match &q.clauses[1] {
        Clause::RedirectEdges(re) => re,
        _ => panic!("expected RedirectEdges"),
    };
    assert_eq!(
        re.edge_types,
        Some(vec!["KNOWS".to_string(), "FOLLOWS".to_string()])
    );
}

#[test]
fn redirect_edges_with_direction() {
    for (cypher, expected) in [
        (
            "MATCH (a), (b) REDIRECT EDGES FROM a TO b DIRECTION OUTGOING",
            RedirectDirection::Outgoing,
        ),
        (
            "MATCH (a), (b) REDIRECT EDGES FROM a TO b DIRECTION INCOMING",
            RedirectDirection::Incoming,
        ),
        (
            "MATCH (a), (b) REDIRECT EDGES FROM a TO b DIRECTION BOTH",
            RedirectDirection::Both,
        ),
    ] {
        let q = parse_ok(cypher);
        match &q.clauses[1] {
            Clause::RedirectEdges(re) => assert_eq!(re.direction, expected, "{cypher}"),
            _ => panic!("expected RedirectEdges for {cypher}"),
        }
    }
}

#[test]
fn redirect_edges_same_source_and_target_rejected() {
    let err = parse_err("MATCH (a) REDIRECT EDGES FROM a TO a");
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("differ"),
        "expected distinct-variable error, got: {msg}"
    );
}

// -- TRIGGER DDL --

#[test]
fn create_trigger_minimal_before_commit() {
    let q = parse_ok(
        "CREATE TRIGGER t1 ON :User CREATE BEFORE COMMIT \
             EXECUTE CREATE (e:AuditEntry {action: $event})",
    );
    let c = match &q.clauses[0] {
        Clause::CreateTrigger(c) => c,
        other => panic!("expected CreateTrigger, got {other:?}"),
    };
    assert_eq!(c.name, "t1");
    assert_eq!(c.target, TriggerTarget::Label("User".into()));
    assert!(c.events.on_create && !c.events.on_update && !c.events.on_delete);
    assert_eq!(c.timing, TriggerTiming::BeforeCommit);
    assert!(c.body_source.starts_with("CREATE"));
    assert!(c.cascade_limit.is_none());
    assert!(c.cascade_fanout.is_none());
    assert!(
        c.on_error.is_none(),
        "no explicit ON ERROR → defaults at executor"
    );
}

#[test]
fn create_trigger_edge_type_target() {
    let q = parse_ok(
        "CREATE TRIGGER t2 ON [:FOLLOWS] UPDATE | DELETE AFTER COMMIT \
             EXECUTE SET n.touched = true",
    );
    let c = match &q.clauses[0] {
        Clause::CreateTrigger(c) => c,
        _ => panic!(),
    };
    assert_eq!(c.target, TriggerTarget::EdgeType("FOLLOWS".into()));
    assert!(c.events.on_update && c.events.on_delete && !c.events.on_create);
    assert_eq!(c.timing, TriggerTiming::AfterCommit);
}

#[test]
fn create_trigger_all_event_kinds() {
    let q = parse_ok(
        "CREATE TRIGGER t3 ON :Post CREATE | UPDATE | DELETE AFTER COMMIT \
             EXECUTE CREATE (a:Log)",
    );
    let c = match &q.clauses[0] {
        Clause::CreateTrigger(c) => c,
        _ => panic!(),
    };
    assert!(c.events.on_create && c.events.on_update && c.events.on_delete);
}

#[test]
fn create_trigger_with_maxdepth_and_on_error_propagate() {
    // `MAXDEPTH n` is the deprecated alias for `CASCADE_LIMIT n` (the trigger architecture).
    let q = parse_ok(
        "CREATE TRIGGER t4 ON :User CREATE BEFORE COMMIT \
             EXECUTE CREATE (a:L) \
             MAXDEPTH 5 \
             ON ERROR PROPAGATE",
    );
    let c = match &q.clauses[0] {
        Clause::CreateTrigger(c) => c,
        _ => panic!(),
    };
    assert_eq!(
        c.cascade_limit,
        Some(5),
        "MAXDEPTH must alias CASCADE_LIMIT"
    );
    assert!(c.cascade_fanout.is_none());
    assert_eq!(c.on_error, Some(OnErrorPolicy::Propagate));
}

#[test]
fn create_trigger_with_cascade_limit_and_fanout() {
    let q = parse_ok(
        "CREATE TRIGGER t8 ON :User CREATE AFTER COMMIT \
             EXECUTE CREATE (a:L) \
             CASCADE_LIMIT 7 \
             CASCADE_FANOUT 250 \
             ON ERROR RETRY 5 WITH BACKOFF 200",
    );
    let c = match &q.clauses[0] {
        Clause::CreateTrigger(c) => c,
        _ => panic!(),
    };
    assert_eq!(c.cascade_limit, Some(7));
    assert_eq!(c.cascade_fanout, Some(250));
    assert_eq!(
        c.on_error,
        Some(OnErrorPolicy::Retry {
            n: 5,
            backoff_ms: 200
        })
    );
}

#[test]
fn create_trigger_options_order_independent() {
    // Order of CASCADE_LIMIT / CASCADE_FANOUT / ON ERROR shouldn't matter.
    let q = parse_ok(
        "CREATE TRIGGER t9 ON :User CREATE AFTER COMMIT \
             EXECUTE CREATE (a:L) \
             ON ERROR DEAD_LETTER \
             CASCADE_FANOUT 42 \
             CASCADE_LIMIT 3",
    );
    let c = match &q.clauses[0] {
        Clause::CreateTrigger(c) => c,
        _ => panic!(),
    };
    assert_eq!(c.cascade_limit, Some(3));
    assert_eq!(c.cascade_fanout, Some(42));
    assert_eq!(c.on_error, Some(OnErrorPolicy::DeadLetter));
}

#[test]
fn create_trigger_on_error_retry_with_backoff() {
    let q = parse_ok(
        "CREATE TRIGGER t5 ON :User CREATE AFTER COMMIT \
             EXECUTE CREATE (a:L) \
             ON ERROR RETRY 7 WITH BACKOFF 500",
    );
    let c = match &q.clauses[0] {
        Clause::CreateTrigger(c) => c,
        _ => panic!(),
    };
    assert_eq!(
        c.on_error,
        Some(OnErrorPolicy::Retry {
            n: 7,
            backoff_ms: 500
        })
    );
}

#[test]
fn create_trigger_on_error_retry_default_backoff() {
    let q = parse_ok(
        "CREATE TRIGGER t6 ON :User CREATE AFTER COMMIT \
             EXECUTE CREATE (a:L) \
             ON ERROR RETRY 3",
    );
    let c = match &q.clauses[0] {
        Clause::CreateTrigger(c) => c,
        _ => panic!(),
    };
    assert_eq!(
        c.on_error,
        Some(OnErrorPolicy::Retry {
            n: 3,
            backoff_ms: 1000
        })
    );
}

#[test]
fn create_trigger_on_error_dead_letter() {
    let q = parse_ok(
        "CREATE TRIGGER t7 ON :User CREATE AFTER COMMIT \
             EXECUTE CREATE (a:L) \
             ON ERROR DEAD_LETTER",
    );
    let c = match &q.clauses[0] {
        Clause::CreateTrigger(c) => c,
        _ => panic!(),
    };
    assert_eq!(c.on_error, Some(OnErrorPolicy::DeadLetter));
}

#[test]
fn create_trigger_default_on_error_per_timing() {
    // the trigger architecture defaults: BEFORE → Propagate, AFTER → Retry 3 / 1000ms
    assert_eq!(
        OnErrorPolicy::default_for(TriggerTiming::BeforeCommit),
        OnErrorPolicy::Propagate
    );
    assert_eq!(
        OnErrorPolicy::default_for(TriggerTiming::AfterCommit),
        OnErrorPolicy::Retry {
            n: 3,
            backoff_ms: 1000
        }
    );
}

#[test]
fn create_trigger_multi_clause_body() {
    let q = parse_ok(
        "CREATE TRIGGER counter ON :Post CREATE AFTER COMMIT \
             EXECUTE MATCH (u:User) \
                     SET u.post_count = u.post_count + 1",
    );
    let c = match &q.clauses[0] {
        Clause::CreateTrigger(c) => c,
        _ => panic!(),
    };
    assert!(c.body_source.contains("MATCH"));
    assert!(c.body_source.contains("SET"));
}

#[test]
fn create_trigger_rejects_empty_event_list() {
    let err = parse("CREATE TRIGGER bad ON :User BEFORE COMMIT EXECUTE CREATE (a:L)");
    assert!(err.is_err(), "missing event list must fail: {err:?}");
}

#[test]
fn create_trigger_rejects_multi_label_target() {
    let err = parse("CREATE TRIGGER bad ON :User:Admin CREATE BEFORE COMMIT EXECUTE CREATE (a:L)");
    assert!(err.is_err(), "multi-label target must fail: {err:?}");
    let msg = format!("{}", err.err().unwrap());
    assert!(
        msg.contains("exactly one label"),
        "error must mention single-label constraint: {msg}"
    );
}

#[test]
fn drop_trigger_basic() {
    let q = parse_ok("DROP TRIGGER my_trigger");
    match &q.clauses[0] {
        Clause::DropTrigger(c) => assert_eq!(c.name, "my_trigger"),
        other => panic!("expected DropTrigger, got {other:?}"),
    }
}

#[test]
fn show_triggers() {
    let q = parse_ok("SHOW TRIGGERS");
    assert!(matches!(&q.clauses[0], Clause::ShowTriggers));
}

#[test]
fn show_sessions() {
    let q = parse_ok("SHOW SESSIONS");
    assert!(matches!(&q.clauses[0], Clause::ShowSessions));
}

#[test]
fn show_transactions() {
    let q = parse_ok("SHOW TRANSACTIONS");
    assert!(matches!(&q.clauses[0], Clause::ShowTransactions));
}

#[test]
fn alter_trigger_disable_enable() {
    let q1 = parse_ok("ALTER TRIGGER t DISABLE");
    let q2 = parse_ok("ALTER TRIGGER t ENABLE");
    match &q1.clauses[0] {
        Clause::AlterTrigger(c) => {
            assert_eq!(c.name, "t");
            assert_eq!(c.action, AlterTriggerAction::Disable);
        }
        _ => panic!(),
    }
    match &q2.clauses[0] {
        Clause::AlterTrigger(c) => {
            assert_eq!(c.action, AlterTriggerAction::Enable);
        }
        _ => panic!(),
    }
}

#[test]
fn alter_trigger_set_body() {
    let q = parse_ok("ALTER TRIGGER t SET EXECUTE CREATE (a:Replacement)");
    match &q.clauses[0] {
        Clause::AlterTrigger(c) => match &c.action {
            AlterTriggerAction::SetBody(s) => assert!(s.contains("Replacement")),
            other => panic!("expected SetBody, got {other:?}"),
        },
        _ => panic!(),
    }
}

#[test]
fn alter_trigger_set_on_error() {
    let q = parse_ok("ALTER TRIGGER t SET ON ERROR DEAD_LETTER");
    match &q.clauses[0] {
        Clause::AlterTrigger(c) => assert_eq!(
            c.action,
            AlterTriggerAction::SetOnError(OnErrorPolicy::DeadLetter)
        ),
        _ => panic!(),
    }
}

// -- SET --

#[test]
fn set_property() {
    let q = parse_ok("MATCH (n:User {id: 42}) SET n.name = 'Bob'");
    if let Clause::Set(ref items, _) = q.clauses[1] {
        assert_eq!(items.len(), 1);
        assert!(matches!(
            items[0],
            SetItem::Property {
                ref variable,
                ref property,
                ..
            } if variable == "n" && property == "name"
        ));
    } else {
        panic!("expected SET clause");
    }
}

#[test]
fn set_multiple_properties() {
    let q = parse_ok("MATCH (n) SET n.name = 'Bob', n.age = 30");
    if let Clause::Set(ref items, _) = q.clauses[1] {
        assert_eq!(items.len(), 2);
    } else {
        panic!("expected SET clause");
    }
}

#[test]
fn set_merge_properties() {
    let q = parse_ok("MATCH (n) SET n += {name: 'Bob', age: 30}");
    if let Clause::Set(ref items, _) = q.clauses[1] {
        assert!(matches!(items[0], SetItem::MergeProperties { .. }));
    } else {
        panic!("expected SET clause");
    }
}

#[test]
fn set_replace_properties() {
    let q = parse_ok("MATCH (n) SET n = {name: 'Bob'}");
    if let Clause::Set(ref items, _) = q.clauses[1] {
        assert!(matches!(items[0], SetItem::ReplaceProperties { .. }));
    } else {
        panic!("expected SET clause");
    }
}

#[test]
fn set_label() {
    let q = parse_ok("MATCH (n) SET n:Admin");
    if let Clause::Set(ref items, _) = q.clauses[1] {
        assert!(matches!(
            items[0],
            SetItem::AddLabel {
                ref variable,
                ref label,
                ..
            } if variable == "n" && label == "Admin"
        ));
    } else {
        panic!("expected SET clause");
    }
}

#[test]
fn set_property_path_deep() {
    let q = parse_ok("MATCH (n) SET n.config.network.ssid = 'home'");
    if let Clause::Set(ref items, _) = q.clauses[1] {
        assert_eq!(items.len(), 1);
        assert!(matches!(
            items[0],
            SetItem::PropertyPath {
                ref variable,
                ref path,
                ..
            } if variable == "n" && path == &["config", "network", "ssid"]
        ));
    } else {
        panic!("expected SET clause");
    }
}

#[test]
fn set_property_path_two_levels_is_property() {
    // Two-level path (n.name) should still be SetItem::Property, not PropertyPath.
    let q = parse_ok("MATCH (n) SET n.name = 'Alice'");
    if let Clause::Set(ref items, _) = q.clauses[1] {
        assert!(matches!(items[0], SetItem::Property { .. }));
    } else {
        panic!("expected SET clause");
    }
}

#[test]
fn set_multiple_with_path() {
    let q = parse_ok("MATCH (n) SET n.config.network.ssid = 'home', n.name = 'Alice'");
    if let Clause::Set(ref items, _) = q.clauses[1] {
        assert_eq!(items.len(), 2);
        assert!(matches!(items[0], SetItem::PropertyPath { .. }));
        assert!(matches!(items[1], SetItem::Property { .. }));
    } else {
        panic!("expected SET clause");
    }
}

// -- doc_* functions (R165) --

#[test]
fn set_doc_push() {
    let q = parse_ok("MATCH (n) SET doc_push(n.tags, 'new')");
    if let Clause::Set(ref items, _) = q.clauses[1] {
        assert!(matches!(
            items[0],
            SetItem::DocFunction {
                ref function,
                ref variable,
                ref path,
                ..
            } if function == "doc_push" && variable == "n" && path == &["tags"]
        ));
    } else {
        panic!("expected SET clause");
    }
}

#[test]
fn set_doc_inc_nested_path() {
    let q = parse_ok("MATCH (n) SET doc_inc(n.stats.views, 1)");
    if let Clause::Set(ref items, _) = q.clauses[1] {
        assert!(matches!(
            items[0],
            SetItem::DocFunction {
                ref function,
                ref variable,
                ref path,
                ..
            } if function == "doc_inc" && variable == "n" && path == &["stats", "views"]
        ));
    } else {
        panic!("expected SET clause");
    }
}

#[test]
fn set_multiple_doc_functions() {
    let q = parse_ok("MATCH (n) SET doc_push(n.tags, 'a'), doc_add_to_set(n.labels, 'b')");
    if let Clause::Set(ref items, _) = q.clauses[1] {
        assert_eq!(items.len(), 2);
        assert!(
            matches!(items[0], SetItem::DocFunction { ref function, .. } if function == "doc_push")
        );
        assert!(
            matches!(items[1], SetItem::DocFunction { ref function, .. } if function == "doc_add_to_set")
        );
    } else {
        panic!("expected SET clause");
    }
}

// -- REMOVE --

#[test]
fn remove_property() {
    let q = parse_ok("MATCH (n) REMOVE n.age");
    if let Clause::Remove(ref items) = q.clauses[1] {
        assert_eq!(items.len(), 1);
        assert!(matches!(
            items[0],
            RemoveItem::Property {
                ref variable,
                ref property,
            } if variable == "n" && property == "age"
        ));
    } else {
        panic!("expected REMOVE clause");
    }
}

#[test]
fn remove_label() {
    let q = parse_ok("MATCH (n) REMOVE n:Admin");
    if let Clause::Remove(ref items) = q.clauses[1] {
        assert!(matches!(
            items[0],
            RemoveItem::Label {
                ref variable,
                ref label,
            } if variable == "n" && label == "Admin"
        ));
    } else {
        panic!("expected REMOVE clause");
    }
}

#[test]
fn remove_multiple() {
    let q = parse_ok("MATCH (n) REMOVE n.age, n:Admin");
    if let Clause::Remove(ref items) = q.clauses[1] {
        assert_eq!(items.len(), 2);
        assert!(matches!(items[0], RemoveItem::Property { .. }));
        assert!(matches!(items[1], RemoveItem::Label { .. }));
    } else {
        panic!("expected REMOVE clause");
    }
}

#[test]
fn remove_property_path_deep() {
    let q = parse_ok("MATCH (n) REMOVE n.config.network.ssid");
    if let Clause::Remove(ref items) = q.clauses[1] {
        assert_eq!(items.len(), 1);
        assert!(matches!(
            items[0],
            RemoveItem::PropertyPath {
                ref variable,
                ref path,
            } if variable == "n" && path == &["config", "network", "ssid"]
        ));
    } else {
        panic!("expected REMOVE clause");
    }
}

#[test]
fn remove_property_two_levels_is_property() {
    let q = parse_ok("MATCH (n) REMOVE n.age");
    if let Clause::Remove(ref items) = q.clauses[1] {
        assert!(matches!(items[0], RemoveItem::Property { .. }));
    } else {
        panic!("expected REMOVE clause");
    }
}

// -- UPSERT MATCH --

#[test]
fn upsert_match() {
    let q = parse_ok(
        "UPSERT MATCH (u:User {email: 'alice@example.com'}) \
             ON MATCH SET u.login_count = u.login_count + 1 \
             ON CREATE CREATE (u:User {email: 'alice@example.com', login_count: 1}) \
             RETURN u",
    );
    if let Clause::Upsert(ref uc) = q.clauses[0] {
        assert_eq!(uc.on_match.len(), 1);
        assert_eq!(uc.on_create.len(), 1);
    } else {
        panic!("expected UPSERT clause");
    }
    assert!(matches!(q.clauses[1], Clause::Return(_)));
}

// -- Complex write queries --

#[test]
fn match_set_return() {
    let q = parse_ok("MATCH (n:User {id: $id}) SET n.name = $name, n.updated = $now RETURN n");
    assert!(matches!(q.clauses[0], Clause::Match(_)));
    assert!(matches!(q.clauses[1], Clause::Set(_, _)));
    assert!(matches!(q.clauses[2], Clause::Return(_)));
}

#[test]
fn create_return() {
    let q = parse_ok("CREATE (n:User {name: 'Alice'}) RETURN n");
    assert!(matches!(q.clauses[0], Clause::Create(_)));
    assert!(matches!(q.clauses[1], Clause::Return(_)));
}

#[test]
fn match_delete_return() {
    let q = parse_ok("MATCH (n:User {id: 42})-[r:KNOWS]->(m) DELETE r RETURN n, m");
    assert!(matches!(q.clauses[0], Clause::Match(_)));
    assert!(matches!(q.clauses[1], Clause::Delete(_)));
    assert!(matches!(q.clauses[2], Clause::Return(_)));
}

#[test]
fn merge_with_relationship() {
    let q = parse_ok(
        "MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'}) \
             MERGE (a)-[:KNOWS]->(b)",
    );
    assert!(matches!(q.clauses[0], Clause::Match(_)));
    assert!(matches!(q.clauses[1], Clause::Merge(_)));
}

// -- Dot-notation (multi-level property access) --

#[test]
fn dot_notation_two_levels() {
    // n.config.version → PropertyAccess(PropertyAccess(Variable("n"), "config"), "version")
    let q = parse_ok("MATCH (n) RETURN n.config.version");
    if let Clause::Return(ref ret) = q.clauses[1] {
        let item = &ret.items[0];
        if let Expr::PropertyAccess {
            expr: outer_expr,
            property: outer_prop,
        } = &item.expr
        {
            assert_eq!(outer_prop, "version");
            if let Expr::PropertyAccess {
                expr: inner_expr,
                property: inner_prop,
            } = outer_expr.as_ref()
            {
                assert_eq!(inner_prop, "config");
                assert!(matches!(inner_expr.as_ref(), Expr::Variable(v) if v == "n"));
            } else {
                panic!("expected inner PropertyAccess, got: {outer_expr:?}");
            }
        } else {
            panic!("expected PropertyAccess, got: {:?}", item.expr);
        }
    }
}

#[test]
fn dot_notation_three_levels() {
    // n.a.b.c → nested PropertyAccess chain
    let q = parse_ok("MATCH (n) RETURN n.a.b.c");
    if let Clause::Return(ref ret) = q.clauses[1] {
        let item = &ret.items[0];
        // Outermost: PropertyAccess { .., property: "c" }
        if let Expr::PropertyAccess { property, expr } = &item.expr {
            assert_eq!(property, "c");
            // Middle: PropertyAccess { .., property: "b" }
            if let Expr::PropertyAccess { property, expr } = expr.as_ref() {
                assert_eq!(property, "b");
                // Inner: PropertyAccess { Variable("n"), property: "a" }
                if let Expr::PropertyAccess { property, expr } = expr.as_ref() {
                    assert_eq!(property, "a");
                    assert!(matches!(expr.as_ref(), Expr::Variable(v) if v == "n"));
                } else {
                    panic!("expected PropertyAccess for 'a'");
                }
            } else {
                panic!("expected PropertyAccess for 'b'");
            }
        } else {
            panic!("expected PropertyAccess for 'c'");
        }
    }
}

#[test]
fn dot_notation_in_where() {
    // WHERE n.config.enabled = true
    let q = parse_ok("MATCH (n) WHERE n.config.enabled = true RETURN n");
    if let Clause::Match(ref m) = q.clauses[0] {
        let w = m.where_clause.as_ref().expect("WHERE clause");
        // Should be BinaryOp { left: PropertyAccess chain, op: Eq, right: Literal(true) }
        if let Expr::BinaryOp { left, op, .. } = w {
            assert_eq!(*op, BinaryOperator::Eq);
            if let Expr::PropertyAccess { property, .. } = left.as_ref() {
                assert_eq!(property, "enabled");
            } else {
                panic!("expected PropertyAccess, got: {left:?}");
            }
        } else {
            panic!("expected BinaryOp");
        }
    }
}

// -- Map projection --

#[test]
fn map_projection_shorthand() {
    let q = parse_ok("MATCH (n:User) RETURN n { .name, .age }");
    if let Clause::Return(ref rc) = q.clauses[1] {
        if let Expr::MapProjection { expr, items } = &rc.items[0].expr {
            assert!(matches!(expr.as_ref(), Expr::Variable(v) if v == "n"));
            assert_eq!(items.len(), 2);
            assert!(matches!(&items[0], MapProjectionItem::Property(p) if p == "name"));
            assert!(matches!(&items[1], MapProjectionItem::Property(p) if p == "age"));
        } else {
            panic!("expected MapProjection, got: {:?}", rc.items[0].expr);
        }
    }
}

#[test]
fn map_projection_with_computed() {
    let q =
        parse_ok("MATCH (n:User)-[:WROTE]->(p:Post) RETURN n { .name, posts: collect(p.title) }");
    if let Clause::Return(ref rc) = q.clauses[1] {
        if let Expr::MapProjection { items, .. } = &rc.items[0].expr {
            assert_eq!(items.len(), 2);
            assert!(matches!(&items[0], MapProjectionItem::Property(p) if p == "name"));
            assert!(matches!(&items[1], MapProjectionItem::Computed(alias, _) if alias == "posts"));
        } else {
            panic!("expected MapProjection");
        }
    }
}

#[test]
fn map_projection_nested() {
    // Nested map projection: collect(p { .title, .body })
    let q = parse_ok(
        "MATCH (u:User)-[:WROTE]->(p:Post) \
             RETURN u { .name, posts: collect(p { .title, .body }) }",
    );
    if let Clause::Return(ref rc) = q.clauses[1] {
        if let Expr::MapProjection { items, .. } = &rc.items[0].expr {
            assert_eq!(items.len(), 2);
            // Second item: posts: collect(p { .title, .body })
            if let MapProjectionItem::Computed(ref alias, ref cexpr) = items[1] {
                assert_eq!(alias, "posts");
                // collect(p { .title, .body })
                assert!(matches!(cexpr, Expr::FunctionCall { name, .. } if name == "collect"));
            } else {
                panic!("expected Computed item");
            }
        } else {
            panic!("expected MapProjection");
        }
    }
}

// ── Per-query hint extraction (G026) ───────────────────────────────

#[test]
fn hint_vector_consistency_snapshot() {
    let q = parse_ok("MATCH (m:Movie) RETURN m /*+ vector_consistency('snapshot') */");
    assert_eq!(q.hints.len(), 1);
    assert_eq!(
        q.hints[0],
        QueryHint::VectorConsistency(
            coordinode_core::graph::types::VectorConsistencyMode::Snapshot
        )
    );
}

#[test]
fn hint_vector_consistency_exact() {
    let q = parse_ok("/*+ vector_consistency('exact') */ MATCH (n:Node) RETURN n");
    assert_eq!(q.hints.len(), 1);
    assert_eq!(
        q.hints[0],
        QueryHint::VectorConsistency(coordinode_core::graph::types::VectorConsistencyMode::Exact)
    );
}

#[test]
fn hint_unknown_key_ignored() {
    let q = parse_ok("MATCH (n) RETURN n /*+ unknown_hint('value') */");
    assert!(
        q.hints.is_empty(),
        "unknown hints should be silently ignored"
    );
}

#[test]
fn hint_no_hints() {
    let q = parse_ok("MATCH (n) RETURN n");
    assert!(q.hints.is_empty());
}

#[test]
fn hint_with_regular_comment() {
    // Regular comments should still be ignored, only /*+ ... */ are hints
    let q =
        parse_ok("/* regular comment */ MATCH (n) RETURN n /*+ vector_consistency('snapshot') */");
    assert_eq!(q.hints.len(), 1);
}

#[test]
fn hint_double_quoted_value() {
    let q = parse_ok("MATCH (n) RETURN n /*+ vector_consistency(\"current\") */");
    assert_eq!(q.hints.len(), 1);
    assert_eq!(
        q.hints[0],
        QueryHint::VectorConsistency(coordinode_core::graph::types::VectorConsistencyMode::Current)
    );
}

// --- CREATE TEXT INDEX DDL (G016) ---

#[test]
fn create_text_index_simple_syntax() {
    let q = parse_ok("CREATE TEXT INDEX article_body ON :Article(body)");
    assert_eq!(q.clauses.len(), 1);
    match &q.clauses[0] {
        Clause::CreateTextIndex(c) => {
            assert_eq!(c.name, "article_body");
            assert_eq!(c.label, "Article");
            assert_eq!(c.fields.len(), 1);
            assert_eq!(c.fields[0].property, "body");
            assert!(c.fields[0].analyzer.is_none());
            assert!(c.default_language.is_none());
            assert!(c.language_override.is_none());
        }
        other => panic!("expected CreateTextIndex, got {other:?}"),
    }
}

#[test]
fn create_text_index_simple_with_language() {
    let q = parse_ok("CREATE TEXT INDEX idx ON :Article(body) LANGUAGE 'russian'");
    match &q.clauses[0] {
        Clause::CreateTextIndex(c) => {
            assert_eq!(c.fields[0].property, "body");
            assert_eq!(c.default_language.as_deref(), Some("russian"));
        }
        other => panic!("expected CreateTextIndex, got {other:?}"),
    }
}

#[test]
fn create_text_index_multi_field() {
    let q = parse_ok(
        r#"CREATE TEXT INDEX article_text ON :Article {
                title: { analyzer: "english" },
                body:  { analyzer: "auto_detect" }
            } DEFAULT LANGUAGE "english""#,
    );
    match &q.clauses[0] {
        Clause::CreateTextIndex(c) => {
            assert_eq!(c.name, "article_text");
            assert_eq!(c.label, "Article");
            assert_eq!(c.fields.len(), 2);
            assert_eq!(c.fields[0].property, "title");
            assert_eq!(c.fields[0].analyzer.as_deref(), Some("english"));
            assert_eq!(c.fields[1].property, "body");
            assert_eq!(c.fields[1].analyzer.as_deref(), Some("auto_detect"));
            assert_eq!(c.default_language.as_deref(), Some("english"));
        }
        other => panic!("expected CreateTextIndex, got {other:?}"),
    }
}

#[test]
fn create_text_index_multi_field_with_override() {
    let q = parse_ok(
        r#"CREATE TEXT INDEX idx ON :Article {
                title: { analyzer: "russian" }
            } DEFAULT LANGUAGE "english" LANGUAGE OVERRIDE "lang""#,
    );
    match &q.clauses[0] {
        Clause::CreateTextIndex(c) => {
            assert_eq!(c.default_language.as_deref(), Some("english"));
            assert_eq!(c.language_override.as_deref(), Some("lang"));
        }
        other => panic!("expected CreateTextIndex, got {other:?}"),
    }
}

#[test]
fn create_text_index_multi_field_no_modifiers() {
    let q = parse_ok(r#"CREATE TEXT INDEX idx ON :Post { content: { analyzer: "german" } }"#);
    match &q.clauses[0] {
        Clause::CreateTextIndex(c) => {
            assert_eq!(c.fields.len(), 1);
            assert_eq!(c.fields[0].property, "content");
            assert_eq!(c.fields[0].analyzer.as_deref(), Some("german"));
            assert!(c.default_language.is_none());
            assert!(c.language_override.is_none());
        }
        other => panic!("expected CreateTextIndex, got {other:?}"),
    }
}

// ── Pattern predicate parsing ────────────────────────────────────────

/// Helper: extract inline WHERE expression from first MATCH clause.
fn extract_match_where(q: &Query) -> Option<&Expr> {
    q.clauses.iter().find_map(|c| match c {
        Clause::Match(m) => m.where_clause.as_ref(),
        _ => None,
    })
}

#[test]
fn parse_where_pattern_predicate() {
    let q = parse("MATCH (a), (b) WHERE (a)-[:KNOWS]->(b) RETURN a").unwrap();
    let where_expr = extract_match_where(&q).expect("WHERE clause missing");
    if let Expr::PatternPredicate(p) = where_expr {
        assert_eq!(p.elements.len(), 3); // node, rel, node
    } else {
        panic!("expected PatternPredicate, got {where_expr:?}");
    }
}

#[test]
fn parse_where_not_pattern_predicate() {
    let q = parse("MATCH (a), (b) WHERE NOT (a)-[:KNOWS]->(b) RETURN a").unwrap();
    let where_expr = extract_match_where(&q).expect("WHERE clause missing");
    if let Expr::UnaryOp {
        op: UnaryOperator::Not,
        expr,
    } = where_expr
    {
        assert!(
            matches!(expr.as_ref(), Expr::PatternPredicate(_)),
            "expected PatternPredicate inside NOT, got {expr:?}"
        );
    } else {
        panic!("expected NOT expression, got {where_expr:?}");
    }
}

#[test]
fn parse_pattern_predicate_with_labels() {
    let q = parse("MATCH (a:Person), (b:Person) WHERE (a)-[:KNOWS]->(b) RETURN a").unwrap();
    let where_expr = extract_match_where(&q).expect("WHERE clause missing");
    assert!(
        matches!(where_expr, Expr::PatternPredicate(_)),
        "expected PatternPredicate"
    );
}

#[test]
fn parse_pattern_predicate_undirected() {
    let q = parse("MATCH (a), (b) WHERE (a)-[:KNOWS]-(b) RETURN a").unwrap();
    let where_expr = extract_match_where(&q).expect("WHERE clause missing");
    if let Expr::PatternPredicate(p) = where_expr {
        if let PatternElement::Relationship(rel) = &p.elements[1] {
            assert_eq!(rel.direction, Direction::Both);
        } else {
            panic!("expected relationship element");
        }
    } else {
        panic!("expected PatternPredicate");
    }
}

#[test]
fn parse_pattern_predicate_and_scalar() {
    let q = parse("MATCH (a), (b) WHERE (a)-[:KNOWS]->(b) AND a.age > 30 RETURN a").unwrap();
    let where_expr = extract_match_where(&q).expect("WHERE clause missing");
    if let Expr::BinaryOp { left, op, right } = where_expr {
        assert_eq!(*op, BinaryOperator::And);
        assert!(
            matches!(left.as_ref(), Expr::PatternPredicate(_)),
            "left should be PatternPredicate"
        );
        assert!(matches!(right.as_ref(), Expr::BinaryOp { .. }));
    } else {
        panic!("expected AND expression, got {where_expr:?}");
    }
}

#[test]
fn parse_parenthesized_expr_not_pattern_predicate() {
    let q = parse("MATCH (n) WHERE (n.age) > 5 RETURN n").unwrap();
    let where_expr = extract_match_where(&q).expect("WHERE clause missing");
    assert!(
        !matches!(where_expr, Expr::PatternPredicate(_)),
        "parenthesized expression should not be PatternPredicate"
    );
}

// --- CREATE INDEX / DROP INDEX DDL (R-API2) ---

#[test]
fn create_index_simple() {
    let q = parse_ok("CREATE INDEX email_idx ON :User(email)");
    assert_eq!(q.clauses.len(), 1);
    match &q.clauses[0] {
        Clause::CreateIndex(c) => {
            assert_eq!(c.name, "email_idx");
            assert_eq!(c.label, "User");
            assert_eq!(c.property, "email");
            assert!(!c.unique);
            assert!(!c.sparse);
            assert!(c.filter_expr.is_none());
        }
        other => panic!("expected CreateIndex, got {other:?}"),
    }
}

#[test]
fn create_unique_index() {
    let q = parse_ok("CREATE UNIQUE INDEX email_idx ON :User(email)");
    match &q.clauses[0] {
        Clause::CreateIndex(c) => {
            assert_eq!(c.name, "email_idx");
            assert!(c.unique, "expected unique=true");
            assert!(!c.sparse);
        }
        other => panic!("expected CreateIndex, got {other:?}"),
    }
}

#[test]
fn create_sparse_index() {
    let q = parse_ok("CREATE SPARSE INDEX opt_idx ON :User(optional_prop)");
    match &q.clauses[0] {
        Clause::CreateIndex(c) => {
            assert_eq!(c.name, "opt_idx");
            assert!(c.sparse, "expected sparse=true");
            assert!(!c.unique);
        }
        other => panic!("expected CreateIndex, got {other:?}"),
    }
}

#[test]
fn create_unique_sparse_index() {
    let q = parse_ok("CREATE UNIQUE SPARSE INDEX us_idx ON :Item(code)");
    match &q.clauses[0] {
        Clause::CreateIndex(c) => {
            assert!(c.unique);
            assert!(c.sparse);
            assert_eq!(c.label, "Item");
            assert_eq!(c.property, "code");
        }
        other => panic!("expected CreateIndex, got {other:?}"),
    }
}

#[test]
fn create_index_with_where_clause() {
    let q = parse_ok("CREATE INDEX active_users ON :User(email) WHERE n.active = true");
    match &q.clauses[0] {
        Clause::CreateIndex(c) => {
            assert_eq!(c.name, "active_users");
            // The filter_expr should be present (partial index).
            assert!(
                c.filter_expr.is_some(),
                "expected filter_expr from WHERE clause"
            );
        }
        other => panic!("expected CreateIndex, got {other:?}"),
    }
}

#[test]
fn drop_index_simple() {
    let q = parse_ok("DROP INDEX email_idx");
    assert_eq!(q.clauses.len(), 1);
    match &q.clauses[0] {
        Clause::DropIndex(c) => {
            assert_eq!(c.name, "email_idx");
        }
        other => panic!("expected DropIndex, got {other:?}"),
    }
}

#[test]
fn create_index_does_not_shadow_create_node() {
    // Verify that CREATE INDEX DDL does not interfere with regular CREATE (node) parsing.
    let q = parse_ok("CREATE (n:User {email: 'alice@example.com'})");
    assert_eq!(q.clauses.len(), 1);
    assert!(
        matches!(q.clauses[0], Clause::Create(_)),
        "regular CREATE should still parse as Clause::Create"
    );
}

#[test]
fn subscript_access_on_function_call() {
    // labels(n)[0] → Subscript { expr: FunctionCall("labels", [Variable("n")]), index: Literal(0) }
    let q = parse_ok("MATCH (n) RETURN labels(n)[0] AS lbl");
    if let Clause::Return(ref rc) = q.clauses[1] {
        if let Expr::Subscript {
            ref expr,
            ref index,
        } = rc.items[0].expr
        {
            assert!(
                matches!(**expr, Expr::FunctionCall { ref name, .. } if name == "labels"),
                "base must be FunctionCall(labels), got {expr:?}"
            );
            assert_eq!(**index, Expr::Literal(Value::Int(0)), "index must be 0");
        } else {
            panic!("expected Subscript expr, got {:?}", rc.items[0].expr);
        }
    } else {
        panic!("expected RETURN clause at index 1");
    }
}

#[test]
fn subscript_access_on_list_literal() {
    // [1, 2, 3][1] → Subscript { expr: List([1,2,3]), index: Literal(1) }
    let q = parse_ok("RETURN [1, 2, 3][1] AS x");
    if let Clause::Return(ref rc) = q.clauses[0] {
        assert!(
            matches!(rc.items[0].expr, Expr::Subscript { .. }),
            "expected Subscript, got {:?}",
            rc.items[0].expr
        );
    } else {
        panic!("expected RETURN clause");
    }
}

#[test]
fn chained_subscript_access() {
    // matrix[0][1] → Subscript { Subscript { Variable("matrix"), 0 }, 1 }
    let q = parse_ok("RETURN matrix[0][1] AS v");
    if let Clause::Return(ref rc) = q.clauses[0] {
        if let Expr::Subscript {
            ref expr,
            ref index,
        } = rc.items[0].expr
        {
            assert_eq!(**index, Expr::Literal(Value::Int(1)));
            assert!(
                matches!(**expr, Expr::Subscript { .. }),
                "inner must also be Subscript, got {expr:?}"
            );
        } else {
            panic!("expected outer Subscript, got {:?}", rc.items[0].expr);
        }
    }
}

// ====== DETACH DOCUMENT (R167) ======

#[test]
fn detach_document_basic() {
    let q = parse_ok(
        "MATCH (n:User) \
             DETACH DOCUMENT n.address AS (a:Address)-[:HAS_ADDRESS]->(n)",
    );
    let Clause::DetachDocument(ref dd) = q.clauses[1] else {
        panic!("expected DetachDocument, got {:?}", q.clauses[1]);
    };
    assert_eq!(dd.source_variable, "n");
    assert_eq!(dd.property_path, vec!["address"]);
    assert_eq!(dd.target_variable, "a");
    assert_eq!(dd.target_labels, vec!["Address"]);
    assert_eq!(dd.edge_type.as_deref(), Some("HAS_ADDRESS"));
    // (a:Address)-[:HAS_ADDRESS]->(n): edge goes a → n, so from `n`'s
    // perspective it is incoming.
    assert_eq!(dd.edge_direction, EdgeFromSource::Incoming);
    assert!(dd.transfer.is_none());
}

#[test]
fn detach_document_nested_path() {
    let q = parse_ok(
        "MATCH (n:User) \
             DETACH DOCUMENT n.meta.shipping AS (s:ShippingAddress)-[:HAS_SHIPPING]->(n)",
    );
    let Clause::DetachDocument(ref dd) = q.clauses[1] else {
        panic!("expected DetachDocument");
    };
    assert_eq!(dd.property_path, vec!["meta", "shipping"]);
    assert_eq!(dd.target_variable, "s");
}

#[test]
fn detach_document_with_transfer_edges_in_list() {
    let q = parse_ok(
        "MATCH (n:User) \
             DETACH DOCUMENT n.address AS (a:Address)-[:HAS_ADDRESS]->(n) \
             TRANSFER EDGES ON n TO a WHERE type(r) IN ['SHIPS_TO', 'LIVES_AT']",
    );
    let Clause::DetachDocument(ref dd) = q.clauses[1] else {
        panic!("expected DetachDocument");
    };
    let t = dd.transfer.as_ref().expect("transfer spec");
    assert_eq!(t.node_variable, "n");
    assert_eq!(t.target_variable, "a");
    // predicate must be `type(r) IN [...]`.
    assert!(matches!(t.predicate, Expr::In { .. }));
}

#[test]
fn detach_document_requires_property_path() {
    // Just `n` alone — grammar requires at least one `.segment`.
    let err = parse_err(
        "MATCH (n:User) \
             DETACH DOCUMENT n AS (a:Address)-[:HAS_ADDRESS]->(n)",
    );
    // Any parse error is acceptable — the important part is that the query
    // does not parse successfully.
    let _ = err;
}

#[test]
fn detach_document_reverse_relationship() {
    // Mirror form: (n)<-[:HAS_ADDRESS]-(a:Address) — same semantics.
    let q = parse_ok(
        "MATCH (n:User) \
             DETACH DOCUMENT n.address AS (n)<-[:HAS_ADDRESS]-(a:Address)",
    );
    let Clause::DetachDocument(ref dd) = q.clauses[1] else {
        panic!("expected DetachDocument");
    };
    assert_eq!(dd.target_variable, "a");
    assert_eq!(dd.target_labels, vec!["Address"]);
    // (n)<-[:HAS_ADDRESS]-(a): edge goes a → n, still Incoming for n.
    assert_eq!(dd.edge_direction, EdgeFromSource::Incoming);
}

// ====== ATTACH DOCUMENT (R168) ======

#[test]
fn attach_document_basic() {
    let q = parse_ok("ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address");
    let Clause::AttachDocument(ref ad) = q.clauses[0] else {
        panic!("expected AttachDocument, got {:?}", q.clauses[0]);
    };
    assert_eq!(ad.source_variable, "a");
    assert_eq!(ad.source_labels, vec!["Address"]);
    assert_eq!(ad.target_variable, "u");
    assert_eq!(ad.target_labels, vec!["User"]);
    assert_eq!(ad.edge_type, "HAS_ADDRESS");
    assert_eq!(ad.edge_direction, EdgeFromSource::Outgoing);
    assert_eq!(ad.target_property_variable, "u");
    assert_eq!(ad.target_property_path, vec!["address"]);
    assert!(ad.transfer.is_none());
    assert!(!ad.on_conflict_replace);
    assert!(!ad.on_remaining_fail);
}

#[test]
fn attach_document_with_transfer_and_options() {
    let q = parse_ok(
        "ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address \
             TRANSFER EDGES ON a TO u WHERE type(r) = 'SHIPS_TO' \
             ON CONFLICT REPLACE \
             ON REMAINING FAIL",
    );
    let Clause::AttachDocument(ref ad) = q.clauses[0] else {
        panic!("expected AttachDocument");
    };
    assert!(ad.transfer.is_some());
    assert!(ad.on_conflict_replace);
    assert!(ad.on_remaining_fail);
}

#[test]
fn attach_document_nested_target_path() {
    let q = parse_ok("ATTACH (a:Shipping)-[:HAS_SHIPPING]->(u:User) INTO u.meta.shipping");
    let Clause::AttachDocument(ref ad) = q.clauses[0] else {
        panic!("expected AttachDocument");
    };
    assert_eq!(ad.target_property_path, vec!["meta", "shipping"]);
}

#[test]
fn attach_document_target_var_must_match_pattern() {
    // INTO target variable `x` doesn't match pattern target `u` → parse error.
    let err = parse_err("ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO x.address");
    let msg = format!("{err}");
    assert!(
        msg.contains("match pattern target") || msg.contains("target node variable"),
        "error should mention mismatch: {msg}"
    );
}

#[test]
fn attach_document_requires_edge_type() {
    // Anonymous relationship `-[]->` has no type → build error (ATTACH
    // requires an explicit type for targeted adjacency delete).
    let err = parse_err("ATTACH (a:Address)-[]->(u:User) INTO u.address");
    let _ = err;
}

#[test]
fn create_edge_type_minimal() {
    let q = parse_ok("CREATE EDGE TYPE WORKS_AT");
    match &q.clauses[0] {
        Clause::CreateEdgeType(c) => {
            assert_eq!(c.name, "WORKS_AT");
            assert!(!c.temporal);
            assert!(c.properties.is_empty());
        }
        other => panic!("expected CreateEdgeType, got {other:?}"),
    }
}

#[test]
fn create_edge_type_temporal_no_properties() {
    let q = parse_ok("CREATE EDGE TYPE WORKS_AT TEMPORAL");
    match &q.clauses[0] {
        Clause::CreateEdgeType(c) => {
            assert_eq!(c.name, "WORKS_AT");
            assert!(c.temporal);
            assert!(c.properties.is_empty());
        }
        other => panic!("expected CreateEdgeType, got {other:?}"),
    }
}

#[test]
fn create_edge_type_temporal_with_properties() {
    let q = parse_ok(
            "CREATE EDGE TYPE WORKS_AT TEMPORAL WITH (role: STRING, valid_from: TIMESTAMP NOT NULL, valid_to: TIMESTAMP)",
        );
    match &q.clauses[0] {
        Clause::CreateEdgeType(c) => {
            assert_eq!(c.name, "WORKS_AT");
            assert!(c.temporal);
            assert_eq!(c.properties.len(), 3);
            assert_eq!(c.properties[0].name, "role");
            assert_eq!(c.properties[0].type_name, "STRING");
            assert!(!c.properties[0].not_null);
            assert_eq!(c.properties[1].name, "valid_from");
            assert_eq!(c.properties[1].type_name, "TIMESTAMP");
            assert!(c.properties[1].not_null);
            assert_eq!(c.properties[2].name, "valid_to");
            assert_eq!(c.properties[2].type_name, "TIMESTAMP");
            assert!(!c.properties[2].not_null);
        }
        other => panic!("expected CreateEdgeType, got {other:?}"),
    }
}

#[test]
fn create_edge_type_non_temporal_with_properties() {
    // TEMPORAL is optional — non-temporal edge types may still declare props.
    let q = parse_ok("CREATE EDGE TYPE LIKES WITH (weight: FLOAT)");
    match &q.clauses[0] {
        Clause::CreateEdgeType(c) => {
            assert!(!c.temporal);
            assert_eq!(c.properties.len(), 1);
            assert_eq!(c.properties[0].name, "weight");
            assert_eq!(c.properties[0].type_name, "FLOAT");
        }
        other => panic!("expected CreateEdgeType, got {other:?}"),
    }
}

#[test]
fn create_edge_type_rejects_unknown_type() {
    // QUATERNION isn't in the property_type_name keyword set → parse error.
    let _ = parse_err("CREATE EDGE TYPE WORKS_AT WITH (foo: QUATERNION)");
}

// ===== CREATE NODE TYPE (R172a per ADR-027) =====

#[test]
fn create_node_type_minimal() {
    let q = parse_ok("CREATE NODE TYPE Person");
    match &q.clauses[0] {
        Clause::CreateNodeType(c) => {
            assert_eq!(c.name, "Person");
            assert!(!c.temporal);
            assert!(c.properties.is_empty());
        }
        other => panic!("expected CreateNodeType, got {other:?}"),
    }
}

#[test]
fn create_node_type_temporal_no_properties() {
    let q = parse_ok("CREATE NODE TYPE Person TEMPORAL");
    match &q.clauses[0] {
        Clause::CreateNodeType(c) => {
            assert_eq!(c.name, "Person");
            assert!(c.temporal);
            assert!(c.properties.is_empty());
        }
        other => panic!("expected CreateNodeType, got {other:?}"),
    }
}

#[test]
fn create_node_type_temporal_with_properties() {
    let q = parse_ok(
            "CREATE NODE TYPE Person TEMPORAL WITH (name: STRING NOT NULL, valid_from: TIMESTAMP NOT NULL, valid_to: TIMESTAMP)",
        );
    match &q.clauses[0] {
        Clause::CreateNodeType(c) => {
            assert_eq!(c.name, "Person");
            assert!(c.temporal);
            assert_eq!(c.properties.len(), 3);
            assert_eq!(c.properties[0].name, "name");
            assert_eq!(c.properties[0].type_name, "STRING");
            assert!(c.properties[0].not_null);
            assert_eq!(c.properties[1].name, "valid_from");
            assert_eq!(c.properties[1].type_name, "TIMESTAMP");
            assert!(c.properties[1].not_null);
            assert_eq!(c.properties[2].name, "valid_to");
            assert!(!c.properties[2].not_null);
        }
        other => panic!("expected CreateNodeType, got {other:?}"),
    }
}

#[test]
fn create_node_type_non_temporal_with_properties() {
    // TEMPORAL is optional — point-in-time labels may still declare props.
    let q = parse_ok("CREATE NODE TYPE User WITH (email: STRING NOT NULL)");
    match &q.clauses[0] {
        Clause::CreateNodeType(c) => {
            assert_eq!(c.name, "User");
            assert!(!c.temporal);
            assert_eq!(c.properties.len(), 1);
            assert_eq!(c.properties[0].name, "email");
            assert!(c.properties[0].not_null);
        }
        other => panic!("expected CreateNodeType, got {other:?}"),
    }
}

#[test]
fn create_node_type_rejects_unknown_property_type() {
    let _ = parse_err("CREATE NODE TYPE Person WITH (foo: QUATERNION)");
}

/// `node` is a context keyword (only meaningful in `CREATE NODE TYPE`),
/// NOT in reserved_word. It must remain a valid identifier elsewhere —
/// e.g., as a variable name in MATCH / RETURN.
#[test]
fn create_node_type_does_not_break_node_as_identifier() {
    // `node` as variable name in MATCH / RETURN must still parse.
    let _ = parse_ok("MATCH (node:Person) RETURN node");
    let _ = parse_ok("MATCH (n:Foo) WHERE n.node = 1 RETURN n");
    // `node` as property accessor inside an expression.
    let _ = parse_ok("MATCH (n) RETURN n.node");
}

/// Empty `WITH ()` block isn't valid grammar — `property_decl_list`
/// requires at least one declaration. This mirrors the edge-type
/// behaviour and prevents accidentally-empty WITH clauses.
#[test]
fn create_node_type_empty_with_block_rejected() {
    // Empty WITH () → grammar rejects (property_decl_list = decl+).
    let _ = parse_err("CREATE NODE TYPE Person TEMPORAL WITH ()");
}

/// Keyword `TEMPORAL` is case-insensitive (^"temporal" in pest); both
/// `TEMPORAL` and `Temporal` and `temporal` must parse identically.
#[test]
fn create_node_type_temporal_case_insensitive() {
    for form in ["TEMPORAL", "Temporal", "temporal"] {
        let q = parse_ok(&format!("CREATE NODE TYPE Person {form}"));
        match &q.clauses[0] {
            Clause::CreateNodeType(c) => {
                assert!(c.temporal, "form '{form}' must set temporal=true")
            }
            other => panic!("expected CreateNodeType for form '{form}', got {other:?}"),
        }
    }
}

#[test]
fn union_branches_parsed() {
    // Plain UNION: one extra branch, not ALL.
    let q = parse_ok("MATCH (a) RETURN a UNION MATCH (b) RETURN b");
    assert_eq!(q.clauses.len(), 2);
    assert_eq!(q.unions.len(), 1);
    assert!(!q.unions[0].all);
    assert_eq!(q.unions[0].clauses.len(), 2);

    // UNION ALL across three branches.
    let q =
        parse_ok("MATCH (a) RETURN a UNION ALL MATCH (b) RETURN b UNION ALL MATCH (c) RETURN c");
    assert_eq!(q.unions.len(), 2);
    assert!(q.unions.iter().all(|b| b.all));
}

#[test]
fn foreach_clause_parsed() {
    use crate::cypher::ast::ForeachClause;
    let q = parse_ok("MATCH (n) FOREACH (x IN [1, 2, 3] | SET n.v = x)");
    let fc = q.clauses.iter().find_map(|c| match c {
        Clause::Foreach(fc) => Some(fc),
        _ => None,
    });
    let ForeachClause { variable, body, .. } = fc.expect("expected a FOREACH clause");
    assert_eq!(variable, "x");
    assert_eq!(body.len(), 1);
    assert!(matches!(body[0], Clause::Set(_, _)));
}

#[test]
fn foreach_multiple_body_clauses_parsed() {
    let q = parse_ok("FOREACH (x IN [1] | CREATE (a:N {v: x}) SET a.y = 2)");
    if let Clause::Foreach(fc) = &q.clauses[0] {
        assert_eq!(fc.body.len(), 2);
        assert!(matches!(fc.body[0], Clause::Create(_)));
        assert!(matches!(fc.body[1], Clause::Set(_, _)));
    } else {
        panic!("expected FOREACH at index 0, got {:?}", q.clauses[0]);
    }
}

#[test]
fn count_subquery_parsed() {
    let q = parse_ok("MATCH (a) RETURN COUNT { MATCH (a)-[:KNOWS]->(b) } AS c");
    // The RETURN item expression is a CountSubquery.
    let has_count = format!("{:?}", q.clauses).contains("CountSubquery");
    assert!(has_count, "expected CountSubquery in {:?}", q.clauses);
}

#[test]
fn collect_subquery_parsed() {
    let q = parse_ok("MATCH (a) RETURN COLLECT { MATCH (a)-[:KNOWS]->(b) RETURN b.name } AS names");
    let has_collect = format!("{:?}", q.clauses).contains("CollectSubquery");
    assert!(has_collect, "expected CollectSubquery in {:?}", q.clauses);
}

#[test]
fn count_collect_functions_still_parse() {
    // count(...) / collect(...) aggregate functions must NOT be captured by the
    // subquery rules (the `{` disambiguates).
    let q = parse_ok("MATCH (a) RETURN count(a), collect(a.name)");
    let dbg = format!("{:?}", q.clauses);
    assert!(dbg.contains("FunctionCall"));
    assert!(!dbg.contains("CountSubquery"));
    assert!(!dbg.contains("CollectSubquery"));
}

#[test]
fn call_subquery_parsed() {
    // Correlated CALL with a leading importing WITH.
    let q =
        parse_ok("MATCH (a:Person) CALL { WITH a MATCH (a)-[:KNOWS]->(b) RETURN b } RETURN a, b");
    let cs = q.clauses.iter().find_map(|c| match c {
        Clause::CallSubquery(cs) => Some(cs),
        _ => None,
    });
    let cs = cs.expect("expected a CALL subquery clause");
    assert!(!cs.optional);
    // body: WITH a, MATCH, RETURN b → at least 3 clauses
    assert!(cs.body.len() >= 3);
    assert!(matches!(cs.body[0], Clause::With(_)));
}

#[test]
fn optional_call_subquery_parsed() {
    let q = parse_ok("MATCH (a) OPTIONAL CALL { MATCH (x:Y) RETURN x } RETURN a, x");
    if let Some(Clause::CallSubquery(cs)) = q
        .clauses
        .iter()
        .find(|c| matches!(c, Clause::CallSubquery(_)))
    {
        assert!(cs.optional);
    } else {
        panic!("expected an OPTIONAL CALL subquery clause");
    }
}

#[test]
fn call_procedure_still_parses() {
    // CALL proc(...) must NOT be captured by the subquery rule (the `{` after
    // CALL disambiguates). `db.test()` routes to a plain procedure call.
    let q = parse_ok("CALL db.test() YIELD value RETURN value");
    assert!(q.clauses.iter().any(|c| matches!(c, Clause::Call(_))));
    assert!(!q
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::CallSubquery(_))));
}

#[test]
fn on_violation_skip_parsed() {
    // SET ... ON VIOLATION SKIP should set ViolationMode::Skip.
    let q = parse_ok("MATCH (n:X) SET n.y = 1 ON VIOLATION SKIP");
    if let Clause::Set(_, violation_mode) = &q.clauses[1] {
        use crate::cypher::ast::ViolationMode;
        assert_eq!(
            *violation_mode,
            ViolationMode::Skip,
            "ON VIOLATION SKIP should set Skip mode"
        );
    } else {
        panic!("expected Clause::Set at index 1, got {:?}", q.clauses[1]);
    }
}
