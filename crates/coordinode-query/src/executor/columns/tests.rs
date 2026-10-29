use super::*;
use crate::cypher::parser::parse;
use crate::planner::build_logical_plan;

fn columns(query: &str) -> Option<NodeColumns> {
    let query = parse(query).unwrap_or_else(|e| panic!("parse failed: {e}"));
    let plan = build_logical_plan(&query).unwrap_or_else(|e| panic!("plan failed: {e}"));
    NodeColumns::for_plan(&plan.root)
}

fn wanted(columns: &NodeColumns, var: &str) -> Option<Vec<String>> {
    columns.of(var).map(|set| {
        let mut names: Vec<String> = set.iter().cloned().collect();
        names.sort();
        names
    })
}

/// A read of some properties keeps exactly those, the pattern's filters
/// included.
#[test]
fn a_scan_keeps_the_properties_the_statement_reads() {
    let c = columns("MATCH (n:R {layer: 'x'}) WHERE n.vtype = 1 RETURN n.key AS k").unwrap();
    assert_eq!(wanted(&c, "n").unwrap(), ["key", "layer", "vtype"]);
}

/// A node passed on under its own name keeps only what the later clauses
/// read; one renamed, returned whole or read whole keeps everything.
#[test]
fn a_node_used_whole_keeps_every_property() {
    let c = columns("MATCH (a:R) WITH a WHERE a.p = 1 RETURN a.q AS q").unwrap();
    assert_eq!(wanted(&c, "a").unwrap(), ["p", "q"]);
    for query in [
        "MATCH (a:R) RETURN a",
        "MATCH (a:R) WITH a AS b RETURN b.q AS q",
        "MATCH (a:R) RETURN properties(a) AS p",
        "MATCH (a:R) RETURN a {.q} AS m",
        "MATCH (a:R) RETURN a.q AS q, count(a) AS n",
    ] {
        let c = columns(query).unwrap();
        assert_eq!(wanted(&c, "a"), None, "{query}");
    }
}

/// id() and labels() read columns every row keeps, so they leave the cut.
#[test]
fn id_and_labels_leave_the_cut() {
    let c = columns("MATCH (a:R) RETURN id(a) AS i, labels(a) AS l, a.q AS q").unwrap();
    assert_eq!(wanted(&c, "a").unwrap(), ["q"]);
}

/// A statement holding an operator the analysis does not cover keeps every
/// property of every node.
#[test]
fn an_uncovered_operator_keeps_everything() {
    for query in [
        "MATCH (a:R)-[:T]->(b:R) RETURN a.q AS q",
        "MATCH (a:R) SET a.q = 1",
        "MATCH (a:R) RETURN *",
    ] {
        assert_eq!(columns(query), None, "{query}");
    }
}

/// The join of the correlated search reads each side's own properties.
#[test]
fn both_sides_of_a_join_keep_what_they_read() {
    let c = columns(
        "MATCH (s:R {layer: 'intent'}) WHERE s.vtype = 1 \
         MATCH (a:R) WHERE a.key = s.parent \
         WITH a, s WHERE a.adoption = 1 \
         RETURN DISTINCT a.key AS key ORDER BY key LIMIT 25",
    )
    .unwrap();
    assert_eq!(wanted(&c, "s").unwrap(), ["layer", "parent", "vtype"]);
    assert_eq!(wanted(&c, "a").unwrap(), ["adoption", "key"]);
}
