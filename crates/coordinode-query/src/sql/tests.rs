use super::*;

fn plan(sql: &str) -> LogicalOp {
    SqlFrontend::new().parse(sql).expect("parse sql").plan.root
}

#[test]
fn select_lowers_to_project_filter_nodescan() {
    let root = plan("SELECT id, name FROM Account WHERE id = 1");
    let LogicalOp::Project { input, items, .. } = root else {
        panic!("expected Project, got {root:?}");
    };
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].alias.as_deref(), Some("id"));
    let LogicalOp::Filter { input, predicate } = *input else {
        panic!("expected Filter");
    };
    assert!(matches!(predicate, Expr::Binary { op: BinOp::Eq, .. }));
    let LogicalOp::NodeScan {
        labels, variable, ..
    } = *input
    else {
        panic!("expected NodeScan");
    };
    assert_eq!(labels, vec!["Account".to_string()]);
    assert_eq!(variable, "Account");
}

#[test]
fn select_without_where_has_no_filter() {
    let root = plan("SELECT name FROM Account");
    let LogicalOp::Project { input, .. } = root else {
        panic!("expected Project");
    };
    assert!(matches!(*input, LogicalOp::NodeScan { .. }));
}

#[test]
fn select_star_projects_the_row_variable() {
    let root = plan("SELECT * FROM Trade");
    let LogicalOp::Project { items, .. } = root else {
        panic!("expected Project");
    };
    assert_eq!(items.len(), 1);
    assert!(matches!(items[0].expr, Expr::Variable(ref v) if v == "Trade"));
}

#[test]
fn select_alias_binds_table_variable() {
    let root = plan("SELECT t.qty FROM Trade AS t WHERE t.qty > 100");
    let LogicalOp::Project { input, .. } = root else {
        panic!("expected Project");
    };
    let LogicalOp::Filter { input, .. } = *input else {
        panic!("expected Filter");
    };
    let LogicalOp::NodeScan { variable, .. } = *input else {
        panic!("expected NodeScan");
    };
    assert_eq!(variable, "t");
}

#[test]
fn insert_lowers_to_create_node() {
    let root = plan("INSERT INTO Account (id, name) VALUES (1, 'Alice')");
    let LogicalOp::CreateNode {
        labels, properties, ..
    } = root
    else {
        panic!("expected CreateNode, got {root:?}");
    };
    assert_eq!(labels, vec!["Account".to_string()]);
    assert_eq!(properties.len(), 2);
    assert_eq!(properties[0].0, "id");
    assert!(matches!(properties[0].1, Expr::Literal(Value::Int(1))));
    assert_eq!(properties[1].0, "name");
    assert!(matches!(properties[1].1, Expr::Literal(Value::String(ref s)) if s == "Alice"));
}

#[test]
fn unsupported_statement_is_rejected() {
    assert!(SqlFrontend::new().parse("DROP TABLE Account").is_err());
    // Joins are not supported yet.
    assert!(SqlFrontend::new()
        .parse("SELECT a.x FROM A a JOIN B b ON a.id = b.id")
        .is_err());
}

#[test]
fn same_sql_has_stable_fingerprint() {
    let fe = SqlFrontend::new();
    let a = fe
        .fingerprint("select id from account where id = 1")
        .unwrap();
    let b = fe
        .fingerprint("SELECT id FROM account WHERE id = 1")
        .unwrap();
    // Canonicalized through the AST -> identical fingerprint regardless of case.
    assert_eq!(a.1, b.1);
}

#[test]
fn update_lowers_to_update_over_filter_nodescan() {
    let root = plan("UPDATE Account SET name = 'Bob' WHERE id = 1");
    let LogicalOp::Update { input, items, .. } = root else {
        panic!("expected Update, got {root:?}");
    };
    assert_eq!(items.len(), 1);
    match &items[0] {
        crate::plan::SetItem::Property { property, expr, .. } => {
            assert_eq!(property, "name");
            assert!(matches!(expr, Expr::Literal(Value::String(ref s)) if s == "Bob"));
        }
        other => panic!("expected SetItem::Property, got {other:?}"),
    }
    assert!(matches!(*input, LogicalOp::Filter { .. }));
}

#[test]
fn delete_lowers_to_delete_over_filter_nodescan() {
    let root = plan("DELETE FROM Account WHERE id = 1");
    let LogicalOp::Delete {
        input, variables, ..
    } = root
    else {
        panic!("expected Delete, got {root:?}");
    };
    assert_eq!(variables, vec!["Account".to_string()]);
    let LogicalOp::Filter { input, .. } = *input else {
        panic!("expected Filter");
    };
    assert!(matches!(*input, LogicalOp::NodeScan { .. }));
}
