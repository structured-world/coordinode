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
    // ALTER TABLE is not lowered yet.
    assert!(
        SqlFrontend::new()
            .parse("ALTER TABLE Account ADD COLUMN age BIGINT")
            .is_err()
    );
    // Joins are not supported yet.
    assert!(
        SqlFrontend::new()
            .parse("SELECT a.x FROM A a JOIN B b ON a.id = b.id")
            .is_err()
    );
    // DROP VIEW is not a table drop.
    assert!(SqlFrontend::new().parse("DROP VIEW v").is_err());
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

/// Fingerprinting preserves the AST's canonical statement separator and hash;
/// avoiding intermediate strings must not change advisor/cache identities.
#[test]
fn sql_multi_statement_fingerprints_keep_the_canonical_separator() {
    let (canonical, hash) = SqlFrontend::new()
        .fingerprint("select id from t;select name from t")
        .expect("canonical statements");
    assert_eq!(canonical, "SELECT id FROM t; SELECT name FROM t");
    assert_eq!(hash, crate::advisor::fingerprint::fingerprint(&canonical));
    assert!(
        SqlFrontend::new()
            .parse("select id from t;select name from t")
            .is_err()
    );
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
fn create_table_lowers_with_primary_key_and_columns() {
    let root = plan("CREATE TABLE Account (id BIGINT PRIMARY KEY, name VARCHAR(64) NOT NULL)");
    let LogicalOp::CreateTable {
        name,
        columns,
        primary_key,
        columnar,
    } = root
    else {
        panic!("expected CreateTable, got {root:?}");
    };
    assert_eq!(name, "Account");
    assert!(!columnar, "SQL CREATE TABLE is row storage");
    assert_eq!(primary_key, vec!["id".to_string()]);
    assert_eq!(columns.len(), 2);
    assert_eq!(columns[0].name, "id");
    assert_eq!(columns[0].type_name, "BIGINT");
    assert_eq!(columns[1].name, "name");
    // VARCHAR(64) normalizes to the base token.
    assert_eq!(columns[1].type_name, "VARCHAR");
    assert!(columns[1].not_null);
}

#[test]
fn create_table_accepts_table_level_primary_key() {
    let root = plan("CREATE TABLE T (a BIGINT, b BIGINT, PRIMARY KEY (a))");
    let LogicalOp::CreateTable { primary_key, .. } = root else {
        panic!("expected CreateTable");
    };
    assert_eq!(primary_key, vec!["a".to_string()]);
}

/// A table without a PRIMARY KEY is keyed by row id: it lowers with no key
/// columns rather than being refused.
#[test]
fn create_table_without_primary_key_is_keyed_by_row_id() {
    let root = plan("CREATE TABLE T (a BIGINT, b BIGINT)");
    let LogicalOp::CreateTable { primary_key, .. } = root else {
        panic!("expected CreateTable");
    };
    assert!(primary_key.is_empty());
}

#[test]
fn drop_table_lowers_to_drop_table() {
    let root = plan("DROP TABLE Account");
    let LogicalOp::DropTable { name } = root else {
        panic!("expected DropTable, got {root:?}");
    };
    assert_eq!(name, "Account");
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

/// A flat expression evades a recursive-descent limit but still creates a
/// recursively formatted/dropped tree. Admission must precede its construction.
#[test]
fn sql_long_flat_chains_are_refused_by_both_entry_points() {
    let query = format!("SELECT {} FROM T", vec!["1"; 1024].join(" + "));
    let frontend = SqlFrontend::new();
    for result in [
        frontend.parse(&query).map(|_| ()),
        frontend.fingerprint(&query).map(|_| ()),
    ] {
        let error = result.expect_err("syntax exceeding the safety budget must be refused");
        assert!(
            error.to_string().contains("SQL structural token limit"),
            "{error}"
        );
    }
}

/// One huge literal consumes one syntax token. Check bytes before tokenization,
/// independently of token/depth limits, and return the same refusal in both APIs.
#[test]
fn sql_input_bytes_are_refused_by_both_entry_points() {
    let query = format!("SELECT '{}' FROM T", "x".repeat(1024 * 1024));
    let frontend = SqlFrontend::new();
    for result in [
        frontend.parse(&query).map(|_| ()),
        frontend.fingerprint(&query).map(|_| ()),
    ] {
        let error = result.expect_err("oversized input must be refused before AST construction");
        assert!(
            error.to_string().contains("SQL input byte limit"),
            "{error}"
        );
    }
}

/// The byte ceiling is inclusive and independent of literal length in tokens.
/// Exercise accepted formatting/lowering/drop at the exact advertised boundary.
#[test]
fn sql_literal_at_the_byte_boundary_is_admitted() {
    on_minimum_stack(|| {
        let overhead = "SELECT '' FROM T".len();
        let query = format!("SELECT '{}' FROM T", "x".repeat(MAX_INPUT_BYTES - overhead));
        assert_eq!(query.len(), MAX_INPUT_BYTES);
        let parsed = SqlFrontend::new()
            .parse(&query)
            .expect("exact byte boundary");
        let (canonical, fingerprint) = SqlFrontend::new().fingerprint(&query).expect("fingerprint");
        assert_eq!(parsed.canonical, canonical);
        assert_eq!(parsed.fingerprint, fingerprint);
    });
}

/// Run the complete public frontend lifecycle on the minimum supported thread
/// stack. Drops happen inside this thread, including on parse/lowering errors.
fn on_minimum_stack(work: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(work)
        .expect("create the minimum-stack host thread")
        .join()
        .expect("the frontend must not panic on the host thread");
}

/// The deepest admitted flat expression exercises formatting, lowering, plan
/// debug/clone and recursive destruction rather than only a parser's counter.
#[test]
fn sql_flat_expression_boundary_keeps_the_host_stack_safe() {
    on_minimum_stack(|| {
        let frontend = SqlFrontend::new();
        // SELECT, 62 operators, FROM = exactly 64 structural tokens.
        let query = format!("SELECT {} FROM T", vec!["1"; 63].join(" + "));
        let parsed = frontend.parse(&query).expect("admitted flat expression");
        let canonical = frontend.fingerprint(&query).expect("admitted fingerprint");
        assert_eq!(parsed.canonical, canonical.0);
        assert_eq!(parsed.fingerprint, canonical.1);
        let cloned = parsed.clone();
        assert!(!format!("{cloned:?}").is_empty());
        drop(cloned);
        drop(parsed);
        let excessive = query.replacen(" FROM T", " + 1 FROM T", 1);
        for result in [
            frontend.parse(&excessive).map(|_| ()),
            frontend.fingerprint(&excessive).map(|_| ()),
        ] {
            assert!(
                result
                    .expect_err("one extra operator exceeds the structural budget")
                    .to_string()
                    .contains("SQL structural token limit")
            );
        }
        // No additional structural token: fail after building the full tree,
        // so the parser's error path must destroy it on this same stack.
        let malformed = format!("{query} trailing_alias unexpected_identifier");
        assert!(frontend.parse(&malformed).is_err());
        assert!(frontend.fingerprint(&malformed).is_err());
        let partial = format!("SELECT {} +", vec!["1"; 63].join(" + "));
        assert!(frontend.parse(&partial).is_err());
        assert!(frontend.fingerprint(&partial).is_err());
    });
}

/// Width and structural depth are independent: a wide projection is admitted
/// through its exact syntax boundary without building a deep expression tree.
#[test]
fn sql_projection_width_has_its_own_boundary() {
    on_minimum_stack(|| {
        let frontend = SqlFrontend::new();
        // SELECT + 2047 identifiers + 2046 commas + FROM + T = 4096 tokens.
        let query = format!("SELECT {} FROM T", vec!["column_name"; 2047].join(", "));
        let parsed = frontend
            .parse(&query)
            .expect("projection at syntax boundary");
        assert!(frontend.fingerprint(&query).is_ok());
        drop(parsed);
        let query = query.replacen(" FROM T", ", extra FROM T", 1);
        for result in [
            frontend.parse(&query).map(|_| ()),
            frontend.fingerprint(&query).map(|_| ()),
        ] {
            assert!(
                result
                    .expect_err("extra column exceeds syntax budget")
                    .to_string()
                    .contains("SQL syntax token limit")
            );
        }
    });
}

/// Flat DDL still consumes the declared structural budget. VALUES also lowers
/// into a recursive operator chain, so exempting its punctuation is unsafe.
#[test]
fn sql_flat_ddl_and_values_obey_the_structural_boundary() {
    on_minimum_stack(|| {
        let frontend = SqlFrontend::new();
        let table = |columns| {
            let columns = (0..columns)
                .map(|index| format!("c{index} VARCHAR(64) NOT NULL"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("CREATE TABLE \"schema_columns\" ({columns})")
        };
        let insert = |rows| {
            format!(
                "INSERT INTO \"rows\" (\"left_column\", \"right_column\") VALUES {}",
                vec!["(1, 'value')"; rows].join(", ")
            )
        };
        // DDL: four surrounding tokens + five per column = 64. INSERT:
        // five surrounding tokens + two per row = 63; another row gives 65.
        for query in [table(12), insert(29)] {
            let parsed = frontend
                .parse(&query)
                .expect("admitted structural boundary");
            let (canonical, fingerprint) = frontend.fingerprint(&query).expect("fingerprint");
            assert_eq!(parsed.canonical, canonical);
            assert_eq!(parsed.fingerprint, fingerprint);
            let cloned = parsed.clone();
            assert!(!format!("{cloned:?}").is_empty());
            drop(cloned);
            drop(parsed);
        }
        for query in [table(13), insert(30)] {
            for result in [
                frontend.parse(&query).map(|_| ()),
                frontend.fingerprint(&query).map(|_| ()),
            ] {
                assert!(
                    result
                        .expect_err("flat syntax still obeys the structural safety ceiling")
                        .to_string()
                        .contains("SQL structural token limit")
                );
            }
        }
    });
}

/// Nested syntax is independently bounded even when it fits the token budget.
/// Unsupported trees must be formatted and dropped safely on refusal too.
#[test]
fn sql_nested_and_unsupported_trees_keep_the_host_stack_safe() {
    on_minimum_stack(|| {
        let frontend = SqlFrontend::new();
        for depth in 1..=64 {
            let query = format!("SELECT {}1{} FROM T", "(".repeat(depth), ")".repeat(depth));
            let parsed = frontend.parse(&query);
            let fingerprint = frontend.fingerprint(&query);
            assert_eq!(parsed.is_ok(), fingerprint.is_ok());
            if depth == 1 {
                assert!(parsed.is_ok());
            } else if depth == 64 {
                assert!(parsed.is_err());
            }
            let unsupported = format!(
                "SELECT {}abs(1){} FROM T",
                "(".repeat(depth),
                ")".repeat(depth)
            );
            assert!(frontend.parse(&unsupported).is_err());
            // Fingerprinting accepts valid unsupported syntax within its budget.
            drop(frontend.fingerprint(&unsupported));
            let broken = format!("SELECT {}1 +", "NOT ".repeat(depth));
            assert!(frontend.parse(&broken).is_err());
            assert!(frontend.fingerprint(&broken).is_err());
        }
    });
}

/// This input fits both token budgets, so only the parser counter can refuse
/// it. A structural preflight must not accidentally mask a missing depth guard.
#[test]
fn sql_parser_recursion_is_bounded_independently_of_lexical_budgets() {
    on_minimum_stack(|| {
        let query = format!("SELECT {}1{} FROM T", "(".repeat(16), ")".repeat(16));
        let frontend = SqlFrontend::new();
        for result in [
            frontend.parse(&query).map(|_| ()),
            frontend.fingerprint(&query).map(|_| ()),
        ] {
            let error = result.expect_err("parser recursion must be refused");
            assert!(
                error.to_string().contains("recursion limit exceeded"),
                "{error}"
            );
        }
    });
}

/// Different SQL grammar branches allocate different recursive AST types.
/// All must refuse or complete, and release partial trees, on the host stack.
#[test]
fn sql_adversarial_grammar_branches_keep_the_host_stack_safe() {
    on_minimum_stack(|| {
        let frontend = SqlFrontend::new();
        for depth in 1..=64 {
            let cases = [
                format!("{}SELECT 1 FROM T", "EXPLAIN ".repeat(depth)),
                format!(
                    "SELECT {}1{} FROM T",
                    "abs(".repeat(depth),
                    ")".repeat(depth)
                ),
                format!(
                    "CREATE TABLE T (x {}INT{})",
                    "ARRAY<".repeat(depth),
                    ">".repeat(depth)
                ),
                format!(
                    "SELECT {}1{} FROM T",
                    "CASE WHEN 1 = 1 THEN ".repeat(depth),
                    " ELSE 0 END".repeat(depth)
                ),
                vec!["SELECT 1 FROM T"; depth].join(" UNION ALL "),
                format!("SELECT {}1 FROM T", "NOT ".repeat(depth)),
                format!(
                    "SELECT {}1{} FROM T",
                    "CAST(".repeat(depth),
                    " AS INT)".repeat(depth)
                ),
            ];
            for query in cases {
                drop(frontend.parse(&query));
                drop(frontend.fingerprint(&query));
                // Force parser-error cleanup at every depth too.
                let malformed = format!("{query} (");
                assert!(frontend.parse(&malformed).is_err());
                assert!(frontend.fingerprint(&malformed).is_err());
            }
        }
    });
}

/// Lexer-aware admission must not treat keyword-like quoted identifiers or
/// punctuation inside literals/comments as recursive syntax.
#[test]
fn sql_literal_and_comment_contents_do_not_consume_structural_budget() {
    on_minimum_stack(|| {
        let query = format!(
            "SELECT '{}' FROM \"SELECT\" /* {} */",
            "(NOT +)".repeat(512),
            "CASE(".repeat(512)
        );
        let parsed = SqlFrontend::new()
            .parse(&query)
            .expect("flat literal and quoted table");
        let (canonical, fingerprint) = SqlFrontend::new().fingerprint(&query).expect("fingerprint");
        assert_eq!(parsed.canonical, canonical);
        assert_eq!(parsed.fingerprint, fingerprint);
    });
}
