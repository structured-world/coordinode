use super::*;

/// A live table renders one sample per family; a removed one renders none,
/// and with no table left nothing at all, so no family header is orphaned.
#[test]
fn a_removed_table_leaves_no_series() {
    let metrics = TableMetrics::default();
    metrics.set("trades", 2, 4096);
    metrics.set("orders", 0, 0);
    let mut out = String::new();
    metrics.render(&mut out);
    assert!(out.contains("coordinode_storage_table_backpressure_tier{table=\"trades\"} 2\n"));
    assert!(
        out.contains("coordinode_storage_table_compaction_debt_bytes{table=\"trades\"} 4096\n")
    );
    assert!(out.contains("coordinode_storage_table_backpressure_tier{table=\"orders\"} 0\n"));

    metrics.remove("trades");
    let mut out = String::new();
    metrics.render(&mut out);
    assert!(!out.contains("trades"), "{out}");
    assert!(out.contains("table=\"orders\""), "{out}");

    metrics.remove("orders");
    let mut out = String::new();
    metrics.render(&mut out);
    assert!(out.is_empty(), "{out}");
}

/// A later verdict replaces the earlier one: one sample per table.
#[test]
fn a_table_has_one_sample_per_family() {
    let metrics = TableMetrics::default();
    metrics.set("trades", 1, 10);
    metrics.set("trades", 0, 3);
    let mut out = String::new();
    metrics.render(&mut out);
    assert_eq!(
        out,
        "# HELP coordinode_storage_table_backpressure_tier Write-pressure tier of a STORAGE \
         COLUMNAR table: 0 none, 1 slowdown, 2 stop.\n\
         # TYPE coordinode_storage_table_backpressure_tier gauge\n\
         coordinode_storage_table_backpressure_tier{table=\"trades\"} 0\n\
         # HELP coordinode_storage_table_compaction_debt_bytes Bytes compaction still owes a \
         STORAGE COLUMNAR table's tree.\n\
         # TYPE coordinode_storage_table_compaction_debt_bytes gauge\n\
         coordinode_storage_table_compaction_debt_bytes{table=\"trades\"} 3\n"
    );
}

/// A label value is escaped as the exposition format requires.
#[test]
fn a_table_name_is_escaped() {
    let metrics = TableMetrics::default();
    metrics.set("a\"b\\c\nd", 0, 0);
    let mut out = String::new();
    metrics.render(&mut out);
    assert!(out.contains("{table=\"a\\\"b\\\\c\\nd\"} 0\n"), "{out}");
}

/// A sample replaces the last one whole: a label missing from it, its last
/// node gone, has no series any more.
#[test]
fn a_label_without_nodes_loses_its_series() {
    let counts = NodeCounts::default();
    counts.replace([("Project".to_owned(), 3), ("Task".to_owned(), 5)].into());
    let mut out = String::new();
    counts.render(&mut out);
    assert!(
        out.contains("coordinode_graph_nodes_total{label=\"Project\"} 3\n"),
        "{out}"
    );
    assert!(
        out.contains("coordinode_graph_nodes_total{label=\"Task\"} 5\n"),
        "{out}"
    );

    counts.replace([("Task".to_owned(), 4)].into());
    let mut out = String::new();
    counts.render(&mut out);
    assert!(!out.contains("Project"), "{out}");
    assert!(
        out.contains("coordinode_graph_nodes_total{label=\"Task\"} 4\n"),
        "{out}"
    );

    counts.replace(BTreeMap::new());
    let mut out = String::new();
    counts.render(&mut out);
    assert!(out.is_empty(), "{out}");
}
