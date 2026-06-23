use super::*;

#[test]
fn suggestion_display_with_ddl() {
    let s = Suggestion::new(
        SuggestionKind::CreateIndex,
        Severity::Critical,
        "Full label scan on User.email — add index for filtered property",
    )
    .with_ddl("CREATE INDEX user_email ON User(email)");

    let text = s.to_string();
    assert!(text.contains("[CRITICAL]"));
    assert!(text.contains("CREATE INDEX"));
    assert!(text.contains("DDL: CREATE INDEX user_email ON User(email)"));
}

#[test]
fn suggestion_display_with_rewrite() {
    let s = Suggestion::new(
        SuggestionKind::AddDepthBound,
        Severity::Warning,
        "Unbounded traversal *.. may cause exponential fan-out",
    )
    .with_rewrite("MATCH (a)-[:KNOWS*1..10]->(b)");

    let text = s.to_string();
    assert!(text.contains("[WARNING]"));
    assert!(text.contains("Rewrite:"));
}

#[test]
fn severity_ordering() {
    assert!(Severity::Critical > Severity::Warning);
    assert!(Severity::Warning > Severity::Info);
}

#[test]
fn explain_suggest_result_no_suggestions() {
    let result = ExplainSuggestResult {
        explain: "Cost: 10 | Plan: NodeScan".to_string(),
        suggestions: vec![],
    };
    let text = result.to_string();
    assert!(text.contains("No suggestions"));
}

#[test]
fn explain_suggest_result_with_suggestions() {
    let result = ExplainSuggestResult {
        explain: "Cost: 1000".to_string(),
        suggestions: vec![
            Suggestion::new(SuggestionKind::CreateIndex, Severity::Critical, "Add index"),
            Suggestion::new(
                SuggestionKind::AddDepthBound,
                Severity::Warning,
                "Bound traversal",
            ),
        ],
    };
    let text = result.to_string();
    assert!(text.contains("SUGGESTIONS (2):"));
    assert!(text.contains("1."));
    assert!(text.contains("2."));
}
