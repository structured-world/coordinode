use super::*;

fn open() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let database = Database::open(dir.path()).unwrap();
    (database, dir)
}

/// A created namespace is listed, a session of it opens and holds its own
/// data, and a second creation of the name is refused naming it.
#[test]
fn a_created_namespace_holds_its_own_data() {
    let (database, _dir) = open();
    database.create_namespace("tenant_a").unwrap();
    assert_eq!(
        database
            .namespaces()
            .unwrap()
            .into_iter()
            .map(|n| (n.name, n.dropping))
            .collect::<Vec<_>>(),
        vec![
            (database.namespace().to_owned(), false),
            ("tenant_a".to_owned(), false),
        ]
    );
    let mut tenant = database.session("tenant_a").unwrap();
    tenant.execute_cypher("CREATE (:Only {v: 1})").unwrap();
    let mut own = database;
    assert!(
        own.execute_cypher("MATCH (n:Only) RETURN n")
            .unwrap()
            .is_empty(),
        "another namespace's data is not seen"
    );
    assert!(matches!(
        own.create_namespace("tenant_a"),
        Err(DatabaseError::NamespaceExists(name)) if name == "tenant_a"
    ));
}

/// Cypher and SQL change and list the catalog through the same path as the
/// embedded API: a namespace created by one statement is listed by the
/// other's and opens a session; the existence guard turns a repeat into a
/// row saying nothing changed, and without it the repeat is refused by
/// name; a session does not drop its own namespace.
#[test]
fn statements_change_and_list_the_catalog() {
    let (mut database, _dir) = open();
    let created = database
        .execute_cypher("CREATE NAMESPACE tenant_a")
        .unwrap();
    assert_eq!(
        created[0].get("changed"),
        Some(&coordinode_core::graph::types::Value::Bool(true))
    );
    database.execute_sql("CREATE NAMESPACE tenant_b").unwrap();
    let listed: Vec<String> = database
        .execute_sql("SHOW NAMESPACES")
        .unwrap()
        .into_iter()
        .filter_map(|row| match row.get("name") {
            Some(coordinode_core::graph::types::Value::String(name)) => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        listed,
        vec![
            database.namespace().to_owned(),
            "tenant_a".into(),
            "tenant_b".into()
        ]
    );
    database.session("tenant_b").unwrap();

    let repeat = database
        .execute_cypher("CREATE NAMESPACE tenant_a IF NOT EXISTS")
        .unwrap();
    assert_eq!(
        repeat[0].get("changed"),
        Some(&coordinode_core::graph::types::Value::Bool(false))
    );
    assert!(matches!(
        database.execute_cypher("CREATE NAMESPACE tenant_a"),
        Err(DatabaseError::NamespaceExists(name)) if name == "tenant_a"
    ));
    database.execute_sql("DROP NAMESPACE tenant_a").unwrap();
    assert!(matches!(
        database.execute_sql("DROP NAMESPACE tenant_a"),
        Err(DatabaseError::UnknownNamespace(_))
    ));
    database
        .execute_cypher("DROP NAMESPACE tenant_a IF EXISTS")
        .unwrap();
    let own = format!("DROP NAMESPACE {}", database.namespace());
    assert!(matches!(
        database.execute_cypher(&own),
        Err(DatabaseError::NamespaceInUse(_))
    ));
}

/// A namespace whose last session is gone closes its trees, and a session
/// opened after reads every row the closed one wrote, its schema with it.
/// One a session holds stays open.
#[test]
fn a_namespace_closed_when_idle_reopens_with_its_data() {
    let (database, _dir) = open();
    database.create_namespace("quiet").unwrap();
    database.create_namespace("busy").unwrap();
    let _busy = database.session("busy").unwrap();
    {
        let mut quiet = database.session("quiet").unwrap();
        quiet
            .execute_cypher("CREATE CONSTRAINT kept_v FOR (n:Kept) REQUIRE n.v IS UNIQUE")
            .unwrap();
        quiet.execute_cypher("CREATE (:Kept {v: 7})").unwrap();
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while database.close_idle_namespaces().unwrap() == 0 {
        assert!(std::time::Instant::now() < deadline, "never closed");
    }
    let mut quiet = database.session("quiet").unwrap();
    let rows = quiet
        .execute_cypher("MATCH (n:Kept) WHERE n.v = 7 RETURN n.v AS v")
        .unwrap();
    assert_eq!(rows.len(), 1, "the reopened namespace holds its row");
    assert!(
        matches!(
            quiet.execute_cypher("CREATE (:Kept {v: 7})"),
            Err(DatabaseError::UniqueViolation { .. })
        ),
        "its constraint came back with it"
    );
}

/// A store of several namespaces reopens: the handle works in the first
/// namespace again, and a session of another reads what it held.
#[test]
fn a_store_of_several_namespaces_reopens_on_its_first() {
    let dir = tempfile::tempdir().unwrap();
    let first = {
        let database = Database::open(dir.path()).unwrap();
        database.create_namespace("second").unwrap();
        database
            .session("second")
            .unwrap()
            .execute_cypher("CREATE (:Held {v: 3})")
            .unwrap();
        database.namespace().to_owned()
    };
    let database = Database::open(dir.path()).unwrap();
    assert_eq!(database.namespace(), first);
    let rows = database
        .session("second")
        .unwrap()
        .execute_cypher("MATCH (n:Held) RETURN n.v AS v")
        .unwrap();
    assert_eq!(rows.len(), 1);
}

/// A name no namespace may have is refused before anything is written.
#[test]
fn an_invalid_namespace_name_is_refused() {
    let (database, _dir) = open();
    assert!(matches!(
        database.create_namespace(""),
        Err(DatabaseError::NamespaceRefused { .. })
    ));
    assert_eq!(database.namespaces().unwrap().len(), 1);
}

/// A dropped namespace is named by no new session, its name is free for a
/// new namespace, which starts empty; the namespace a handle works in is
/// not dropped from inside it, and an unknown one is refused by name.
#[test]
fn a_dropped_namespace_is_gone_and_its_name_free() {
    let (database, _dir) = open();
    database.create_namespace("gone").unwrap();
    database
        .session("gone")
        .unwrap()
        .execute_cypher("CREATE (:Kept {v: 1})")
        .unwrap();
    database.drop_namespace("gone").unwrap();
    assert!(matches!(
        database.session("gone"),
        Err(DatabaseError::UnknownNamespace(_))
    ));
    assert!(matches!(
        database.drop_namespace("gone"),
        Err(DatabaseError::UnknownNamespace(_))
    ));
    assert!(matches!(
        database.drop_namespace(database.namespace()),
        Err(DatabaseError::NamespaceInUse(_))
    ));

    database.create_namespace("gone").unwrap();
    let mut again = database.session("gone").unwrap();
    assert!(
        again
            .execute_cypher("MATCH (n:Kept) RETURN n")
            .unwrap()
            .is_empty(),
        "a namespace created under a dropped one's name starts empty"
    );
}
