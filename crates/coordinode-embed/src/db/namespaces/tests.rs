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
