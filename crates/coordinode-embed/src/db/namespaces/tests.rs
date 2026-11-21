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
