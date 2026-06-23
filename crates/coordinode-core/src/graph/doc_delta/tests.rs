use super::*;

fn make_map(entries: Vec<(&str, rmpv::Value)>) -> rmpv::Value {
    rmpv::Value::Map(
        entries
            .into_iter()
            .map(|(k, v)| (rmpv::Value::String(k.into()), v))
            .collect(),
    )
}

// --- SetPath tests ---

#[test]
fn set_path_single_level() {
    let mut doc = make_map(vec![("name", rmpv::Value::String("Alice".into()))]);
    let delta = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["name".into()],
        value: rmpv::Value::String("Bob".into()),
    };
    assert!(delta.apply(&mut doc));
    assert_eq!(
        doc,
        make_map(vec![("name", rmpv::Value::String("Bob".into()))])
    );
}

#[test]
fn set_path_empty_path_replaces_root() {
    // SetPath with an empty path replaces the entire document at the root.
    // Used by e.g. ATTACH DOCUMENT when promoting a graph node back into a
    // single-segment nested property (`PropField(fid)` + empty subpath).
    let mut doc = make_map(vec![("old", rmpv::Value::Boolean(true))]);
    let replacement = make_map(vec![
        ("city", rmpv::Value::String("Prague".into())),
        ("zip", rmpv::Value::String("11000".into())),
    ]);
    let delta = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec![],
        value: replacement.clone(),
    };
    assert!(delta.apply(&mut doc));
    assert_eq!(doc, replacement);
}

#[test]
fn set_path_creates_intermediates() {
    let mut doc = make_map(vec![]);
    let delta = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["config".into(), "network".into(), "ssid".into()],
        value: rmpv::Value::String("home".into()),
    };
    assert!(delta.apply(&mut doc));

    let result = super::super::document::extract_at_path(&doc, &["config", "network", "ssid"]);
    assert_eq!(result, rmpv::Value::String("home".into()));
}

#[test]
fn set_path_overwrites_existing() {
    let mut doc = make_map(vec![(
        "a",
        make_map(vec![("b", rmpv::Value::Integer(1.into()))]),
    )]);
    let delta = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["a".into(), "b".into()],
        value: rmpv::Value::Integer(42.into()),
    };
    assert!(delta.apply(&mut doc));
    assert_eq!(
        super::super::document::extract_at_path(&doc, &["a", "b"]),
        rmpv::Value::Integer(42.into())
    );
}

#[test]
fn set_path_empty_path_replaces_any_root_value() {
    // Empty path replaces the root regardless of its prior shape —
    // nil, scalar, map, or array. Used by ATTACH DOCUMENT to promote
    // a graph node's properties into a single-segment nested field.
    let mut doc = rmpv::Value::Nil;
    let delta = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec![],
        value: rmpv::Value::Integer(1.into()),
    };
    assert!(delta.apply(&mut doc));
    assert_eq!(doc, rmpv::Value::Integer(1.into()));

    // And it replaces scalar roots just as well.
    let mut doc = rmpv::Value::String("stale".into());
    let delta = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec![],
        value: rmpv::Value::Boolean(true),
    };
    assert!(delta.apply(&mut doc));
    assert_eq!(doc, rmpv::Value::Boolean(true));
}

// --- DeletePath tests ---

#[test]
fn delete_path_removes_key() {
    let mut doc = make_map(vec![
        ("a", rmpv::Value::Integer(1.into())),
        ("b", rmpv::Value::Integer(2.into())),
    ]);
    let delta = DocDelta::DeletePath {
        target: PathTarget::Extra,
        path: vec!["a".into()],
    };
    assert!(delta.apply(&mut doc));
    assert_eq!(doc, make_map(vec![("b", rmpv::Value::Integer(2.into()))]));
}

#[test]
fn delete_path_nested() {
    let mut doc = make_map(vec![(
        "config",
        make_map(vec![
            ("a", rmpv::Value::Integer(1.into())),
            ("b", rmpv::Value::Integer(2.into())),
        ]),
    )]);
    let delta = DocDelta::DeletePath {
        target: PathTarget::Extra,
        path: vec!["config".into(), "a".into()],
    };
    assert!(delta.apply(&mut doc));
    assert_eq!(
        super::super::document::extract_at_path(&doc, &["config", "b"]),
        rmpv::Value::Integer(2.into())
    );
    assert_eq!(
        super::super::document::extract_at_path(&doc, &["config", "a"]),
        rmpv::Value::Nil
    );
}

#[test]
fn delete_path_missing_is_noop() {
    let mut doc = make_map(vec![("a", rmpv::Value::Integer(1.into()))]);
    let delta = DocDelta::DeletePath {
        target: PathTarget::Extra,
        path: vec!["nonexistent".into()],
    };
    assert!(!delta.apply(&mut doc));
    assert_eq!(doc, make_map(vec![("a", rmpv::Value::Integer(1.into()))]));
}

// --- ArrayPush tests ---

#[test]
fn array_push_to_existing() {
    let mut doc = make_map(vec![(
        "tags",
        rmpv::Value::Array(vec![rmpv::Value::String("a".into())]),
    )]);
    let delta = DocDelta::ArrayPush {
        target: PathTarget::Extra,
        path: vec!["tags".into()],
        value: rmpv::Value::String("b".into()),
    };
    assert!(delta.apply(&mut doc));
    assert_eq!(
        doc,
        make_map(vec![(
            "tags",
            rmpv::Value::Array(vec![
                rmpv::Value::String("a".into()),
                rmpv::Value::String("b".into()),
            ])
        )])
    );
}

#[test]
fn array_push_creates_array() {
    let mut doc = make_map(vec![]);
    let delta = DocDelta::ArrayPush {
        target: PathTarget::Extra,
        path: vec!["tags".into()],
        value: rmpv::Value::String("first".into()),
    };
    assert!(delta.apply(&mut doc));
    assert_eq!(
        doc,
        make_map(vec![(
            "tags",
            rmpv::Value::Array(vec![rmpv::Value::String("first".into())])
        )])
    );
}

// --- ArrayPull tests ---

#[test]
fn array_pull_removes_first_match() {
    let mut doc = make_map(vec![(
        "tags",
        rmpv::Value::Array(vec![
            rmpv::Value::String("a".into()),
            rmpv::Value::String("b".into()),
            rmpv::Value::String("a".into()),
        ]),
    )]);
    let delta = DocDelta::ArrayPull {
        target: PathTarget::Extra,
        path: vec!["tags".into()],
        value: rmpv::Value::String("a".into()),
    };
    assert!(delta.apply(&mut doc));
    // Only first "a" removed.
    assert_eq!(
        doc,
        make_map(vec![(
            "tags",
            rmpv::Value::Array(vec![
                rmpv::Value::String("b".into()),
                rmpv::Value::String("a".into()),
            ])
        )])
    );
}

#[test]
fn array_pull_missing_value_noop() {
    let mut doc = make_map(vec![(
        "tags",
        rmpv::Value::Array(vec![rmpv::Value::String("a".into())]),
    )]);
    let delta = DocDelta::ArrayPull {
        target: PathTarget::Extra,
        path: vec!["tags".into()],
        value: rmpv::Value::String("z".into()),
    };
    assert!(!delta.apply(&mut doc));
}

// --- ArrayAddToSet tests ---

#[test]
fn add_to_set_adds_new() {
    let mut doc = make_map(vec![(
        "tags",
        rmpv::Value::Array(vec![rmpv::Value::String("a".into())]),
    )]);
    let delta = DocDelta::ArrayAddToSet {
        target: PathTarget::Extra,
        path: vec!["tags".into()],
        value: rmpv::Value::String("b".into()),
    };
    assert!(delta.apply(&mut doc));
    assert_eq!(
        doc,
        make_map(vec![(
            "tags",
            rmpv::Value::Array(vec![
                rmpv::Value::String("a".into()),
                rmpv::Value::String("b".into()),
            ])
        )])
    );
}

#[test]
fn add_to_set_skips_duplicate() {
    let mut doc = make_map(vec![(
        "tags",
        rmpv::Value::Array(vec![rmpv::Value::String("a".into())]),
    )]);
    let delta = DocDelta::ArrayAddToSet {
        target: PathTarget::Extra,
        path: vec!["tags".into()],
        value: rmpv::Value::String("a".into()),
    };
    assert!(!delta.apply(&mut doc));
    // Array unchanged.
    assert_eq!(
        doc,
        make_map(vec![(
            "tags",
            rmpv::Value::Array(vec![rmpv::Value::String("a".into())])
        )])
    );
}

// --- Increment tests ---

#[test]
fn increment_integer() {
    let mut doc = make_map(vec![(
        "stats",
        make_map(vec![("views", rmpv::Value::Integer(10.into()))]),
    )]);
    let delta = DocDelta::Increment {
        target: PathTarget::Extra,
        path: vec!["stats".into(), "views".into()],
        amount: 5.0,
    };
    assert!(delta.apply(&mut doc));
    assert_eq!(
        super::super::document::extract_at_path(&doc, &["stats", "views"]),
        rmpv::Value::Integer(15.into())
    );
}

#[test]
fn increment_float() {
    let mut doc = make_map(vec![("score", rmpv::Value::F64(1.5))]);
    let delta = DocDelta::Increment {
        target: PathTarget::Extra,
        path: vec!["score".into()],
        amount: 0.5,
    };
    assert!(delta.apply(&mut doc));
    assert_eq!(doc, make_map(vec![("score", rmpv::Value::F64(2.0))]));
}

#[test]
fn increment_from_nil() {
    let mut doc = make_map(vec![]);
    let delta = DocDelta::Increment {
        target: PathTarget::Extra,
        path: vec!["counter".into()],
        amount: 1.0,
    };
    assert!(delta.apply(&mut doc));
    assert_eq!(
        doc,
        make_map(vec![("counter", rmpv::Value::Integer(1.into()))])
    );
}

#[test]
fn increment_fractional_promotes_to_float() {
    let mut doc = make_map(vec![("val", rmpv::Value::Integer(10.into()))]);
    let delta = DocDelta::Increment {
        target: PathTarget::Extra,
        path: vec!["val".into()],
        amount: 0.5,
    };
    assert!(delta.apply(&mut doc));
    assert_eq!(doc, make_map(vec![("val", rmpv::Value::F64(10.5))]));
}

// --- Encode/decode roundtrip ---

#[test]
fn encode_decode_roundtrip() {
    let deltas = vec![
        DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["a".into(), "b".into()],
            value: rmpv::Value::Integer(42.into()),
        },
        DocDelta::DeletePath {
            target: PathTarget::Extra,
            path: vec!["x".into()],
        },
        DocDelta::ArrayPush {
            target: PathTarget::Extra,
            path: vec!["tags".into()],
            value: rmpv::Value::String("new".into()),
        },
        DocDelta::ArrayPull {
            target: PathTarget::Extra,
            path: vec!["tags".into()],
            value: rmpv::Value::String("old".into()),
        },
        DocDelta::ArrayAddToSet {
            target: PathTarget::Extra,
            path: vec!["tags".into()],
            value: rmpv::Value::String("unique".into()),
        },
        DocDelta::Increment {
            target: PathTarget::Extra,
            path: vec!["count".into()],
            amount: 1.0,
        },
    ];

    for delta in &deltas {
        let encoded = delta.encode().expect("encode failed");
        assert_eq!(encoded[0], PREFIX_DOC_DELTA);
        let decoded = DocDelta::decode(&encoded[1..]).expect("decode failed");
        assert_eq!(&decoded, delta, "roundtrip failed for {delta:?}");
    }
}
