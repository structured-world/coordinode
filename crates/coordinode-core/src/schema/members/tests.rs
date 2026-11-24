use rmpv::Value;

use super::{MemberError, check_members, same_document};

fn s(text: &str) -> Value {
    Value::from(text)
}

fn map(fields: Vec<(&str, Value)>) -> Value {
    Value::Map(fields.into_iter().map(|(k, v)| (s(k), v)).collect())
}

fn list(items: Vec<Value>) -> Value {
    Value::Array(items)
}

/// One member of the declared list: a record of scalars, a nested record
/// and a field that may be absent.
fn member(node: u64, revision: u64, kind: &str, result: Option<i64>) -> Value {
    let mut fields = vec![
        ("node", Value::from(node)),
        ("revision", Value::from(revision)),
        ("operation", map(vec![("kind", s(kind))])),
    ];
    if let Some(result) = result {
        fields.push(("result", Value::from(result)));
    }
    map(fields)
}

const OPS: &[&[&str]] = &[&["ops"]];

/// Nothing declared.
const NONE: &[&[&str]] = &[];

/// The fields of a map compare by name with nothing declared: a map with
/// its pairs reversed, at the top and one level down, is the same document.
#[test]
fn named_fields_compare_in_any_order() {
    let a = map(vec![
        ("x", Value::from(1)),
        ("y", map(vec![("p", s("a")), ("q", s("b"))])),
    ]);
    let b = map(vec![
        ("y", map(vec![("q", s("b")), ("p", s("a"))])),
        ("x", Value::from(1)),
    ]);
    assert!(same_document(&a, &b, NONE));
    let other = map(vec![
        ("y", map(vec![("q", s("b")), ("p", s("c"))])),
        ("x", Value::from(1)),
    ]);
    assert!(!same_document(&a, &other, NONE), "another field value");
    let fewer = map(vec![("x", Value::from(1))]);
    assert!(!same_document(&a, &fewer, NONE), "a missing field");
}

/// A declared list is the same in any order of its members; the same list
/// with nothing declared, and any list beside it, differs by order.
#[test]
fn only_a_declared_list_ignores_the_order_of_its_members() {
    let doc =
        |ops: Vec<Value>, tags: Vec<Value>| map(vec![("ops", list(ops)), ("tags", list(tags))]);
    let (m1, m2) = (member(1, 7, "put", Some(3)), member(2, 7, "del", None));
    let tags = || vec![s("a"), s("b")];
    let a = doc(vec![m1.clone(), m2.clone()], tags());
    let b = doc(vec![m2.clone(), m1.clone()], tags());
    assert!(same_document(&a, &b, OPS));
    assert!(
        !same_document(&a, &b, NONE),
        "nothing declared: by position"
    );
    let swapped_tags = doc(vec![m1, m2], vec![s("b"), s("a")]);
    assert!(
        !same_document(&a, &swapped_tags, OPS),
        "a list beside the declared one keeps its order"
    );
}

/// A member differing in any one field, a nested one or an absent one
/// included, is another member.
#[test]
fn a_member_differs_by_any_one_of_its_fields() {
    let base = member(1, 7, "put", Some(3));
    let doc = |m: Value| map(vec![("ops", list(vec![member(9, 9, "x", None), m]))]);
    let reversed = |m: Value| map(vec![("ops", list(vec![m, member(9, 9, "x", None)]))]);
    assert!(same_document(
        &doc(base.clone()),
        &reversed(base.clone()),
        OPS
    ));
    for changed in [
        member(2, 7, "put", Some(3)),
        member(1, 8, "put", Some(3)),
        member(1, 7, "del", Some(3)),
        member(1, 7, "put", Some(4)),
        member(1, 7, "put", None),
    ] {
        assert!(
            !same_document(&doc(base.clone()), &reversed(changed.clone()), OPS),
            "{changed}"
        );
    }
    // The widest revision is a member's own value, not a rounded one.
    assert!(!same_document(
        &doc(member(1, u64::MAX, "put", None)),
        &reversed(member(1, u64::MAX - 1, "put", None)),
        OPS
    ));
}

/// A number is the value it represents, whatever holds it: an integer
/// equals the float of the same value and so repeats it in a declared list,
/// and an integer no float holds differs from the float beside it. A binary
/// payload compares by its bytes.
#[test]
fn numbers_compare_by_value_and_binaries_by_bytes() {
    let of = |v: Value| map(vec![("ops", list(vec![v]))]);
    assert!(same_document(
        &of(Value::from(1)),
        &of(Value::F64(1.0)),
        OPS
    ));
    assert!(same_document(
        &of(Value::F32(1.0)),
        &of(Value::F64(1.0)),
        OPS
    ));
    assert!(same_document(
        &of(Value::F64(-0.0)),
        &of(Value::from(0)),
        OPS
    ));
    assert!(!same_document(
        &of(Value::from(1)),
        &of(Value::F64(1.5)),
        OPS
    ));
    assert_eq!(
        check_members(
            &map(vec![("ops", list(vec![Value::from(1), Value::F64(1.0)]))]),
            OPS
        ),
        Err(MemberError::Repeated {
            path: "ops".to_string()
        })
    );
    // 2^53 is a float; 2^53 + 1 is not, and is not rounded to it.
    let exact = 1_i64 << 53;
    assert!(same_document(
        &of(Value::from(exact)),
        &of(Value::F64(exact as f64)),
        OPS
    ));
    assert!(!same_document(
        &of(Value::from(exact + 1)),
        &of(Value::F64(exact as f64)),
        OPS
    ));
    assert_eq!(
        check_members(
            &map(vec![(
                "ops",
                list(vec![Value::from(exact + 1), Value::F64(exact as f64)])
            )]),
            OPS
        ),
        Ok(())
    );
    // The widest integers lie on their side of the floats around them.
    assert!(!same_document(
        &of(Value::from(u64::MAX)),
        &of(Value::F64(18_446_744_073_709_551_616.0)),
        OPS
    ));
    assert!(!same_document(
        &of(Value::from(i64::MIN + 1)),
        &of(Value::F64(i64::MIN as f64)),
        OPS
    ));
    assert!(same_document(
        &of(Value::from(i64::MIN)),
        &of(Value::F64(i64::MIN as f64)),
        OPS
    ));
    // A NaN is the value it is; an infinity is no integer.
    assert!(same_document(
        &of(Value::F64(f64::NAN)),
        &of(Value::F64(f64::NAN)),
        OPS
    ));
    assert!(!same_document(
        &of(Value::F64(f64::INFINITY)),
        &of(Value::from(u64::MAX)),
        OPS
    ));
    assert!(same_document(
        &of(Value::Binary(vec![2, 1])),
        &of(Value::Binary(vec![2, 1])),
        OPS
    ));
    assert!(!same_document(
        &of(Value::Binary(vec![2, 1])),
        &of(Value::Binary(vec![1, 2])),
        OPS
    ));
}

/// A path passes through a list on its way: the list inside every element
/// is the declared one, while the outer list keeps its order.
#[test]
fn a_path_reaches_the_list_inside_every_element() {
    let sets: &[&[&str]] = &[&["steps", "inputs"]];
    let step = |inputs: Vec<i64>| {
        map(vec![(
            "inputs",
            list(inputs.into_iter().map(Value::from).collect()),
        )])
    };
    let doc = |steps: Vec<Value>| map(vec![("steps", list(steps))]);
    assert!(same_document(
        &doc(vec![step(vec![1, 2]), step(vec![3, 4])]),
        &doc(vec![step(vec![2, 1]), step(vec![4, 3])]),
        sets
    ));
    assert!(!same_document(
        &doc(vec![step(vec![1, 2]), step(vec![3, 4])]),
        &doc(vec![step(vec![3, 4]), step(vec![1, 2])]),
        sets
    ));
}

/// A declared list inside a member of a declared list: both ignore order.
#[test]
fn declared_lists_nest() {
    let sets: &[&[&str]] = &[&["ops"], &["ops", "reads"]];
    let op = |id: i64, reads: Vec<i64>| {
        map(vec![
            ("id", Value::from(id)),
            ("reads", list(reads.into_iter().map(Value::from).collect())),
        ])
    };
    let doc = |ops: Vec<Value>| map(vec![("ops", list(ops))]);
    assert!(same_document(
        &doc(vec![op(1, vec![1, 2]), op(2, vec![3, 4])]),
        &doc(vec![op(2, vec![4, 3]), op(1, vec![2, 1])]),
        sets
    ));
    // The same members with the inner order counted are two different ones.
    assert_eq!(
        check_members(&doc(vec![op(1, vec![1, 2]), op(1, vec![2, 1])]), sets),
        Err(MemberError::Repeated {
            path: "ops".to_string()
        })
    );
}

/// A member held twice is refused with the path of its list, whatever the
/// order of its fields; members that differ are admitted.
#[test]
fn a_repeated_member_is_refused_with_its_path() {
    let twice = map(vec![(
        "ops",
        list(vec![
            map(vec![("a", Value::from(1)), ("b", Value::from(2))]),
            map(vec![("b", Value::from(2)), ("a", Value::from(1))]),
        ]),
    )]);
    assert_eq!(
        check_members(&twice, OPS),
        Err(MemberError::Repeated {
            path: "ops".to_string()
        })
    );
    assert_eq!(check_members(&twice, NONE), Ok(()), "nothing declared");
    let nested: &[&[&str]] = &[&["a", "b"]];
    let deep = map(vec![(
        "a",
        map(vec![("b", list(vec![s("x"), s("y"), s("x")]))]),
    )]);
    assert_eq!(
        check_members(&deep, nested),
        Err(MemberError::Repeated {
            path: "a.b".to_string()
        })
    );
    let once = map(vec![(
        "ops",
        list(vec![
            member(1, 1, "put", None),
            member(1, 1, "put", Some(0)),
        ]),
    )]);
    assert_eq!(check_members(&once, OPS), Ok(()));
}

/// The value itself may be the declared list; an absent or null list is
/// admitted, any other kind of value at a declared path is not.
#[test]
fn what_stands_at_a_declared_path_is_a_list_or_absent() {
    let root: &[&[&str]] = &[&[]];
    assert_eq!(
        check_members(&list(vec![s("a"), s("a")]), root),
        Err(MemberError::Repeated {
            path: String::new()
        })
    );
    assert!(same_document(
        &list(vec![s("a"), s("b")]),
        &list(vec![s("b"), s("a")]),
        root
    ));
    assert_eq!(check_members(&map(vec![]), OPS), Ok(()), "absent");
    assert_eq!(
        check_members(&map(vec![("ops", Value::Nil)]), OPS),
        Ok(()),
        "null"
    );
    assert_eq!(
        check_members(&map(vec![("ops", s("x"))]), OPS),
        Err(MemberError::NotAList {
            path: "ops".to_string()
        })
    );
}
