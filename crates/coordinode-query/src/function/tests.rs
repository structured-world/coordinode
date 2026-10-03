use super::*;

/// Every listed name resolves, and to a function of its own table.
#[test]
fn every_listed_name_resolves() {
    for entry in SCALAR_FUNCTIONS {
        assert!(ScalarFn::resolve(entry.name).is_some(), "{}", entry.name);
        assert!(AggregateFn::resolve(entry.name).is_none(), "{}", entry.name);
    }
    for entry in AGGREGATE_FUNCTIONS {
        assert!(AggregateFn::resolve(entry.name).is_some(), "{}", entry.name);
        assert!(ScalarFn::resolve(entry.name).is_none(), "{}", entry.name);
    }
}

/// Function names are case-insensitive, whatever spelling the call uses.
#[test]
fn resolution_ignores_case() {
    assert_eq!(ScalarFn::resolve("TOUPPER"), Some(ScalarFn::ToUpper));
    assert_eq!(ScalarFn::resolve("Coalesce"), Some(ScalarFn::Coalesce));
    assert_eq!(
        ScalarFn::resolve("POINT.DISTANCE"),
        Some(ScalarFn::PointDistance)
    );
    assert_eq!(AggregateFn::resolve("COUNT"), Some(AggregateFn::Count));
    assert_eq!(AggregateFn::resolve("stdevp"), Some(AggregateFn::StDevP));
    assert_eq!(ScalarFn::resolve("nosuch"), None);
}

/// Aliases resolve to the same function.
#[test]
fn aliases_share_one_function() {
    assert_eq!(ScalarFn::resolve("lower"), ScalarFn::resolve("toLower"));
    assert_eq!(ScalarFn::resolve("btrim"), ScalarFn::resolve("trim"));
    assert_eq!(
        ScalarFn::resolve("character_length"),
        ScalarFn::resolve("char_length")
    );
}

/// No name is listed twice, in any casing, so resolution is unambiguous.
#[test]
fn listed_names_are_unique_ignoring_case() {
    let all = catalog();
    for (i, a) in all.iter().enumerate() {
        for b in &all[i + 1..] {
            assert!(
                !a.name.eq_ignore_ascii_case(b.name),
                "{} listed twice",
                a.name
            );
        }
    }
}

/// The listing is in name order and carries full signatures.
#[test]
fn catalog_is_sorted_with_full_signatures() {
    let all = catalog();
    assert!(all.windows(2).all(|w| w[0].name <= w[1].name));
    let upper = all.iter().find(|e| e.name == "toUpper").expect("listed");
    assert_eq!(upper.signature(), "toUpper(input :: STRING) :: STRING");
    assert!(!upper.aggregating);
    let count = all.iter().find(|e| e.name == "count").expect("listed");
    assert!(count.aggregating);
}
