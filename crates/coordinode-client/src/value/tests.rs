use super::*;

/// Round-trip: proto → Value → proto preserves scalar types.
#[test]
fn round_trip_scalars() {
    let cases: Vec<PropertyValue> = vec![
        PropertyValue {
            value: Some(property_value::Value::IntValue(42)),
        },
        PropertyValue {
            value: Some(property_value::Value::FloatValue(1.5)),
        },
        PropertyValue {
            value: Some(property_value::Value::StringValue("hello".into())),
        },
        PropertyValue {
            value: Some(property_value::Value::BoolValue(true)),
        },
        PropertyValue { value: None },
        // A timestamp stays a timestamp, including one before 1970.
        PropertyValue {
            value: Some(property_value::Value::TimestampValue(
                crate::proto::common::Timestamp {
                    unix_micros: -86_400_000_000,
                },
            )),
        },
    ];

    for pv in cases {
        let v = from_proto(pv.clone());
        let back = to_proto(v);
        assert_eq!(pv, back);
    }
}

/// Display impl produces readable output.
#[test]
fn display_values() {
    assert_eq!(Value::Null.to_string(), "null");
    assert_eq!(Value::Bool(false).to_string(), "false");
    assert_eq!(Value::Int(-7).to_string(), "-7");
    assert_eq!(Value::String("hi".into()).to_string(), "hi");
    assert_eq!(
        Value::Timestamp(1_700_000_000_123_456).to_string(),
        "2023-11-14T22:13:20.123456Z"
    );
    assert_eq!(
        Value::Timestamp(-1).to_string(),
        "1969-12-31T23:59:59.999999Z"
    );
    assert_eq!(
        Value::Timestamp(951_782_400_000_000).to_string(),
        "2000-02-29T00:00:00.000000Z"
    );
    assert_eq!(
        Value::Timestamp(-62_135_596_800_000_000).to_string(),
        "0001-01-01T00:00:00.000000Z"
    );
    // The extremes of the range format without overflow, in expanded years.
    assert!(
        Value::Timestamp(i64::MIN)
            .to_string()
            .starts_with("-290308-")
    );
    assert!(
        Value::Timestamp(i64::MAX)
            .to_string()
            .starts_with("+294247-")
    );
    assert_eq!(
        Value::List(vec![Value::Int(1), Value::Int(2)]).to_string(),
        "[1, 2]"
    );
}
