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
        Value::List(vec![Value::Int(1), Value::Int(2)]).to_string(),
        "[1, 2]"
    );
}
