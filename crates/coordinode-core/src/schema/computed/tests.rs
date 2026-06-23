use super::*;

// ── DecayFormula::evaluate ────────────────────────────────────

#[test]
fn linear_at_boundaries() {
    assert!((DecayFormula::Linear.evaluate(0.0) - 1.0).abs() < f64::EPSILON);
    assert!((DecayFormula::Linear.evaluate(1.0) - 0.0).abs() < f64::EPSILON);
    assert!((DecayFormula::Linear.evaluate(0.5) - 0.5).abs() < f64::EPSILON);
}

#[test]
fn linear_clamps_beyond_duration() {
    assert!((DecayFormula::Linear.evaluate(1.5) - 0.0).abs() < f64::EPSILON);
    assert!((DecayFormula::Linear.evaluate(-0.5) - 1.0).abs() < f64::EPSILON);
}

#[test]
fn exponential_at_zero() {
    let f = DecayFormula::Exponential { lambda: 1.0 };
    assert!((f.evaluate(0.0) - 1.0).abs() < f64::EPSILON);
}

#[test]
fn exponential_decays_monotonically() {
    let f = DecayFormula::Exponential { lambda: 2.0 };
    let v0 = f.evaluate(0.0);
    let v1 = f.evaluate(0.25);
    let v2 = f.evaluate(0.5);
    let v3 = f.evaluate(0.75);
    let v4 = f.evaluate(1.0);
    assert!(v0 > v1);
    assert!(v1 > v2);
    assert!(v2 > v3);
    assert!(v3 > v4);
    assert!(v4 > 0.0);
}

#[test]
fn power_law_at_zero() {
    let f = DecayFormula::PowerLaw {
        tau: 0.5,
        alpha: 1.0,
    };
    assert!((f.evaluate(0.0) - 1.0).abs() < f64::EPSILON);
}

#[test]
fn power_law_fat_tail() {
    // Power law should decay slower than exponential at high t
    let pl = DecayFormula::PowerLaw {
        tau: 0.5,
        alpha: 1.0,
    };
    let exp = DecayFormula::Exponential { lambda: 2.0 };
    // At t=1.0, power law has fat tail (higher residual)
    assert!(pl.evaluate(1.0) > exp.evaluate(1.0));
}

#[test]
fn power_law_zero_tau_returns_zero() {
    let f = DecayFormula::PowerLaw {
        tau: 0.0,
        alpha: 1.0,
    };
    assert!((f.evaluate(0.5) - 0.0).abs() < f64::EPSILON);
}

#[test]
fn step_binary() {
    assert!((DecayFormula::Step.evaluate(0.0) - 1.0).abs() < f64::EPSILON);
    assert!((DecayFormula::Step.evaluate(0.99) - 1.0).abs() < f64::EPSILON);
    assert!((DecayFormula::Step.evaluate(1.0) - 0.0).abs() < f64::EPSILON);
}

// ── ComputedSpec ─────────────────────────────────────────────

#[test]
fn computed_spec_anchor_field() {
    let decay = ComputedSpec::Decay {
        formula: DecayFormula::Linear,
        initial: 1.0,
        target: 0.0,
        duration_secs: 86400,
        anchor_field: "created_at".into(),
    };
    assert_eq!(decay.anchor_field(), "created_at");

    let ttl = ComputedSpec::Ttl {
        duration_secs: 3600,
        anchor_field: "expires_at".into(),
        scope: TtlScope::Node,
        target_field: None,
    };
    assert_eq!(ttl.anchor_field(), "expires_at");
}

// ── Serialization roundtrip ──────────────────────────────────

#[test]
fn decay_formula_msgpack_roundtrip() {
    let formulas = vec![
        DecayFormula::Linear,
        DecayFormula::Exponential { lambda: 0.693 },
        DecayFormula::PowerLaw {
            tau: 0.5,
            alpha: 1.5,
        },
        DecayFormula::Step,
    ];
    for f in &formulas {
        let bytes = rmp_serde::to_vec(f).expect("serialize");
        let decoded: DecayFormula = rmp_serde::from_slice(&bytes).expect("deserialize");
        assert_eq!(*f, decoded);
    }
}

#[test]
fn computed_spec_msgpack_roundtrip() {
    let specs = vec![
        ComputedSpec::Decay {
            formula: DecayFormula::Exponential { lambda: 0.001 },
            initial: 1.0,
            target: 0.0,
            duration_secs: 604800,
            anchor_field: "created_at".into(),
        },
        ComputedSpec::Ttl {
            duration_secs: 2592000,
            anchor_field: "created_at".into(),
            scope: TtlScope::Node,
            target_field: None,
        },
        ComputedSpec::VectorDecay {
            formula: DecayFormula::PowerLaw {
                tau: 0.5,
                alpha: 1.0,
            },
            duration_secs: 1209600,
            anchor_field: "indexed_at".into(),
        },
    ];
    for spec in &specs {
        let bytes = rmp_serde::to_vec(spec).expect("serialize");
        let decoded: ComputedSpec = rmp_serde::from_slice(&bytes).expect("deserialize");
        assert_eq!(*spec, decoded);
    }
}

/// `target_field = Some(...)` must survive msgpack roundtrip.
#[test]
fn ttl_with_target_field_msgpack_roundtrip() {
    let spec = ComputedSpec::Ttl {
        duration_secs: 3600,
        anchor_field: "created_at".into(),
        scope: TtlScope::Subtree,
        target_field: Some("profile_data".into()),
    };
    let bytes = rmp_serde::to_vec(&spec).expect("serialize");
    let decoded: ComputedSpec = rmp_serde::from_slice(&bytes).expect("deserialize");
    assert_eq!(spec, decoded);
}

#[test]
fn ttl_scope_roundtrip() {
    for scope in &[TtlScope::Field, TtlScope::Subtree, TtlScope::Node] {
        let bytes = rmp_serde::to_vec(scope).expect("serialize");
        let decoded: TtlScope = rmp_serde::from_slice(&bytes).expect("deserialize");
        assert_eq!(*scope, decoded);
    }
}
