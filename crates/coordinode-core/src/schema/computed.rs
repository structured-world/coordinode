//! COMPUTED property types: query-time evaluated fields stored as schema metadata.
//!
//! COMPUTED properties add zero storage overhead per node — the computation
//! spec lives in the label schema, and values are evaluated inline during
//! query execution (~10ns per field).
//!
//! Three variants:
//! - **Decay**: value interpolated between initial and target over duration
//! - **Ttl**: countdown timer that triggers background deletion
//! - **VectorDecay**: multiplier applied to similarity scores at query time

use serde::{Deserialize, Serialize};

/// Decay formula for time-based value interpolation.
///
/// All formulas take a normalized time parameter `t ∈ [0, 1]` where
/// `t = min(1, elapsed_seconds / duration_seconds)`.
///
/// CE tier: Linear, Exponential, PowerLaw, Step.
/// EE tier: Ebbinghaus (access-aware, requires merge operator for access_count).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecayFormula {
    /// `f(t) = max(0, 1 - t)` — constant rate, hard cutoff at duration.
    Linear,

    /// `f(t) = e^(-λt)` where λ = ln(2) / half_life_fraction.
    /// `lambda` is pre-computed from desired half-life as fraction of duration.
    Exponential { lambda: f64 },

    /// `f(t) = (1 + t/τ)^(-α)` — fat tail, recent items decay fast.
    /// `tau` is time scale as fraction of duration, `alpha` is decay exponent.
    PowerLaw { tau: f64, alpha: f64 },

    /// `f(t) = { 1.0 if t < 1.0, 0.0 otherwise }` — binary, equivalent to TTL.
    Step,
}

impl DecayFormula {
    /// Evaluate the decay formula at normalized time `t ∈ [0, 1]`.
    ///
    /// Returns a weight in `[0, 1]` where 1.0 = full strength, 0.0 = fully decayed.
    pub fn evaluate(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => (1.0 - t).max(0.0),
            Self::Exponential { lambda } => (-lambda * t).exp(),
            Self::PowerLaw { tau, alpha } => {
                if *tau <= 0.0 {
                    return 0.0;
                }
                (1.0 + t / tau).powf(-alpha)
            }
            Self::Step => {
                if t < 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// Scope of TTL deletion when a COMPUTED TTL field expires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TtlScope {
    /// Delete just this property from the node.
    Field,
    /// Delete this DOCUMENT property and all nested content.
    Subtree,
    /// Delete the entire node and all its edges.
    Node,
}

/// Specification for a COMPUTED property.
///
/// Stored in `LabelSchema` as part of `PropertyDef`. NOT stored per-node —
/// the spec is evaluated at query time using the node's anchor field value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComputedSpec {
    /// Time-based value interpolation.
    ///
    /// `value = initial + (target - initial) × weight`
    /// where `weight = formula.evaluate(elapsed / duration)`.
    Decay {
        /// Decay formula (Linear, Exponential, PowerLaw, Step).
        formula: DecayFormula,
        /// Starting value at elapsed = 0.
        initial: f64,
        /// Target value at elapsed ≥ duration.
        target: f64,
        /// Duration in seconds over which decay occurs.
        duration_secs: u64,
        /// Name of the TIMESTAMP property to measure elapsed time from.
        anchor_field: String,
    },

    /// Auto-delete after duration. Background reaper handles actual deletion.
    ///
    /// Evaluates to seconds remaining (positive Int) or Null (expired).
    Ttl {
        /// Duration in seconds before expiration.
        duration_secs: u64,
        /// Name of the TIMESTAMP property to measure elapsed time from.
        anchor_field: String,
        /// What to delete when expired.
        scope: TtlScope,
        /// For `scope = Subtree`: the DOCUMENT property to delete on expiry.
        ///
        /// The `anchor_field` is the TIMESTAMP trigger; `target_field` is the
        /// content to remove. If `None` (or `scope != Subtree`), the anchor
        /// field itself is removed (same as `scope = Field`).
        ///
        /// Example: `anchor_field = "created_at"`, `target_field = Some("metadata")`
        /// → deletes the `metadata` DOCUMENT property when `created_at` is older
        /// than `duration_secs`, while `created_at` is left intact.
        #[serde(default)]
        target_field: Option<String>,
    },

    /// Multiplier applied to vector similarity scores at query time.
    ///
    /// Evaluates to a float in [0, 1] that the query planner multiplies
    /// with `vector_similarity()` results.
    VectorDecay {
        /// Decay formula for the multiplier.
        formula: DecayFormula,
        /// Duration in seconds over which decay occurs.
        duration_secs: u64,
        /// Name of the TIMESTAMP property to measure elapsed time from.
        anchor_field: String,
    },
}

impl ComputedSpec {
    /// The anchor field name that this computed spec depends on.
    pub fn anchor_field(&self) -> &str {
        match self {
            Self::Decay { anchor_field, .. }
            | Self::Ttl { anchor_field, .. }
            | Self::VectorDecay { anchor_field, .. } => anchor_field,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
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

    #[test]
    fn ttl_scope_roundtrip() {
        for scope in &[TtlScope::Field, TtlScope::Subtree, TtlScope::Node] {
            let bytes = rmp_serde::to_vec(scope).expect("serialize");
            let decoded: TtlScope = rmp_serde::from_slice(&bytes).expect("deserialize");
            assert_eq!(*scope, decoded);
        }
    }
}
