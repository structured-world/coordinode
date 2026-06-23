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
mod tests;
