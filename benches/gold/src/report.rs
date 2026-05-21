//! Shared report-rendering helpers — print the structured per-
//! modality + composite blocks defined in `arch/benchmarks/methodology.md`.

/// One row in a per-modality table.
pub struct Row {
    pub metric: &'static str,
    pub cn: f64,
    pub leader: f64,
    /// `true` for throughput metrics (higher is better → cn/leader).
    /// `false` for latency metrics (lower is better → leader/cn).
    pub higher_is_better: bool,
}

/// Ratio between CoordiNode and the leader for a throughput metric.
/// Higher result = CoordiNode beats the leader.
pub fn throughput_ratio(cn: f64, leader: f64) -> f64 {
    if leader == 0.0 {
        return 0.0;
    }
    cn / leader
}

/// Ratio between leader and CoordiNode for a latency metric. Same
/// interpretation as throughput: higher result = CoordiNode wins.
pub fn latency_ratio(cn_us: f64, leader_us: f64) -> f64 {
    if cn_us == 0.0 {
        return f64::INFINITY;
    }
    leader_us / cn_us
}

impl Row {
    pub fn ratio(&self) -> f64 {
        if self.higher_is_better {
            throughput_ratio(self.cn, self.leader)
        } else {
            latency_ratio(self.cn, self.leader)
        }
    }
}

/// Geometric mean of a non-empty slice of ratios. Returns 0.0 for
/// an empty slice. Negative or NaN inputs are clamped to 0.0 so
/// the report never carries garbage downstream.
pub fn geometric_mean(ratios: &[f64]) -> f64 {
    if ratios.is_empty() {
        return 0.0;
    }
    let clean: Vec<f64> = ratios
        .iter()
        .map(|r| if r.is_finite() && *r > 0.0 { *r } else { 0.0 })
        .collect();
    let product: f64 = clean.iter().product();
    product.powf(1.0 / clean.len() as f64)
}

pub fn format_header(records: u64, operations: u64) -> String {
    format!(
        "═══════════════════════════════════════════════════════════════\n\
         CoordiNode Gold Bench — YCSB ({records} records, {operations} ops)\n\
         Hardware: $(uname -a) — see env\n\
         ═══════════════════════════════════════════════════════════════",
    )
}

pub fn format_modality_block(title: &str, rows: &[Row], leader_label: &str) -> String {
    let mut out = String::new();
    out.push_str("\n┌─────────────────────────────────────────────────────────────┐\n");
    out.push_str(&format!("│ {title:<60}│\n"));
    out.push_str("├──────────────────────┬────────────┬─────────────┬─────────┤\n");
    out.push_str("│ Metric               │ CoordiNode │ Leader      │ Ratio   │\n");
    out.push_str("├──────────────────────┼────────────┼─────────────┼─────────┤\n");
    for row in rows {
        out.push_str(&format!(
            "│ {:<20} │ {:>10.2} │ {:>11.2} │ {:>6.2}x │\n",
            row.metric,
            row.cn,
            row.leader,
            row.ratio(),
        ));
    }
    out.push_str("├──────────────────────┴────────────┴─────────────┴─────────┤\n");
    out.push_str(&format!("│ Leader baseline: {leader_label:<43}│\n"));
    let ratios: Vec<f64> = rows.iter().map(|r| r.ratio()).collect();
    out.push_str(&format!(
        "│ Modality score (geometric mean): {:.3}x vs leader            │\n",
        geometric_mean(&ratios),
    ));
    out.push_str("└─────────────────────────────────────────────────────────────┘");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_mean_empty_returns_zero() {
        assert_eq!(geometric_mean(&[]), 0.0);
    }

    #[test]
    fn geometric_mean_single_value_is_identity() {
        assert!((geometric_mean(&[2.5]) - 2.5).abs() < 1e-9);
    }

    #[test]
    fn geometric_mean_two_values_is_sqrt_of_product() {
        // sqrt(2 * 8) = sqrt(16) = 4
        assert!((geometric_mean(&[2.0, 8.0]) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn throughput_ratio_higher_is_better() {
        // CoordiNode 200k ops/s vs Leader 100k → 2.0x
        assert!((throughput_ratio(200_000.0, 100_000.0) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn latency_ratio_lower_is_better() {
        // CoordiNode 100 µs vs Leader 200 µs → 2.0x (we're twice as fast)
        assert!((latency_ratio(100.0, 200.0) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn geometric_mean_clamps_bad_inputs_to_zero() {
        // NaN, negative, infinity → contribute 0 → product → 0 → mean → 0
        assert_eq!(geometric_mean(&[f64::NAN, 2.0]), 0.0);
        assert_eq!(geometric_mean(&[-1.0, 2.0]), 0.0);
        assert_eq!(geometric_mean(&[f64::INFINITY, 2.0]), 0.0);
    }
}
