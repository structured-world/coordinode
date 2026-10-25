use super::*;

/// The canonical tree as its definition states it: adjacent terms added
/// pairwise level by level, an odd last term carried to the next level.
fn levelwise(mut terms: Vec<f64>) -> f64 {
    if terms.is_empty() {
        return 0.0;
    }
    while terms.len() > 1 {
        let mut next = Vec::with_capacity(terms.len().div_ceil(2));
        let (pairs, odd) = terms.as_chunks::<2>();
        for pair in pairs {
            next.push(pair[0] + pair[1]);
        }
        next.extend_from_slice(odd);
        terms = next;
    }
    terms[0]
}

/// The mathematical sum of `terms`, correctly rounded: Shewchuk's exact
/// expansion, an oracle independent of the tree.
fn exact_sum(terms: &[f64]) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for &t in terms {
        let mut x = t;
        let mut kept = 0;
        for i in 0..partials.len() {
            let mut y = partials[i];
            if x.abs() < y.abs() {
                core::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[kept] = lo;
                kept += 1;
            }
            x = hi;
        }
        partials.truncate(kept);
        partials.push(x);
    }
    // Round the expansion to one f64, from the most significant part down.
    let mut hi = 0.0f64;
    let mut lo = 0.0f64;
    while let Some(x) = partials.pop() {
        let prev = hi;
        hi = prev + x;
        lo = x - (hi - prev);
        if lo != 0.0 {
            break;
        }
    }
    if let Some(&next) = partials.last() {
        if (lo < 0.0 && next < 0.0) || (lo > 0.0 && next > 0.0) {
            let y = lo * 2.0;
            let x = hi + y;
            if y == x - hi {
                hi = x;
            }
        }
    }
    hi
}

/// Deterministic values covering ordinary, tiny, subnormal, huge and signed
/// zero components.
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let mantissa = ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
            match s % 11 {
                0 => mantissa * 1e-40, // subnormal
                1 => mantissa * 1e-25, // squares underflow f32
                2 => mantissa * 3e38,  // squares overflow f32
                3 => mantissa * 1e20,  // products overflow f32
                4 => -0.0,
                5 => 0.0,
                _ => mantissa,
            }
        })
        .collect()
}

/// Every pair-kernel output, by name, for one kernel set.
fn outputs(k: Kernels, a: &[f32], b: &[f32]) -> [(&'static str, u64); 7] {
    let (dot, ssa, ssb) = k.cosine_parts(a, b);
    [
        ("dot", k.dot(a, b).to_bits()),
        ("squared_l2", k.squared_l2(a, b).to_bits()),
        ("l1", k.l1(a, b).to_bits()),
        ("sum_of_squares", k.sum_of_squares(a).to_bits()),
        ("cosine.dot", dot.to_bits()),
        ("cosine.ssa", ssa.to_bits()),
        ("cosine.ssb", ssb.to_bits()),
    ]
}

/// The block-and-counter reduction is the canonical tree for every length,
/// including lengths around whole blocks and powers of two.
#[test]
fn the_reduction_is_the_levelwise_tree() {
    for n in (0..=300).chain([511, 512, 513, 1023, 1024, 1025, 4096, 65_536, 65_537]) {
        let a = values(n as u64 + 1, n);
        let b = values(n as u64 + 7, n);
        let products: Vec<f64> = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .collect();
        assert_eq!(
            dot_scalar(&a, &b).to_bits(),
            levelwise(products).to_bits(),
            "n {n}"
        );
        let diffs: Vec<f64> = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
            .collect();
        assert_eq!(
            l1_scalar(&a, &b).to_bits(),
            levelwise(diffs).to_bits(),
            "n {n}"
        );
    }
}

/// Every kernel set this CPU runs returns the reference's bits, for every
/// metric and length.
#[test]
fn every_kernel_set_returns_the_reference_bits() {
    let sets = Kernels::available();
    for n in (0..=300).chain([1023, 1024, 1025, 3072, 65_536]) {
        let a = values(n as u64 + 3, n);
        let b = values(n as u64 + 5, n);
        let reference = outputs(Kernels::Scalar, &a, &b);
        for &set in &sets {
            assert_eq!(outputs(set, &a, &b), reference, "{set:?}, n {n}");
        }
    }
}

/// The kernel set dispatch picks is one of the sets the CPU runs, and the
/// widest of them.
#[test]
fn dispatch_picks_the_widest_available_set() {
    assert_eq!(Some(&kernels()), Kernels::available().last());
}

/// A nonzero vector far below 1 keeps a nonzero squared distance: its f32
/// squares would underflow to zero.
#[test]
fn a_tiny_nonzero_difference_has_a_nonzero_squared_l2() {
    let a = vec![1e-25f32; 64];
    let b = vec![0.0f32; 64];
    let got = squared_l2(&a, &b).unwrap();
    let want = f64::from(1e-25f32) * f64::from(1e-25f32) * 64.0;
    assert!(got > 0.0);
    assert!((got - want).abs() <= want * 1e-15, "{got} vs {want}");
    assert!(l2(&a, &b).unwrap() > 0.0);
}

/// Opposite components near the f32 limit cancel: the f32 kernel's product
/// overflows to inf - inf = NaN, the f64 evaluation keeps the exact zero.
#[test]
fn opposite_large_products_cancel_exactly() {
    let a = [3e38f32, 3e38];
    let b = [3e38f32, -3e38];
    assert_eq!(dot(&a, &b).unwrap(), 0.0);
    assert_eq!(cosine_similarity(&a, &b).unwrap(), 0.0);
}

/// A dot product whose terms overflow f32 is finite and exact in f64.
#[test]
fn a_dot_product_overflowing_f32_is_finite() {
    let a = [1e20f32; 4];
    let want = f64::from(1e20f32) * f64::from(1e20f32) * 4.0;
    assert_eq!(dot(&a, &a).unwrap(), want);
    assert_eq!(cosine_similarity(&a, &a).unwrap(), 1.0);
}

/// The evaluation rounds its sums: cancellation can lose a term the exact sum
/// keeps. The canonical result is fixed by the tree, not the exact value.
#[test]
fn cancellation_follows_the_tree_not_the_exact_sum() {
    let big = 2f32.powi(60);
    let a = [big, 1.0, -big];
    let b = [1.0f32, 1.0, 1.0];
    let terms = [f64::from(big), 1.0, -f64::from(big)];
    // ((2^60 + 1) + -2^60): the 1 is lost in the first rounding.
    assert_eq!(dot(&a, &b).unwrap(), 0.0);
    assert_eq!(exact_sum(&terms), 1.0);
}

/// Signed zeros come out as +0, so equal scores compare and print equal.
#[test]
fn zero_scores_are_positive_zero() {
    let neg = [-0.0f32];
    let pos = [0.0f32];
    assert_eq!(dot(&neg, &pos).unwrap().to_bits(), 0.0f64.to_bits());
    assert_eq!(l1(&neg, &neg).unwrap().to_bits(), 0.0f64.to_bits());
    assert_eq!(squared_l2(&neg, &pos).unwrap().to_bits(), 0.0f64.to_bits());
    // Norms with exact square roots, so the similarity is exactly 1.
    let a = [3.0f32, -4.0];
    assert_eq!(cosine_distance(&a, &a).unwrap().to_bits(), 0.0f64.to_bits());
}

/// Subnormal f32 inputs are kept, not flushed: their squares are normal f64.
#[test]
fn subnormal_components_are_not_flushed() {
    let tiny = f32::from_bits(1); // smallest positive subnormal
    let a = [tiny, 0.0];
    let ss = sum_of_squares(&a);
    assert_eq!(ss, f64::from(tiny) * f64::from(tiny));
    assert!(ss > 0.0);
    assert_eq!(cosine_similarity(&a, &a).unwrap(), 1.0);
}

/// Cosine of a zero vector is a domain error on either side; a tiny nonzero
/// vector still has a direction.
#[test]
fn cosine_of_a_zero_vector_is_a_domain_error() {
    let zero = [0.0f32, -0.0, 0.0];
    let other = [1.0f32, 2.0, 3.0];
    for (a, b) in [(&zero, &other), (&other, &zero), (&zero, &zero)] {
        assert_eq!(
            cosine_similarity(a, b),
            Err(VectorDomainError::ZeroVectorCosine)
        );
        assert_eq!(
            cosine_distance(a, b),
            Err(VectorDomainError::ZeroVectorCosine)
        );
    }
    assert!(is_zero(&zero));
    let tiny = [1e-40f32, 0.0, 0.0];
    assert!(!is_zero(&tiny));
    assert_eq!(cosine_similarity(&tiny, &tiny).unwrap(), 1.0);
    // L1, L2 and Dot of a zero vector are ordinary values.
    assert_eq!(dot(&zero, &other).unwrap(), 0.0);
    assert_eq!(squared_l2(&zero, &zero).unwrap(), 0.0);
    assert_eq!(l1(&zero, &other).unwrap(), 6.0);
}

/// NaN and infinite components are refused, naming the first one.
#[test]
fn non_finite_components_are_refused() {
    assert_eq!(
        validate(&[1.0, f32::NAN, f32::INFINITY]),
        Err(VectorDomainError::NonFinite { index: 1 })
    );
    assert_eq!(
        validate(&[f32::NEG_INFINITY]),
        Err(VectorDomainError::NonFinite { index: 0 })
    );
    assert_eq!(validate(&[f32::MAX, f32::MIN, -0.0]), Ok(()));
}

/// Empty vectors and vectors of different dimensions are refused.
#[test]
fn empty_and_mismatched_vectors_are_refused() {
    assert_eq!(validate(&[]), Err(VectorDomainError::Empty));
    let mismatch = Err(VectorDomainError::DimensionMismatch { left: 2, right: 3 });
    let (a, b) = ([1.0f32, 2.0], [1.0f32, 2.0, 3.0]);
    assert_eq!(dot(&a, &b), mismatch);
    assert_eq!(squared_l2(&a, &b), mismatch);
    assert_eq!(l2(&a, &b), mismatch);
    assert_eq!(l1(&a, &b), mismatch);
    assert_eq!(cosine_similarity(&a, &b), mismatch);
    assert_eq!(cosine_distance(&a, &b), mismatch);
    assert_eq!(cosine_similarity_with_sum_of_squares(&a, 5.0, &b), mismatch);
}

/// Cosine is clamped to [-1, 1] when rounding lands just outside, and the
/// cached sum of squares gives the same bits as computing it.
#[test]
fn cosine_stays_in_range_and_matches_the_cached_norm_path() {
    for n in 1..=200 {
        let a = values(n as u64 + 11, n);
        if is_zero(&a) {
            continue;
        }
        let same = cosine_similarity(&a, &a).unwrap();
        assert!((-1.0..=1.0).contains(&same), "n {n}: {same}");
        let b = values(n as u64 + 13, n);
        if is_zero(&b) {
            continue;
        }
        let direct = cosine_similarity(&a, &b).unwrap();
        let cached = cosine_similarity_with_sum_of_squares(&a, sum_of_squares(&a), &b).unwrap();
        assert_eq!(direct.to_bits(), cached.to_bits(), "n {n}");
        assert!((-1.0..=1.0).contains(&direct));
    }
    assert_eq!(cosine_from_parts(3.0, 1.0, 1.0), Ok(1.0));
    assert_eq!(cosine_from_parts(-3.0, 1.0, 1.0), Ok(-1.0));
}

/// Scores of nearly tied inputs keep their order: two vectors one f32 ulp
/// apart give distinct, correctly ordered squared distances.
#[test]
fn nearly_tied_candidates_keep_their_order() {
    let q = [1.0f32; 32];
    let near = [1.0f32; 32];
    let mut far = [1.0f32; 32];
    far[31] = f32::from_bits(1.0f32.to_bits() + 1);
    let a = squared_l2(&q, &near).unwrap();
    let b = squared_l2(&q, &far).unwrap();
    assert!(a < b, "{a} vs {b}");
}

/// The rounded tree stays within the bound of pairwise summation of
/// same-sign terms against the exact oracle.
#[test]
fn error_against_the_exact_sum_is_bounded() {
    for n in [16usize, 100, 1024, 4096, 65_536] {
        let a: Vec<f32> = values(n as u64 + 17, n)
            .into_iter()
            .map(|x| {
                if x.is_finite() && x.abs() < 1e4 {
                    x
                } else {
                    0.5
                }
            })
            .collect();
        let terms: Vec<f64> = a.iter().map(|&x| f64::from(x) * f64::from(x)).collect();
        let exact = exact_sum(&terms);
        let got = sum_of_squares(&a);
        let depth = f64::from(usize::BITS - n.leading_zeros());
        let bound = exact * f64::EPSILON * depth;
        assert!((got - exact).abs() <= bound, "n {n}: {got} vs {exact}");
    }
}

/// A pairwise score of an input with a NaN or an infinity is refused naming
/// the component, never returned as a NaN score; an empty pair is refused
/// rather than scored 0. Guards the lazy check, which scans the inputs only
/// when the reduction came out non-finite.
#[test]
fn pairwise_scores_refuse_non_finite_and_empty_inputs() {
    let nan = [1.0, f32::NAN, 2.0];
    let inf = [1.0, 2.0, f32::INFINITY];
    let ok = [1.0, 1.0, 1.0];
    assert_eq!(
        dot(&nan, &ok),
        Err(VectorDomainError::NonFinite { index: 1 })
    );
    assert_eq!(
        l1(&ok, &inf),
        Err(VectorDomainError::NonFinite { index: 2 })
    );
    assert_eq!(
        squared_l2(&inf, &ok),
        Err(VectorDomainError::NonFinite { index: 2 })
    );
    assert_eq!(
        l2(&nan, &ok),
        Err(VectorDomainError::NonFinite { index: 1 })
    );
    // An infinity against a zero component: the product is NaN, still caught.
    assert_eq!(
        cosine_similarity(&[f32::INFINITY, 0.0], &[0.0, 1.0]),
        Err(VectorDomainError::NonFinite { index: 0 })
    );
    assert_eq!(dot(&[], &[]), Err(VectorDomainError::Empty));
    let query = ExactQuery::new(&ok, VectorMetric::L2).unwrap();
    assert_eq!(
        query.distance(&nan),
        Err(VectorDomainError::NonFinite { index: 1 })
    );
    let query = ExactQuery::new(&ok, VectorMetric::Cosine).unwrap();
    assert_eq!(
        query.distance(&inf),
        Err(VectorDomainError::NonFinite { index: 2 })
    );
}

/// The public sum is the canonical tree over the terms' order.
#[test]
fn sum_is_the_canonical_tree() {
    for n in [1usize, 2, 3, 15, 16, 17, 37, 100] {
        let terms: Vec<f64> = values(n as u64, n)
            .into_iter()
            .map(|x| if x.is_finite() { f64::from(x) } else { 1.0 })
            .collect();
        assert_eq!(
            sum(&terms).to_bits(),
            (levelwise(terms.clone()) + 0.0).to_bits()
        );
    }
}

/// MaxSim is the canonical sum, over query rows, of each row's largest exact
/// dot product with a document row: bit-for-bit, not within a tolerance.
#[test]
fn maxsim_sums_each_query_rows_best_exact_dot() {
    let q: Vec<Vec<f32>> = (0..5)
        .map(|i| (0..8).map(|j| ((i + j) as f32) * 0.1 - 0.3).collect())
        .collect();
    let d: Vec<Vec<f32>> = (0..9)
        .map(|i| (0..8).map(|j| ((i * 2 + j) as f32) * 0.05 - 0.2).collect())
        .collect();
    let best: Vec<f64> = q
        .iter()
        .map(|q| {
            d.iter()
                .map(|d| dot(q, d).unwrap())
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    assert_eq!(maxsim(&d, &q).unwrap().to_bits(), levelwise(best).to_bits());
    // One row each: the dot product itself.
    let (a, b) = (vec![vec![1.0, 2.0, 3.0]], vec![vec![4.0, 5.0, 6.0]]);
    assert_eq!(maxsim(&b, &a), Ok(32.0));
}

/// Products past the f32 range stay exact in binary64: a matrix of 1e20
/// components scores 2e40 per row, where an f32 kernel overflows to infinity.
#[test]
fn maxsim_holds_products_beyond_f32() {
    let rows = vec![vec![1e20f32, 1e20]];
    let expected = 2.0 * f64::from(1e20f32) * f64::from(1e20f32);
    assert_eq!(maxsim(&rows, &rows), Ok(expected));
}

/// Empty matrices or rows, rows of different widths and non-finite components
/// are domain errors, not a score of 0.
#[test]
fn maxsim_refuses_what_it_cannot_score() {
    let row = vec![vec![1.0f32, 0.0]];
    assert_eq!(maxsim(&[], &row), Err(VectorDomainError::Empty));
    assert_eq!(maxsim(&row, &[]), Err(VectorDomainError::Empty));
    assert_eq!(maxsim(&[vec![]], &[vec![]]), Err(VectorDomainError::Empty));
    assert_eq!(
        maxsim(&[vec![1.0, 0.0, 0.0]], &row),
        Err(VectorDomainError::DimensionMismatch { left: 3, right: 2 })
    );
    assert_eq!(
        maxsim(&row, &[vec![1.0, 0.0], vec![1.0]]),
        Err(VectorDomainError::DimensionMismatch { left: 1, right: 2 })
    );
    assert_eq!(
        maxsim(&[vec![1.0, f32::NAN]], &row),
        Err(VectorDomainError::NonFinite { index: 1 })
    );
}

/// A distance from a cached sum of squares has the bits of the distance that
/// reduces it, under every metric and kernel width: the cache is the same
/// canonical value, not an approximation of it.
#[test]
fn a_cached_sum_of_squares_gives_the_same_distance() {
    for n in [1usize, 15, 16, 17, 100, 769] {
        let q: Vec<f32> = values(n as u64 * 3 + 1, n)
            .into_iter()
            .map(|x| if x.is_finite() { x } else { 0.25 })
            .collect();
        let c: Vec<f32> = values(n as u64 * 5 + 2, n)
            .into_iter()
            .map(|x| if x.is_finite() { x } else { -0.75 })
            .collect();
        for metric in [
            VectorMetric::Cosine,
            VectorMetric::L2,
            VectorMetric::DotProduct,
            VectorMetric::L1,
        ] {
            let Ok(query) = ExactQuery::new(&q, metric) else {
                continue;
            };
            let direct = query.distance(&c);
            let cached = query.distance_with_sum_of_squares(&c, sum_of_squares(&c));
            assert_eq!(
                direct.map(f64::to_bits),
                cached.map(f64::to_bits),
                "{metric:?} n {n}"
            );
        }
    }
}
