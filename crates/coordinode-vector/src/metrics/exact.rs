//! Exact vector evaluation: binary64 scores with one canonical reduction.
//!
//! Retained values stay f32. Every component is widened to f64 before any
//! subtraction, multiplication or absolute difference, and every intermediate,
//! norm and score stays f64. The terms of a metric (products, squared
//! differences, absolute differences) are summed by one fixed tree over
//! component order, so the scalar, AVX2 and NEON kernels return the same bits,
//! whatever the SIMD width or the caller's batching:
//!
//! - adjacent terms are added pairwise, level by level, and an odd last term
//!   is carried to the next level unchanged;
//! - equivalently, a range of `n > 1` terms is the sum of its leading
//!   `2^k` terms (the largest power of two below `n`) and the rest, each
//!   reduced by the same rule.
//!
//! Additions and multiplications are separately rounded (IEEE 754 binary64,
//! round to nearest, ties to even): no fused multiply-add, no reassociation,
//! no flushing of subnormals. The products, squares and differences of f32
//! values are exact in f64, so only the sums round: the result is the rounded
//! evaluation of this tree, not the correctly rounded mathematical sum.
//!
//! Inputs are expected to have passed [`validate`] (non-empty, finite); the
//! pairwise entry points check that dimensions agree.

use coordinode_core::graph::types::VectorMetric;

/// A vector a numerical evaluation cannot accept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VectorDomainError {
    /// The two vectors have different dimensions.
    #[error("vector dimensions differ: {left} and {right}")]
    DimensionMismatch {
        /// Dimension of the first vector.
        left: usize,
        /// Dimension of the second vector.
        right: usize,
    },
    /// The vector has no components.
    #[error("a vector has no components")]
    Empty,
    /// A component is NaN or infinite.
    #[error("vector component {index} is not finite")]
    NonFinite {
        /// Position of the first non-finite component.
        index: usize,
    },
    /// Cosine was asked of a vector whose components are all zero: it has no
    /// direction.
    #[error("cosine is undefined for a zero vector")]
    ZeroVectorCosine,
}

/// Check that `v` can take part in a numerical evaluation: at least one
/// component, every component finite.
///
/// # Errors
///
/// [`VectorDomainError::Empty`] or [`VectorDomainError::NonFinite`] naming the
/// first offending component.
pub fn validate(v: &[f32]) -> Result<(), VectorDomainError> {
    if v.is_empty() {
        return Err(VectorDomainError::Empty);
    }
    match v.iter().position(|x| !x.is_finite()) {
        Some(index) => Err(VectorDomainError::NonFinite { index }),
        None => Ok(()),
    }
}

/// Whether every component of `v` is zero, judged from the components, not a
/// threshold: a tiny nonzero vector has a direction.
#[must_use]
pub fn is_zero(v: &[f32]) -> bool {
    v.iter().all(|&x| x == 0.0)
}

fn same_dims(a: &[f32], b: &[f32]) -> Result<(), VectorDomainError> {
    if a.len() != b.len() {
        return Err(VectorDomainError::DimensionMismatch {
            left: a.len(),
            right: b.len(),
        });
    }
    if a.is_empty() {
        return Err(VectorDomainError::Empty);
    }
    Ok(())
}

/// `reduced` when it is finite, else the non-finite component of `a` or `b`
/// it came from. Squares and products of finite f32 values are at most about
/// 4.6e77 in binary64, so no reduction of a slice's worth of them overflows:
/// a non-finite result is the only trace a non-finite input leaves, and the
/// inputs are scanned only then.
#[inline(always)]
fn finite(reduced: f64, a: &[f32], b: &[f32]) -> Result<f64, VectorDomainError> {
    if reduced.is_finite() {
        Ok(reduced)
    } else {
        Err(non_finite_input(a, b))
    }
}

#[cold]
#[inline(never)]
fn non_finite_input(a: &[f32], b: &[f32]) -> VectorDomainError {
    match validate(a).and_then(|()| validate(b)) {
        Err(e) => e,
        // Unreachable by the bound above; reported rather than trusted.
        Ok(()) => VectorDomainError::NonFinite { index: 0 },
    }
}

/// `+0.0` for either zero, `x` otherwise: equal scores compare equal and print
/// the same way, whatever sign the arithmetic left on a zero.
#[inline(always)]
fn canonical(x: f64) -> f64 {
    // -0 + +0 is +0 under round to nearest; any other value is unchanged.
    x + 0.0
}

/// Dot product.
///
/// # Errors
///
/// [`VectorDomainError::DimensionMismatch`], [`VectorDomainError::Empty`] or
/// [`VectorDomainError::NonFinite`]. The pairwise functions below fail the
/// same way.
pub fn dot(a: &[f32], b: &[f32]) -> Result<f64, VectorDomainError> {
    same_dims(a, b)?;
    finite(canonical(kernels().dot(a, b)), a, b)
}

/// Squared Euclidean distance, from the component differences (never from
/// norms and the dot product).
///
/// # Errors
///
/// As [`dot`].
pub fn squared_l2(a: &[f32], b: &[f32]) -> Result<f64, VectorDomainError> {
    same_dims(a, b)?;
    finite(canonical(kernels().squared_l2(a, b)), a, b)
}

/// Euclidean distance: the correctly rounded square root of [`squared_l2`].
///
/// # Errors
///
/// As [`dot`].
pub fn l2(a: &[f32], b: &[f32]) -> Result<f64, VectorDomainError> {
    Ok(squared_l2(a, b)?.sqrt())
}

/// Manhattan distance.
///
/// # Errors
///
/// As [`dot`].
pub fn l1(a: &[f32], b: &[f32]) -> Result<f64, VectorDomainError> {
    same_dims(a, b)?;
    finite(canonical(kernels().l1(a, b)), a, b)
}

/// Sum of the squared components of `a`: what a cached norm holds. The norm
/// is its square root.
#[must_use]
pub fn sum_of_squares(a: &[f32]) -> f64 {
    kernels().sum_of_squares(a)
}

/// Cosine similarity from its parts: the dot product and both sums of
/// squares, as [`dot`] and [`sum_of_squares`] return them. Divides the dot
/// product by the product of the two square roots, in that order, and clamps
/// the finite quotient to `[-1, 1]`.
///
/// # Errors
///
/// [`VectorDomainError::ZeroVectorCosine`] when either sum of squares is zero.
/// The square of a nonzero finite f32 component is a normal positive f64, so a
/// zero sum means every component is zero.
pub fn cosine_from_parts(
    dot: f64,
    sum_of_squares_a: f64,
    sum_of_squares_b: f64,
) -> Result<f64, VectorDomainError> {
    if sum_of_squares_a == 0.0 || sum_of_squares_b == 0.0 {
        return Err(VectorDomainError::ZeroVectorCosine);
    }
    let similarity = dot / (sum_of_squares_a.sqrt() * sum_of_squares_b.sqrt());
    Ok(canonical(similarity.clamp(-1.0, 1.0)))
}

/// Cosine similarity.
///
/// # Errors
///
/// As [`dot`], or [`VectorDomainError::ZeroVectorCosine`] when either vector
/// is all zeros.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f64, VectorDomainError> {
    same_dims(a, b)?;
    let (dot, ss_a, ss_b) = kernels().cosine_parts(a, b);
    // A non-finite component of either vector leaves the dot product
    // non-finite (an infinity times anything is infinite or NaN).
    cosine_from_parts(finite(dot, a, b)?, ss_a, ss_b)
}

/// Cosine similarity of `a` and `b` where `a`'s sum of squares is known, as
/// [`sum_of_squares`] returned it: a query compared against many candidates
/// computes it once.
///
/// # Errors
///
/// As [`cosine_similarity`].
pub fn cosine_similarity_with_sum_of_squares(
    a: &[f32],
    sum_of_squares_a: f64,
    b: &[f32],
) -> Result<f64, VectorDomainError> {
    same_dims(a, b)?;
    let (dot, _, ss_b) = kernels().cosine_parts(a, b);
    cosine_from_parts(finite(dot, a, b)?, sum_of_squares_a, ss_b)
}

/// Cosine distance: binary64 `1 - ` the canonical similarity.
///
/// # Errors
///
/// As [`cosine_similarity`].
pub fn cosine_distance(a: &[f32], b: &[f32]) -> Result<f64, VectorDomainError> {
    Ok(canonical(1.0 - cosine_similarity(a, b)?))
}

/// A query vector checked once for a metric, scoring candidates by that
/// metric's ranking distance: lower ranks first.
///
/// | metric | ranking distance |
/// |---|---|
/// | Cosine | `1 - ` [`cosine_similarity`] |
/// | L2 | [`l2`] |
/// | DotProduct | `-` [`dot`] |
/// | L1 | [`l1`] |
#[derive(Debug, Clone, Copy)]
pub struct ExactQuery<'a> {
    vector: &'a [f32],
    metric: VectorMetric,
    /// The query's sum of squares, for cosine; unused otherwise.
    sum_of_squares: f64,
}

impl<'a> ExactQuery<'a> {
    /// Check `vector` as a query for `metric`.
    ///
    /// # Errors
    ///
    /// [`VectorDomainError::Empty`] or [`VectorDomainError::NonFinite`], and
    /// [`VectorDomainError::ZeroVectorCosine`] for a zero cosine query.
    pub fn new(vector: &'a [f32], metric: VectorMetric) -> Result<Self, VectorDomainError> {
        validate(vector)?;
        let sum_of_squares = match metric {
            VectorMetric::Cosine => {
                let ss = sum_of_squares(vector);
                if ss == 0.0 {
                    return Err(VectorDomainError::ZeroVectorCosine);
                }
                ss
            }
            VectorMetric::L2 | VectorMetric::DotProduct | VectorMetric::L1 => 0.0,
        };
        Ok(Self {
            vector,
            metric,
            sum_of_squares,
        })
    }

    /// The query vector.
    #[must_use]
    pub fn vector(&self) -> &'a [f32] {
        self.vector
    }

    /// The metric the query ranks by.
    #[must_use]
    pub fn metric(&self) -> VectorMetric {
        self.metric
    }

    /// The ranking distance of `candidate`.
    ///
    /// # Errors
    ///
    /// [`VectorDomainError::DimensionMismatch`] or
    /// [`VectorDomainError::NonFinite`] for the candidate, and
    /// [`VectorDomainError::ZeroVectorCosine`] for a zero cosine candidate.
    pub fn distance(&self, candidate: &[f32]) -> Result<f64, VectorDomainError> {
        same_dims(self.vector, candidate)?;
        let (q, k) = (self.vector, kernels());
        let d = match self.metric {
            VectorMetric::Cosine => {
                let (dot, _, ss) = k.cosine_parts(q, candidate);
                1.0 - cosine_from_parts(finite(dot, q, candidate)?, self.sum_of_squares, ss)?
            }
            VectorMetric::L2 => finite(k.squared_l2(q, candidate), q, candidate)?.sqrt(),
            VectorMetric::DotProduct => -finite(k.dot(q, candidate), q, candidate)?,
            VectorMetric::L1 => finite(k.l1(q, candidate), q, candidate)?,
        };
        Ok(canonical(d))
    }
}

/// Check `vector` for admission to an index of `metric`: finite, nonempty,
/// and nonzero for cosine.
///
/// # Errors
///
/// As [`ExactQuery::new`].
pub fn admit(vector: &[f32], metric: VectorMetric) -> Result<(), VectorDomainError> {
    validate(vector)?;
    if matches!(metric, VectorMetric::Cosine) && is_zero(vector) {
        return Err(VectorDomainError::ZeroVectorCosine);
    }
    Ok(())
}

/// The order of exact ranking distances, nearest first, equal distances by
/// ascending identity. Exact scores of admitted vectors are finite, so this is
/// the IEEE total order on them.
#[inline]
#[must_use]
pub fn ranked_order(a: f64, a_id: u64, b: f64, b_id: u64) -> core::cmp::Ordering {
    a.total_cmp(&b).then(a_id.cmp(&b_id))
}

/// The canonical sum of `terms`: the same balanced tree over their order that
/// reduces a metric's components.
#[must_use]
pub fn sum(terms: &[f64]) -> f64 {
    canonical(reduce_scalar(terms.len(), |i| terms[i]))
}

/// Late-interaction MaxSim of `query` against `doc`: for each query row the
/// largest exact dot product with any document row, summed canonically in
/// query-row order.
///
/// # Errors
///
/// [`VectorDomainError::Empty`] for an empty matrix or row,
/// [`VectorDomainError::DimensionMismatch`] for a row of another width than
/// the first query row, [`VectorDomainError::NonFinite`] for a non-finite
/// component.
pub fn maxsim(doc: &[Vec<f32>], query: &[Vec<f32>]) -> Result<f64, VectorDomainError> {
    let width = query.first().ok_or(VectorDomainError::Empty)?.len();
    if doc.is_empty() {
        return Err(VectorDomainError::Empty);
    }
    for row in doc.iter().chain(query) {
        validate(row)?;
        if row.len() != width {
            return Err(VectorDomainError::DimensionMismatch {
                left: row.len(),
                right: width,
            });
        }
    }
    let kernels = kernels();
    // Every row is finite and nonempty, so each dot product is finite and the
    // fold's starting infinity is always replaced.
    let best: Vec<f64> = query
        .iter()
        .map(|q| {
            doc.iter()
                .map(|d| canonical(kernels.dot(q, d)))
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    Ok(sum(&best))
}

// --- Canonical tree ---

/// Terms one SIMD or scalar block reduces before handing a single value to
/// the [`Tree`]: a complete subtree, so blocks are merged as leaves of size
/// `BLOCK`.
pub(crate) const BLOCK: usize = 16;

/// The canonical reduction of 16 terms.
#[inline(always)]
fn tree16(t: &[f64; BLOCK]) -> f64 {
    let l1 = [
        t[0] + t[1],
        t[2] + t[3],
        t[4] + t[5],
        t[6] + t[7],
        t[8] + t[9],
        t[10] + t[11],
        t[12] + t[13],
        t[14] + t[15],
    ];
    let l2 = [l1[0] + l1[1], l1[2] + l1[3], l1[4] + l1[5], l1[6] + l1[7]];
    let l3 = [l2[0] + l2[1], l2[2] + l2[3]];
    l3[0] + l3[1]
}

/// The canonical reduction of fewer than `BLOCK` terms (the tail), or of any
/// slice: the leading power of two plus the rest.
fn tree_small(t: &[f64]) -> f64 {
    match t.len() {
        0 => 0.0,
        1 => t[0],
        n => {
            // Largest power of two strictly below n.
            let split = 1usize << (usize::BITS - 1 - (n - 1).leading_zeros());
            tree_small(&t[..split]) + tree_small(&t[split..])
        }
    }
}

/// Merges the block sums of a reduction into the canonical tree as they
/// arrive: complete subtrees of equal size are combined at once (a binary
/// counter over the blocks), and the left-over subtrees, largest on the left,
/// are folded from the right when the input ends.
pub(crate) struct Tree {
    /// Pending complete subtrees, largest first; 64 levels cover any count.
    stack: [f64; 64],
    len: usize,
    blocks: u64,
}

impl Tree {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            stack: [0.0; 64],
            len: 0,
            blocks: 0,
        }
    }

    /// Add the sum of the next complete block.
    #[inline(always)]
    pub(crate) fn push(&mut self, block_sum: f64) {
        let mut value = block_sum;
        // Each trailing one bit of the count is a subtree of the same size
        // as the one being completed: merge it.
        let mut count = self.blocks;
        while count & 1 == 1 {
            self.len -= 1;
            // IEEE addition is commutative: the operand order changes no bit.
            value += self.stack[self.len];
            count >>= 1;
        }
        self.stack[self.len] = value;
        self.len += 1;
        self.blocks += 1;
    }

    /// The canonical sum of every pushed block and then `tail`, the reduced
    /// remaining terms (`None` when the input was a whole number of blocks).
    #[inline(always)]
    pub(crate) fn finish(self, tail: Option<f64>) -> f64 {
        let mut stack = self.stack[..self.len].iter().rev();
        let mut acc = match tail {
            Some(t) => t,
            None => match stack.next() {
                Some(&last) => last,
                None => return 0.0,
            },
        };
        for &left in stack {
            acc += left;
        }
        acc
    }
}

// --- Scalar kernels: the reference every other kernel matches bit for bit ---

/// Canonical reduction of `n` terms produced by `term(i)`.
#[inline(always)]
fn reduce_scalar(n: usize, term: impl Fn(usize) -> f64) -> f64 {
    let mut tree = Tree::new();
    let blocks = n / BLOCK;
    for b in 0..blocks {
        let base = b * BLOCK;
        let t: [f64; BLOCK] = core::array::from_fn(|i| term(base + i));
        tree.push(tree16(&t));
    }
    let rest = n - blocks * BLOCK;
    let tail = (rest > 0).then(|| {
        let mut t = [0.0f64; BLOCK];
        for (i, slot) in t.iter_mut().enumerate().take(rest) {
            *slot = term(blocks * BLOCK + i);
        }
        tree_small(&t[..rest])
    });
    tree.finish(tail)
}

pub(crate) fn dot_scalar(a: &[f32], b: &[f32]) -> f64 {
    reduce_scalar(a.len(), |i| f64::from(a[i]) * f64::from(b[i]))
}

pub(crate) fn squared_l2_scalar(a: &[f32], b: &[f32]) -> f64 {
    reduce_scalar(a.len(), |i| {
        let d = f64::from(a[i]) - f64::from(b[i]);
        d * d
    })
}

pub(crate) fn l1_scalar(a: &[f32], b: &[f32]) -> f64 {
    reduce_scalar(a.len(), |i| (f64::from(a[i]) - f64::from(b[i])).abs())
}

pub(crate) fn sum_of_squares_scalar(a: &[f32]) -> f64 {
    reduce_scalar(a.len(), |i| {
        let x = f64::from(a[i]);
        x * x
    })
}

pub(crate) fn cosine_parts_scalar(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    (
        dot_scalar(a, b),
        sum_of_squares_scalar(a),
        sum_of_squares_scalar(b),
    )
}

/// The tail of a SIMD kernel: the terms after the last whole block, reduced
/// by the scalar rule.
#[inline(always)]
fn tail(n: usize, term: impl Fn(usize) -> f64) -> Option<f64> {
    let start = n / BLOCK * BLOCK;
    (start < n).then(|| {
        let mut t = [0.0f64; BLOCK];
        for (i, slot) in t.iter_mut().enumerate().take(n - start) {
            *slot = term(start + i);
        }
        tree_small(&t[..n - start])
    })
}

// --- AVX2 kernels (x86_64, runtime-detected; no FMA by contract) ---

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{BLOCK, Tree, tail};
    use std::arch::x86_64::*;

    /// The canonical reduction of the 16 terms in `p0..p3` (terms 0-3, 4-7,
    /// 8-11, 12-15).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn block(p0: __m256d, p1: __m256d, p2: __m256d, p3: __m256d) -> f64 {
        // Adjacent pairs: [t0+t1, t4+t5, t2+t3, t6+t7] and the same for 8-15.
        let h01 = _mm256_hadd_pd(p0, p1);
        let h23 = _mm256_hadd_pd(p2, p3);
        // In term order: [t01, t23, t45, t67] and [t89, t1011, t1213, t1415].
        let o01 = _mm256_permute4x64_pd::<0b11_01_10_00>(h01);
        let o23 = _mm256_permute4x64_pd::<0b11_01_10_00>(h23);
        // Pairs of pairs: [s0-3, s8-11, s4-7, s12-15].
        let q = _mm256_hadd_pd(o01, o23);
        // [s0-7, s8-15].
        let lo = _mm256_castpd256_pd128(q);
        let hi = _mm256_extractf128_pd::<1>(q);
        let r = _mm_add_pd(lo, hi);
        _mm_cvtsd_f64(r) + _mm_cvtsd_f64(_mm_unpackhi_pd(r, r))
    }

    /// Four f32 at `p`, widened.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load(p: *const f32) -> __m256d {
        // SAFETY: the caller keeps `p..p+4` inside its slice.
        unsafe { _mm256_cvtps_pd(_mm_loadu_ps(p)) }
    }

    /// # Safety
    ///
    /// AVX2 is available; `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len();
        let mut tree = Tree::new();
        for base in (0..n / BLOCK).map(|i| i * BLOCK) {
            // SAFETY: base + 16 <= n for every whole block.
            unsafe {
                let (pa, pb) = (a.as_ptr().add(base), b.as_ptr().add(base));
                let p = |o: usize| _mm256_mul_pd(load(pa.add(o)), load(pb.add(o)));
                tree.push(block(p(0), p(4), p(8), p(12)));
            }
        }
        tree.finish(tail(n, |i| f64::from(a[i]) * f64::from(b[i])))
    }

    /// # Safety
    ///
    /// AVX2 is available; `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn squared_l2(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len();
        let mut tree = Tree::new();
        for base in (0..n / BLOCK).map(|i| i * BLOCK) {
            // SAFETY: base + 16 <= n for every whole block.
            unsafe {
                let (pa, pb) = (a.as_ptr().add(base), b.as_ptr().add(base));
                let p = |o: usize| {
                    let d = _mm256_sub_pd(load(pa.add(o)), load(pb.add(o)));
                    _mm256_mul_pd(d, d)
                };
                tree.push(block(p(0), p(4), p(8), p(12)));
            }
        }
        tree.finish(tail(n, |i| {
            let d = f64::from(a[i]) - f64::from(b[i]);
            d * d
        }))
    }

    /// # Safety
    ///
    /// AVX2 is available; `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn l1(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len();
        let mut tree = Tree::new();
        let sign = _mm256_set1_pd(-0.0);
        for base in (0..n / BLOCK).map(|i| i * BLOCK) {
            // SAFETY: base + 16 <= n for every whole block.
            unsafe {
                let (pa, pb) = (a.as_ptr().add(base), b.as_ptr().add(base));
                let p = |o: usize| {
                    _mm256_andnot_pd(sign, _mm256_sub_pd(load(pa.add(o)), load(pb.add(o))))
                };
                tree.push(block(p(0), p(4), p(8), p(12)));
            }
        }
        tree.finish(tail(n, |i| (f64::from(a[i]) - f64::from(b[i])).abs()))
    }

    /// # Safety
    ///
    /// AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sum_of_squares(a: &[f32]) -> f64 {
        let n = a.len();
        let mut tree = Tree::new();
        for base in (0..n / BLOCK).map(|i| i * BLOCK) {
            // SAFETY: base + 16 <= n for every whole block.
            unsafe {
                let pa = a.as_ptr().add(base);
                let p = |o: usize| {
                    let x = load(pa.add(o));
                    _mm256_mul_pd(x, x)
                };
                tree.push(block(p(0), p(4), p(8), p(12)));
            }
        }
        tree.finish(tail(n, |i| {
            let x = f64::from(a[i]);
            x * x
        }))
    }

    /// The dot product and both sums of squares in one pass over the inputs.
    ///
    /// # Safety
    ///
    /// AVX2 is available; `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn cosine_parts(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
        let n = a.len();
        let (mut dot, mut ssa, mut ssb) = (Tree::new(), Tree::new(), Tree::new());
        for base in (0..n / BLOCK).map(|i| i * BLOCK) {
            // SAFETY: base + 16 <= n for every whole block.
            unsafe {
                let (pa, pb) = (a.as_ptr().add(base), b.as_ptr().add(base));
                let va = [load(pa), load(pa.add(4)), load(pa.add(8)), load(pa.add(12))];
                let vb = [load(pb), load(pb.add(4)), load(pb.add(8)), load(pb.add(12))];
                let ab = |i: usize| _mm256_mul_pd(va[i], vb[i]);
                let aa = |i: usize| _mm256_mul_pd(va[i], va[i]);
                let bb = |i: usize| _mm256_mul_pd(vb[i], vb[i]);
                dot.push(block(ab(0), ab(1), ab(2), ab(3)));
                ssa.push(block(aa(0), aa(1), aa(2), aa(3)));
                ssb.push(block(bb(0), bb(1), bb(2), bb(3)));
            }
        }
        (
            dot.finish(tail(n, |i| f64::from(a[i]) * f64::from(b[i]))),
            ssa.finish(tail(n, |i| f64::from(a[i]) * f64::from(a[i]))),
            ssb.finish(tail(n, |i| f64::from(b[i]) * f64::from(b[i]))),
        )
    }
}

// --- NEON kernels (aarch64) ---

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{BLOCK, Tree, tail};
    use std::arch::aarch64::*;

    /// The canonical reduction of the 16 terms in `p` (two per register, in
    /// term order).
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn block(p: [float64x2_t; 8]) -> f64 {
        // `vpaddq_f64(x, y)` is `[x0 + x1, y0 + y1]`: adjacent pairs.
        let q0 = vpaddq_f64(p[0], p[1]);
        let q1 = vpaddq_f64(p[2], p[3]);
        let q2 = vpaddq_f64(p[4], p[5]);
        let q3 = vpaddq_f64(p[6], p[7]);
        let r0 = vpaddq_f64(q0, q1);
        let r1 = vpaddq_f64(q2, q3);
        vpaddd_f64(vpaddq_f64(r0, r1))
    }

    /// Sixteen f32 from `p`, widened, in term order.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load16(p: *const f32) -> [float64x2_t; 8] {
        // SAFETY: the caller keeps `p..p+16` inside its slice.
        unsafe {
            let v = [
                vld1q_f32(p),
                vld1q_f32(p.add(4)),
                vld1q_f32(p.add(8)),
                vld1q_f32(p.add(12)),
            ];
            [
                vcvt_f64_f32(vget_low_f32(v[0])),
                vcvt_high_f64_f32(v[0]),
                vcvt_f64_f32(vget_low_f32(v[1])),
                vcvt_high_f64_f32(v[1]),
                vcvt_f64_f32(vget_low_f32(v[2])),
                vcvt_high_f64_f32(v[2]),
                vcvt_f64_f32(vget_low_f32(v[3])),
                vcvt_high_f64_f32(v[3]),
            ]
        }
    }

    /// # Safety
    ///
    /// NEON is available; `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len();
        let mut tree = Tree::new();
        for base in (0..n / BLOCK).map(|i| i * BLOCK) {
            // SAFETY: base + 16 <= n for every whole block.
            unsafe {
                let va = load16(a.as_ptr().add(base));
                let vb = load16(b.as_ptr().add(base));
                tree.push(block(core::array::from_fn(|i| vmulq_f64(va[i], vb[i]))));
            }
        }
        tree.finish(tail(n, |i| f64::from(a[i]) * f64::from(b[i])))
    }

    /// # Safety
    ///
    /// NEON is available; `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn squared_l2(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len();
        let mut tree = Tree::new();
        for base in (0..n / BLOCK).map(|i| i * BLOCK) {
            // SAFETY: base + 16 <= n for every whole block.
            unsafe {
                let va = load16(a.as_ptr().add(base));
                let vb = load16(b.as_ptr().add(base));
                tree.push(block(core::array::from_fn(|i| {
                    let d = vsubq_f64(va[i], vb[i]);
                    vmulq_f64(d, d)
                })));
            }
        }
        tree.finish(tail(n, |i| {
            let d = f64::from(a[i]) - f64::from(b[i]);
            d * d
        }))
    }

    /// # Safety
    ///
    /// NEON is available; `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn l1(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len();
        let mut tree = Tree::new();
        for base in (0..n / BLOCK).map(|i| i * BLOCK) {
            // SAFETY: base + 16 <= n for every whole block.
            unsafe {
                let va = load16(a.as_ptr().add(base));
                let vb = load16(b.as_ptr().add(base));
                tree.push(block(core::array::from_fn(|i| {
                    vabsq_f64(vsubq_f64(va[i], vb[i]))
                })));
            }
        }
        tree.finish(tail(n, |i| (f64::from(a[i]) - f64::from(b[i])).abs()))
    }

    /// # Safety
    ///
    /// NEON is available.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn sum_of_squares(a: &[f32]) -> f64 {
        let n = a.len();
        let mut tree = Tree::new();
        for base in (0..n / BLOCK).map(|i| i * BLOCK) {
            // SAFETY: base + 16 <= n for every whole block.
            unsafe {
                let va = load16(a.as_ptr().add(base));
                tree.push(block(core::array::from_fn(|i| vmulq_f64(va[i], va[i]))));
            }
        }
        tree.finish(tail(n, |i| {
            let x = f64::from(a[i]);
            x * x
        }))
    }

    /// The dot product and both sums of squares in one pass over the inputs.
    ///
    /// # Safety
    ///
    /// NEON is available; `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn cosine_parts(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
        let n = a.len();
        let (mut dot, mut ssa, mut ssb) = (Tree::new(), Tree::new(), Tree::new());
        for base in (0..n / BLOCK).map(|i| i * BLOCK) {
            // SAFETY: base + 16 <= n for every whole block.
            unsafe {
                let va = load16(a.as_ptr().add(base));
                let vb = load16(b.as_ptr().add(base));
                dot.push(block(core::array::from_fn(|i| vmulq_f64(va[i], vb[i]))));
                ssa.push(block(core::array::from_fn(|i| vmulq_f64(va[i], va[i]))));
                ssb.push(block(core::array::from_fn(|i| vmulq_f64(vb[i], vb[i]))));
            }
        }
        (
            dot.finish(tail(n, |i| f64::from(a[i]) * f64::from(b[i]))),
            ssa.finish(tail(n, |i| f64::from(a[i]) * f64::from(a[i]))),
            ssb.finish(tail(n, |i| f64::from(b[i]) * f64::from(b[i]))),
        )
    }
}

// --- Dispatch ---

/// The kernel set this CPU runs, chosen once. Every set returns the scalar
/// reference's bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kernels {
    /// The portable reference.
    Scalar,
    /// x86_64 AVX2.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// aarch64 NEON.
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Kernels {
    /// The kernel sets this CPU can run, the reference first.
    pub(crate) fn available() -> Vec<Self> {
        let mut sets = vec![Self::Scalar];
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            sets.push(Self::Avx2);
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            sets.push(Self::Neon);
        }
        sets
    }

    #[inline(always)]
    pub(crate) fn dot(self, a: &[f32], b: &[f32]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        match self {
            Self::Scalar => dot_scalar(a, b),
            // SAFETY: `Avx2` is chosen only when the CPU has AVX2; the caller
            // checked the dimensions.
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => unsafe { avx2::dot(a, b) },
            // SAFETY: `Neon` is chosen only when the CPU has NEON.
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::dot(a, b) },
        }
    }

    #[inline(always)]
    pub(crate) fn squared_l2(self, a: &[f32], b: &[f32]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        match self {
            Self::Scalar => squared_l2_scalar(a, b),
            // SAFETY: as in `dot`.
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => unsafe { avx2::squared_l2(a, b) },
            // SAFETY: as in `dot`.
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::squared_l2(a, b) },
        }
    }

    #[inline(always)]
    pub(crate) fn l1(self, a: &[f32], b: &[f32]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        match self {
            Self::Scalar => l1_scalar(a, b),
            // SAFETY: as in `dot`.
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => unsafe { avx2::l1(a, b) },
            // SAFETY: as in `dot`.
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::l1(a, b) },
        }
    }

    #[inline(always)]
    pub(crate) fn sum_of_squares(self, a: &[f32]) -> f64 {
        match self {
            Self::Scalar => sum_of_squares_scalar(a),
            // SAFETY: `Avx2` is chosen only when the CPU has AVX2.
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => unsafe { avx2::sum_of_squares(a) },
            // SAFETY: `Neon` is chosen only when the CPU has NEON.
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::sum_of_squares(a) },
        }
    }

    #[inline(always)]
    pub(crate) fn cosine_parts(self, a: &[f32], b: &[f32]) -> (f64, f64, f64) {
        debug_assert_eq!(a.len(), b.len());
        match self {
            Self::Scalar => cosine_parts_scalar(a, b),
            // SAFETY: as in `dot`.
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => unsafe { avx2::cosine_parts(a, b) },
            // SAFETY: as in `dot`.
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::cosine_parts(a, b) },
        }
    }
}

/// The best kernel set this CPU has, detected once: features do not change
/// for the life of the process.
#[inline(always)]
pub(crate) fn kernels() -> Kernels {
    static CHOSEN: std::sync::OnceLock<Kernels> = std::sync::OnceLock::new();
    // no-std: a caller-provided kernel choice
    *CHOSEN.get_or_init(|| {
        Kernels::available()
            .last()
            .copied()
            .unwrap_or(Kernels::Scalar)
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
