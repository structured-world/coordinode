use super::*;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

#[test]
fn calibrate_is_deterministic() {
    let p1 = RaBitQParams::calibrate(128, 42);
    let p2 = RaBitQParams::calibrate(128, 42);
    // FHT-Kac stores the two sign vectors instead of a dense rotation
    // matrix; same seed must reproduce both bit-identical so a recovered
    // segment can re-derive the rotation from the persisted seed alone.
    let v = vec![0.123_f32; 128];
    assert_eq!(p1.encode(&v).code, p2.encode(&v).code);
}

#[test]
fn rotation_is_approximately_orthonormal() {
    // Probe the composite rotation R = FHT · S_b · FHT · S_a indirectly:
    // it preserves L2 norm to within f32 precision (the defining property
    // of an orthonormal transform).
    let p = RaBitQParams::calibrate(64, 12345);
    let d = p.effective_dims() as usize;

    // Norm preservation under R for a handful of test inputs.
    for seed in [1u64, 7, 99, 1_000_000_007] {
        let mut rng = Xorshift64Star::new(seed);
        let mut v: Vec<f32> = (0..d).map(|_| rng.gaussian()).collect();
        let in_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        p.fht_kac_in_place(&mut v);
        let out_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            approx_eq(in_norm, out_norm, 1e-3),
            "norm not preserved: in={in_norm} out={out_norm}",
        );
    }

    // Sanity-shape the function on a one-hot input: rotation should
    // spread mass across all coordinates with comparable magnitudes
    // (the all-coordinates-equal property of Hadamard).
    for (i, k) in [(0usize, 1usize), (0, 13), (5, 17), (30, 63)] {
        let mut e_i = vec![0.0f32; d];
        e_i[i] = 1.0;
        p.fht_kac_in_place(&mut e_i);
        let mut e_k = vec![0.0f32; d];
        e_k[k] = 1.0;
        p.fht_kac_in_place(&mut e_k);
        let dot: f32 = e_i.iter().zip(&e_k).map(|(a, b)| a * b).sum();
        assert!(
            approx_eq(dot, 0.0, 1e-4),
            "columns ({i},{k}) dot = {dot}, expected ~0",
        );
    }
}

#[test]
fn encode_produces_expected_layout() {
    let p = RaBitQParams::calibrate(128, 1);
    let v = vec![0.5f32; 128];
    let c = p.encode(&v);
    assert_eq!(c.code.len(), 128 / 64);
    assert!(c.norm > 0.0);
}

#[test]
fn identical_vectors_have_zero_xor() {
    let p = RaBitQParams::calibrate(128, 99);
    // Use a non-constant vector — a constant vector hits Gaussian symmetry
    // edge cases for the sign bits and isn't representative.
    let v: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
    let c1 = p.encode(&v);
    let c2 = p.encode(&v);
    assert_eq!(c1, c2);
    assert_eq!(popcount::xor_popcount(&c1.code, &c2.code), 0);
}

#[test]
fn similarity_ranks_neighbours_correctly() {
    // Build three vectors: q, near (q + small noise), far (q rotated 180°).
    // The estimator must rank `near` closer to `q` than `far` does.
    let p = RaBitQParams::calibrate(256, 0xABCD);
    let q: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.07).cos()).collect();
    let near: Vec<f32> = q
        .iter()
        .enumerate()
        .map(|(i, x)| x + 0.01 * (i as f32).sin())
        .collect();
    let far: Vec<f32> = q.iter().map(|x| -x).collect();

    let qc = p.encode(&q);
    let nc = p.encode(&near);
    let fc = p.encode(&far);

    let sim_near = p.estimate_inner_product(&qc, &nc);
    let sim_far = p.estimate_inner_product(&qc, &fc);
    assert!(
        sim_near > sim_far,
        "estimator must rank near above far: near={}, far={}",
        sim_near,
        sim_far
    );
}

#[test]
fn code_size_matches_spec() {
    // D=1024 → 128 bytes code + 8 bytes scalars + 4 bytes signed_sum = 140 bytes.
    // signed_sum was added for the asymmetric Eq. 20 kernel (paper §3.3.2).
    let p = RaBitQParams::calibrate(1024, 7);
    let v = vec![1.0f32; 1024];
    let c = p.encode(&v);
    assert_eq!(c.size_bytes(), 140);
}

#[test]
#[should_panic(expected = "dimension mismatch")]
fn encode_rejects_wrong_dim() {
    let p = RaBitQParams::calibrate(128, 0);
    let _ = p.encode(&[0.0f32; 64]);
}

#[test]
fn calibrate_pads_non_64_aligned_dims_internally() {
    // Calibration accepts any dims > 0. The rotation matrix and
    // packed code arrays are sized at the next multiple of 64 above
    // the user-supplied dim; encode pads the input with implicit
    // zeros so popcount stays a whole-u64-words operation.
    let p = RaBitQParams::calibrate(100, 0);
    assert_eq!(p.dims(), 100);
    assert_eq!(p.effective_dims(), 128);
    // Code length = effective_dims / 64 u64 words.
    let v: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.01).collect();
    let c = p.encode(&v);
    assert_eq!(
        c.code.len(),
        2,
        "code uses 2 u64 words at effective_dims=128"
    );
    // Roundtrip identity: encode the same vector twice → same code.
    let c2 = p.encode(&v);
    assert_eq!(c, c2);
}

#[test]
fn padding_preserves_cosine_ranking_dim_100() {
    // Regression: glove-100-angular bench on commit 8fa0f2f returned
    // recall@10 = 0.17 plateau across the full ef sweep for codec=rabitq
    // while codec=none reached 0.94. Hypothesis: the dim=100 → 128
    // zero-padding logic in `encode` breaks cosine rank preservation.
    //
    // Theoretical claim being tested: for orthonormal R ∈ ℝ^{128×128},
    // the first 100 columns M = R[:, 0:100] satisfy MᵀM = I_100, so M
    // is an isometry ℝ^100 → ℝ^128 that preserves cosine similarity.
    // RaBitQ sign bits of M·x should give the standard LSH separation
    // arcsin(ρ)/π between true neighbours and random pairs. Recall@10
    // on a 1000-vector corpus with a planted nearest neighbour should
    // be ≥ 0.6, far above the broken 0.17 plateau.
    //
    // If THIS test fails (recall < 0.5), the bug is in `encode` or in
    // `estimate_cosine_distance` and isolated from HNSW graph build /
    // search. If it passes, the bug is elsewhere in the index path.
    let dims = 100u32;
    let n_base = 1000usize;
    let n_query = 100usize;
    let k = 10usize;
    let seed = 0xC0FFEE_u64;

    let params = RaBitQParams::calibrate(dims, seed);
    assert_eq!(params.effective_dims(), 128);

    // Generate unit-norm Gaussian vectors. For each query, plant one
    // true near-neighbour at cosine ≈ 0.95 (a tight high-similarity
    // pair like glove's top-1) so we have a known correct answer.
    let mut rng = Xorshift64Star::new(0xBADC0DE);

    let make_unit = |rng: &mut Xorshift64Star| -> Vec<f32> {
        let mut v: Vec<f32> = (0..dims as usize).map(|_| rng.gaussian()).collect();
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v.iter_mut() {
            *x /= n.max(f32::EPSILON);
        }
        v
    };

    let mut base: Vec<Vec<f32>> = (0..n_base).map(|_| make_unit(&mut rng)).collect();

    // Plant the first `n_query` base vectors as the "true" nearest
    // neighbours of the queries: query[i] = base[i] + small noise, both
    // re-normalized. Cosine sim ≈ 1 - ‖noise‖²/2 ≈ 0.95 for noise=0.32.
    let queries: Vec<Vec<f32>> = (0..n_query)
        .map(|i| {
            let mut q = base[i].clone();
            for x in q.iter_mut() {
                *x += 0.32 * rng.gaussian() / (dims as f32).sqrt();
            }
            let n: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in q.iter_mut() {
                *x /= n.max(f32::EPSILON);
            }
            q
        })
        .collect();
    // Re-normalize base just in case (defensive — already unit above).
    for v in base.iter_mut() {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v.iter_mut() {
            *x /= n.max(f32::EPSILON);
        }
    }

    // Encode all base vectors and queries.
    let base_codes: Vec<RaBitQCode> = base.iter().map(|v| params.encode(v)).collect();
    let query_codes: Vec<RaBitQCode> = queries.iter().map(|v| params.encode(v)).collect();

    // For each query, find top-k by RaBitQ distance and check whether
    // the planted true-NN (index i) is in the returned top-k.
    let mut hits = 0usize;
    for (qi, qc) in query_codes.iter().enumerate() {
        let mut scored: Vec<(f32, usize)> = base_codes
            .iter()
            .enumerate()
            .map(|(bi, bc)| (params.estimate_cosine_distance(qc, bc), bi))
            .collect();
        scored.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .expect("RaBitQ distance values are finite")
        });
        if scored.iter().take(k).any(|(_, idx)| *idx == qi) {
            hits += 1;
        }
    }
    let recall = hits as f32 / n_query as f32;

    // Loose threshold (0.5) — well above broken plateau 0.17, well
    // below the f32-exact ceiling. Real RaBitQ on this workload
    // should land 0.7-0.9.
    assert!(
        recall >= 0.5,
        "RaBitQ recall@{k} on dim={dims} with padding = {recall:.3} \
             (need ≥0.5; broken 8fa0f2f gave 0.17)"
    );
}

#[test]
fn calibrate_rejects_zero_dims() {
    let r = std::panic::catch_unwind(|| RaBitQParams::calibrate(0, 0));
    assert!(r.is_err(), "dims=0 still panics");
}

#[test]
fn params_serde_round_trip_preserves_rotation_and_codes() {
    // The rotation matrix is durable index state — on segment reload we
    // serialise it, hand it back via `HnswIndex::set_rabitq_params`, and
    // re-encoded vectors must match the codes that were on disk. Verify
    // the full cycle: encode → serialise params → deserialise → re-encode
    // the same vector → bit-identical code.
    let dims = 128u32;
    let seed = 0xFEED_FACEu64;
    let params = RaBitQParams::calibrate(dims, seed);
    let v: Vec<f32> = (0..dims as usize)
        .map(|i| ((i as f32) * 0.13).sin())
        .collect();
    let code_before = params.encode(&v);

    let bytes = rmp_serde::to_vec(&params).expect("serialise params");
    let params2: RaBitQParams = rmp_serde::from_slice(&bytes).expect("deserialise params");
    assert_eq!(params2.dims(), dims);
    assert_eq!(params2.seed(), seed);

    let code_after = params2.encode(&v);
    assert_eq!(
        code_before, code_after,
        "round-tripped params must encode identically"
    );
}

#[test]
fn code_serde_round_trip_is_bit_identical() {
    // Codes live on disk too (in-RAM index ↔ persisted segment). Verify
    // a (de)serialise cycle round-trips the bit-string + scalars exactly.
    let params = RaBitQParams::calibrate(256, 7);
    let v: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05).cos()).collect();
    let code = params.encode(&v);

    let bytes = rmp_serde::to_vec(&code).expect("serialise code");
    let code2: RaBitQCode = rmp_serde::from_slice(&bytes).expect("deserialise code");
    assert_eq!(code, code2);
}

// ── Extended-RaBitQ (R862) ──────────────────────────────────────

#[test]
fn ext_encode_layout_2_3_4_bit() {
    let dims = 128u32;
    let p = RaBitQParams::calibrate(dims, 7);
    let v: Vec<f32> = (0..dims as usize)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();
    for bits in [2u8, 3, 4] {
        let c = p.encode_ext(&v, bits);
        assert_eq!(c.bits, bits);
        assert_eq!(c.dims, dims);
        // packed length must match ceil(dims × bits / 8)
        let expected_packed = (dims as usize * bits as usize).div_ceil(8);
        assert_eq!(
            c.packed.len(),
            expected_packed,
            "bits={bits}: packed length expected {expected_packed}, got {}",
            c.packed.len()
        );
        let max = 1u8 << bits;
        for i in 0..dims as usize {
            assert!(
                c.level(i) < max,
                "bits={bits}: level({i})={} must be < {max}",
                c.level(i)
            );
        }
        assert!(c.norm > 0.0);
    }
}

#[test]
fn ext_code_size_matches_spec() {
    // True bit-packed layout per the SIGMOD 2025 paper:
    //   bits=2 → 256 B packed + 13 B scalars (dims:u32 + bits:u8 + 2×f32) = 269 B
    //   bits=3 → 384 B + 13 = 397 B
    //   bits=4 → 512 B + 13 = 525 B
    // The arch doc quotes the packed body only (256/384/512); the
    // extra 13 B per code is scalar metadata + dims field for safe
    // decoding when dims × bits doesn't fit cleanly in a byte.
    let p = RaBitQParams::calibrate(1024, 7);
    let v = vec![1.0f32; 1024];
    for (bits, packed_bytes) in [(2u8, 256), (3, 384), (4, 512)] {
        let c = p.encode_ext(&v, bits);
        assert_eq!(
            c.packed.len(),
            packed_bytes,
            "bits={bits}: packed body must be {packed_bytes} B"
        );
        assert_eq!(
            c.size_bytes(),
            packed_bytes + 13,
            "bits={bits}: total size mismatch"
        );
    }
}

#[test]
#[should_panic(expected = "bits must be 2, 3, or 4")]
fn ext_rejects_one_bit() {
    let p = RaBitQParams::calibrate(64, 0);
    let _ = p.encode_ext(&[0.5f32; 64], 1);
}

#[test]
#[should_panic(expected = "bits must be 2, 3, or 4")]
fn ext_rejects_five_bit() {
    let p = RaBitQParams::calibrate(64, 0);
    let _ = p.encode_ext(&[0.5f32; 64], 5);
}

#[test]
fn ext_ranks_neighbours_correctly_at_each_bit_width() {
    // Mirrors similarity_ranks_neighbours_correctly for 1-bit:
    // near must score above far at every bit width.
    let p = RaBitQParams::calibrate(256, 0xABCD);
    let q: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.07).cos()).collect();
    let near: Vec<f32> = q
        .iter()
        .enumerate()
        .map(|(i, x)| x + 0.01 * (i as f32).sin())
        .collect();
    let far: Vec<f32> = q.iter().map(|x| -x).collect();

    for bits in [2u8, 3, 4] {
        let qc = p.encode_ext(&q, bits);
        let nc = p.encode_ext(&near, bits);
        let fc = p.encode_ext(&far, bits);

        let sim_near = p.estimate_inner_product_ext(&qc, &nc);
        let sim_far = p.estimate_inner_product_ext(&qc, &fc);
        assert!(
            sim_near > sim_far,
            "bits={bits}: IP estimator must rank near above far: \
                 near={sim_near}, far={sim_far}"
        );

        let dist_near = p.estimate_cosine_distance_ext(&qc, &nc);
        let dist_far = p.estimate_cosine_distance_ext(&qc, &fc);
        assert!(
            dist_near < dist_far,
            "bits={bits}: cosine distance must rank near below far: \
                 near={dist_near}, far={dist_far}"
        );
    }
}

#[test]
fn ext_higher_bits_estimate_closer_to_true_ip() {
    // Pareto property from SIGMOD 2025: estimator error decreases
    // monotonically with bit width. Averaged over a small sample,
    // 4-bit MUST beat 2-bit mean absolute error.
    let dims = 256u32;
    let p = RaBitQParams::calibrate(dims, 0xC0FFEE);

    fn synth(seed: u64, dims: usize) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..dims)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = (s >> 40) as f32 / (1u32 << 24) as f32;
                2.0 * u - 1.0
            })
            .collect()
    }

    let mut err2 = 0.0f64;
    let mut err4 = 0.0f64;
    let mut n = 0usize;
    for seed_q in 0..6u64 {
        let q = synth(seed_q, dims as usize);
        for seed_x in 10..16u64 {
            let x = synth(seed_x, dims as usize);
            let true_ip: f32 = q.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

            let q2 = p.encode_ext(&q, 2);
            let x2 = p.encode_ext(&x, 2);
            let est2 = p.estimate_inner_product_ext(&q2, &x2);

            let q4 = p.encode_ext(&q, 4);
            let x4 = p.encode_ext(&x, 4);
            let est4 = p.estimate_inner_product_ext(&q4, &x4);

            err2 += ((est2 - true_ip) as f64).abs();
            err4 += ((est4 - true_ip) as f64).abs();
            n += 1;
        }
    }
    let avg2 = err2 / n as f64;
    let avg4 = err4 / n as f64;
    assert!(
        avg4 < avg2,
        "4-bit mean abs IP error ({avg4}) must be lower than 2-bit ({avg2})"
    );
}

#[test]
fn from_slice_kernel_matches_reference_path() {
    // The contiguous-store search hot path calls
    // `estimate_cosine_distance_q_from_slice` with raw code words
    // and scalar parameters lifted out of an inline per-node block.
    // For every neighbour, the returned distance must agree with
    // the reference `estimate_cosine_distance_q(&code, &query)` to
    // the last bit — any drift here biases the HNSW heap and would
    // show up as recall regression at the head of the result set.
    let dims = 128u32;
    let p = RaBitQParams::calibrate(dims, 0xC0DE);
    let query_vec: Vec<f32> = (0..dims as usize)
        .map(|i| (i as f32 * 0.07).sin())
        .collect();
    let q = p.encode_query(&query_vec);
    for seed in 0..8u32 {
        let v: Vec<f32> = (0..dims as usize)
            .map(|i| ((i as f32 + seed as f32) * 0.11).cos())
            .collect();
        let code = p.encode(&v);
        let reference = p.estimate_cosine_distance_q(&code, &q);
        let from_slice = p.estimate_cosine_distance_q_from_slice(
            code.code.as_slice(),
            code.norm,
            code.signed_sum,
            code.correction,
            code.radial,
            code.cluster_id,
            &q,
        );
        assert!(
            (reference - from_slice).abs() < 1e-6,
            "from_slice drift: reference={reference}, from_slice={from_slice}",
        );
    }
}

#[test]
fn ext_serde_round_trip_is_value_identical() {
    let dims = 128u32;
    let p = RaBitQParams::calibrate(dims, 0xFACE);
    let v: Vec<f32> = (0..dims as usize)
        .map(|i| (i as f32 * 0.05).cos())
        .collect();
    for bits in [2u8, 3, 4] {
        let c = p.encode_ext(&v, bits);
        let bytes = rmp_serde::to_vec(&c).expect("ser");
        let c2: RaBitQExtCode = rmp_serde::from_slice(&bytes).expect("de");
        assert_eq!(c, c2);
    }
}
