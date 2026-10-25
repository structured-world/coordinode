use super::*;

#[test]
fn calibrate_from_vectors() {
    let v1 = vec![0.0, 1.0, -1.0];
    let v2 = vec![1.0, 0.0, 1.0];
    let v3 = vec![0.5, 0.5, 0.0];

    let params = Sq8Params::calibrate(&[&v1, &v2, &v3]).expect("calibrate");
    assert_eq!(params.dims(), 3);
    assert_eq!(params.mins[0], 0.0);
    assert_eq!(params.maxs[0], 1.0);
    assert_eq!(params.mins[2], -1.0);
    assert_eq!(params.maxs[2], 1.0);
}

#[test]
fn quantize_dequantize_roundtrip() {
    let vectors: Vec<Vec<f32>> = (0..100)
        .map(|i| {
            vec![
                (i as f32) / 100.0,
                ((i * 3) as f32) / 100.0 - 1.0,
                (i as f32).sin(),
            ]
        })
        .collect();

    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let params = Sq8Params::calibrate(&refs).expect("calibrate");

    for v in &vectors {
        let quantized = params.quantize(v);
        let dequantized = params.dequantize(&quantized);

        for (i, (&orig, &restored)) in v.iter().zip(dequantized.iter()).enumerate() {
            let range = params.maxs[i] - params.mins[i];
            let error = (orig - restored).abs() / range;
            assert!(
                error < 0.01,
                "dim {i}: orig={orig}, restored={restored}, error={error}"
            );
        }
    }
}

#[test]
fn quantize_clamps_out_of_range() {
    let params = Sq8Params {
        mins: vec![0.0],
        maxs: vec![1.0],
    };
    assert_eq!(params.quantize(&[-0.5]), vec![0]);
    assert_eq!(params.quantize(&[1.5]), vec![255]);
    assert_eq!(params.quantize(&[0.0]), vec![0]);
    assert_eq!(params.quantize(&[1.0]), vec![255]);
}

#[test]
fn memory_savings_4x() {
    let qv = QuantizedVector {
        data: vec![128u8; 384],
    };
    assert_eq!(qv.size_bytes(), 384);
    assert_eq!(qv.original_size_bytes(), 384 * 4);
    assert_eq!(qv.original_size_bytes() / qv.size_bytes(), 4);
}

#[test]
fn calibrate_empty_returns_none() {
    assert!(Sq8Params::calibrate(&[]).is_none());
}

#[test]
fn calibrate_zero_dims_returns_none() {
    let empty: Vec<f32> = vec![];
    assert!(Sq8Params::calibrate(&[&empty]).is_none());
}

#[test]
fn calibrate_dimension_mismatch_returns_none() {
    let v1 = vec![1.0, 2.0];
    let v2 = vec![1.0, 2.0, 3.0];
    assert!(Sq8Params::calibrate(&[&v1, &v2]).is_none());
}

/// A dimension every vector holds one value in keeps an empty range: it
/// quantizes to code 0 and dequantizes to that value exactly, whatever its
/// magnitude (an absolute widening vanishes beside `1e38`).
#[test]
fn calibrate_constant_dimension() {
    for constant in [5.0f32, 1e38, -1e-30] {
        let v1 = vec![constant, 1.0];
        let v2 = vec![constant, 2.0];
        let params = Sq8Params::calibrate(&[&v1, &v2]).expect("calibrate");
        assert_eq!(params.mins[0], params.maxs[0]);
        let q = params.quantize(&v1);
        assert_eq!(q[0], 0);
        assert_eq!(params.dequantize(&q)[0], constant);
    }
}

/// Quantization is scale-free: the same layout scaled to tiny or huge
/// magnitudes, or spanning opposite values near the f32 limit, restores to
/// the same relative accuracy as at unit scale.
#[test]
fn quantization_holds_at_every_scale() {
    let base: Vec<Vec<f32>> = (0..50)
        .map(|i| vec![(i as f32) / 50.0 - 0.5, ((i * 7) % 50) as f32 / 25.0 - 1.0])
        .collect();
    for scale in [1.0f32, 1e-30, 1e30, 6e37] {
        let vectors: Vec<Vec<f32>> = base
            .iter()
            .map(|v| v.iter().map(|x| x * scale).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let params = Sq8Params::calibrate(&refs).expect("calibrate");
        for v in &vectors {
            let restored = params.dequantize(&params.quantize(v));
            for (i, (&orig, &back)) in v.iter().zip(&restored).enumerate() {
                assert!(back.is_finite(), "scale {scale}: dim {i}");
                let range = f64::from(params.maxs[i]) - f64::from(params.mins[i]);
                let error = (f64::from(orig) - f64::from(back)).abs() / range;
                assert!(error < 0.01, "scale {scale}: dim {i}: {orig} -> {back}");
            }
        }
    }
}

#[test]
fn high_dimensional_768() {
    let dims = 768;
    let vectors: Vec<Vec<f32>> = (0..10)
        .map(|seed| (0..dims).map(|d| ((seed * d) as f32).sin()).collect())
        .collect();

    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let params = Sq8Params::calibrate(&refs).expect("calibrate");
    assert_eq!(params.dims(), 768);

    let quantized = params.quantize(&vectors[0]);
    assert_eq!(quantized.len(), 768);
}

#[test]
fn serialization_roundtrip() {
    let params = Sq8Params {
        mins: vec![-1.0, 0.0],
        maxs: vec![1.0, 2.0],
    };
    let bytes = rmp_serde::to_vec(&params).expect("serialize");
    let restored: Sq8Params = rmp_serde::from_slice(&bytes).expect("deserialize");
    assert_eq!(params, restored);
}

#[test]
fn validate_dims_ok() {
    let v = vec![0.0f32; 384];
    assert!(validate_dimensions(&v, 384).is_ok());
}

#[test]
fn validate_dims_mismatch() {
    let v = vec![0.0f32; 384];
    assert!(validate_dimensions(&v, 512).is_err());
}

#[test]
fn validate_dims_exceeds_max() {
    let v = vec![0.0f32; 1];
    assert!(validate_dimensions(&v, MAX_DIMENSIONS + 1).is_err());
}

#[test]
fn kv_separation_threshold() {
    assert!(!should_kv_separate(128));
    assert!(!should_kv_separate(256));
    assert!(should_kv_separate(257));
    assert!(should_kv_separate(768));
}
