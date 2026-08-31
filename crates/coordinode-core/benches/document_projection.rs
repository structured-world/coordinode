//! Benchmark: document projection optimization.
//!
//! Compares full deserialization + extract_at_path (clone-based)
//! vs extract_at_path_bytes (byte-level skip, only target deserialized).
//!

#![allow(clippy::expect_used)]
//! Target: 10KB document, 1 path → extract_at_path_bytes <50% of full deser.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn build_10kb_document() -> rmpv::Value {
    // Build a document with ~10KB of MessagePack data.
    // Structure: { field_0: "...100 bytes...", ..., field_99: "...", target: { nested: { value: 42 } } }
    let mut entries: Vec<(rmpv::Value, rmpv::Value)> = Vec::new();

    // 100 fields × ~100 bytes each ≈ 10KB
    for i in 0..100 {
        let key = rmpv::Value::String(format!("field_{i:03}").into());
        let val = rmpv::Value::String("x".repeat(90).into());
        entries.push((key, val));
    }

    // Target path at the end: target.nested.value = 42
    let nested = rmpv::Value::Map(vec![(
        rmpv::Value::String("nested".into()),
        rmpv::Value::Map(vec![(
            rmpv::Value::String("value".into()),
            rmpv::Value::Integer(42.into()),
        )]),
    )]);
    entries.push((rmpv::Value::String("target".into()), nested));

    rmpv::Value::Map(entries)
}

fn bench_projection(c: &mut Criterion) {
    let doc = build_10kb_document();
    let bytes = {
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &doc).expect("encode");
        buf
    };

    let doc_size = bytes.len();
    let mut group = c.benchmark_group("document/projection");

    // Baseline: full deserialization + clone-based extract_at_path
    group.bench_with_input(
        BenchmarkId::new("full_deser_then_extract", doc_size),
        &bytes,
        |b, bytes| {
            b.iter(|| {
                let full: rmpv::Value =
                    rmpv::decode::read_value(&mut std::io::Cursor::new(bytes.as_slice()))
                        .expect("decode");
                coordinode_core::graph::document::extract_at_path(
                    &full,
                    &["target", "nested", "value"],
                )
            });
        },
    );

    // Optimized: extract_at_path_bytes (owned traversal, drops unneeded entries)
    group.bench_with_input(
        BenchmarkId::new("extract_at_path_bytes", doc_size),
        &bytes,
        |b, bytes| {
            b.iter(|| {
                coordinode_core::graph::document::extract_at_path_bytes(
                    bytes,
                    &["target", "nested", "value"],
                )
            });
        },
    );

    // Also test with target at the beginning (best case for early termination)
    let mut early_entries: Vec<(rmpv::Value, rmpv::Value)> = Vec::new();
    let nested = rmpv::Value::Map(vec![(
        rmpv::Value::String("nested".into()),
        rmpv::Value::Map(vec![(
            rmpv::Value::String("value".into()),
            rmpv::Value::Integer(42.into()),
        )]),
    )]);
    early_entries.push((rmpv::Value::String("target".into()), nested));
    for i in 0..100 {
        let key = rmpv::Value::String(format!("field_{i:03}").into());
        let val = rmpv::Value::String("x".repeat(90).into());
        early_entries.push((key, val));
    }
    let early_doc = rmpv::Value::Map(early_entries);
    let early_bytes = {
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &early_doc).expect("encode");
        buf
    };

    group.bench_with_input(
        BenchmarkId::new("bytes_target_first", early_bytes.len()),
        &early_bytes,
        |b, bytes| {
            b.iter(|| {
                coordinode_core::graph::document::extract_at_path_bytes(
                    bytes,
                    &["target", "nested", "value"],
                )
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_projection);
criterion_main!(benches);
