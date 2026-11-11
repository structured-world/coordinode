//! The vector population the vector benches load: the same components for
//! the same seed on every run and every revision.

use std::collections::HashMap;

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

/// Vectors created per statement while loading.
const BATCH: usize = 500;

/// Components in `[-1, 1)`, deterministic per seed.
pub fn synth(seed: u64, dim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..dim)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 40) as f32 / (1u32 << 24) as f32;
            2.0 * u - 1.0
        })
        .collect()
}

pub fn params(entries: Vec<(&str, Value)>) -> HashMap<String, Value> {
    entries
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

/// `n` nodes of `label`, `i` from 0 and `e` the vector of seed `seed + i`.
pub fn load(db: &Database, label: &str, dim: usize, n: usize, seed: u64) {
    for start in (0..n).step_by(BATCH) {
        let rows: Vec<Value> = (start..(start + BATCH).min(n))
            .map(|i| {
                Value::Map(
                    [
                        ("i".to_string(), Value::Int(i as i64)),
                        ("e".to_string(), Value::Vector(synth(seed + i as u64, dim))),
                    ]
                    .into(),
                )
            })
            .collect();
        db.execute_cypher_shared(
            &format!("UNWIND $rows AS r CREATE (:{label} {{i: r.i, e: r.e}})"),
            Some(params(vec![("rows", Value::Array(rows))])),
            None,
            None,
            None,
        )
        .expect("load");
    }
}

/// A store with a euclidean vector index on `:V(e)` made before its
/// vectors arrive, and `n` vectors of dimension `dim` in both `V` and `W`
/// (the same vectors, `W` without an index).
pub fn open_loaded(dir: &std::path::Path, dim: usize, n: usize) -> Database {
    let db = Database::open(dir).expect("open");
    db.execute_cypher_shared(
        &format!(
            "CREATE VECTOR INDEX v_e ON :V(e) OPTIONS {{metric: \"euclidean\", dimensions: {dim}}}"
        ),
        None,
        None,
        None,
        None,
    )
    .expect("index");
    load(&db, "V", dim, n, 10);
    load(&db, "W", dim, n, 10);
    db
}
