//! The scenario of the statement cost benches: an indexed point read and a
//! create, update and delete of one node, as parameterized statements.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

/// Nodes the reads look up.
const NODES: i64 = 1000;
/// Reads measured.
pub const READS: usize = 20_000;
/// Create, update and delete rounds measured.
pub const ROUNDS: usize = 2_000;
/// The update and the delete of a round's node, found by its key.
const UPDATE: &str = "MATCH (n:Item) WHERE n.id = $id SET n.v = 1";
const DELETE: &str = "MATCH (n:Item) WHERE n.id = $id DELETE n";

/// A database holding the read nodes and their index, recording statements
/// for the advisor or not.
pub fn open(dir: &std::path::Path, advisory: bool) -> Database {
    let mut db = Database::open(dir).expect("open");
    db.execute_cypher(&format!(
        "UNWIND range(0, {}) AS i CREATE (:User {{id: i, name: 'u' + toString(i)}})",
        NODES - 1
    ))
    .expect("seed");
    db.execute_cypher("CREATE INDEX user_id ON :User(id)")
        .expect("index");
    // The rounds find their node by its key: the update and the delete go
    // through the index, not a scan of the label's churned history.
    db.execute_cypher("CREATE INDEX item_id ON :Item(id)")
        .expect("index");
    for query in [UPDATE, DELETE] {
        let plan = db.explain_cypher(query).expect("explain");
        println!("plan of {query}: {}", plan.replace('\n', " | "));
        assert!(
            plan.contains("IndexScan"),
            "{query} does not use its index: {plan}"
        );
    }
    if !advisory {
        switch_off(&db);
    }
    db
}

fn param(id: i64) -> HashMap<String, Value> {
    HashMap::from([("id".to_string(), Value::Int(id))])
}

/// The time of each read, run inside `around`.
pub fn reads(db: &mut Database, around: &mut dyn FnMut(&mut dyn FnMut())) -> Vec<Duration> {
    let mut times = Vec::with_capacity(READS);
    around(&mut || {
        for i in 0..READS {
            let started = Instant::now();
            db.execute_cypher_with_params(
                "MATCH (n:User) WHERE n.id = $id RETURN n.name",
                param(i as i64 % NODES),
            )
            .expect("read");
            times.push(started.elapsed());
        }
    });
    times
}

/// The time of each create, update and delete, run inside `around`.
pub fn writes(db: &mut Database, around: &mut dyn FnMut(&mut dyn FnMut())) -> Vec<Duration> {
    let mut times = Vec::with_capacity(ROUNDS * 3);
    around(&mut || {
        for i in 0..ROUNDS {
            let id = 1_000_000 + i as i64;
            for query in ["CREATE (:Item {id: $id, v: 0})", UPDATE, DELETE] {
                let started = Instant::now();
                db.execute_cypher_with_params(query, param(id))
                    .expect("write");
                times.push(started.elapsed());
            }
        }
    });
    times
}

/// p50, p99 and p999 of `times`.
// The allocation bench reports no times.
#[allow(dead_code)]
pub fn percentiles(mut times: Vec<Duration>) -> String {
    times.sort_unstable();
    let at = |q: usize| times[(times.len() - 1) * q / 1000];
    format!("p50={:?} p99={:?} p999={:?}", at(500), at(990), at(999))
}

/// Turn recording off. A build without the setting records every statement:
/// the comparison runs it with `advisory` on only.
fn switch_off(db: &Database) {
    db.set_auto_advisory(false); // advisor-switch
}
