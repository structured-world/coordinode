use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_core::index::encoding::{encode_entry_key, encode_tuple, encode_unique_entry_key};
use coordinode_modality::{
    IndexStore as _, LocalIndexStore, LocalNodeStore, Mismatch, NodeStore as _,
};

use super::super::super::test_env::{TestEnv, commit};
use super::*;
use crate::index::{
    BuildFailure, IndexBuildOutcome, IndexBuildRecord, IndexDescriptor, IndexState,
};

/// Users indexed by a unique `email` and a plain `city`.
const USERS: u64 = 40;

/// Records of another label, large, that no checked index covers.
const LARGE: u64 = 10;

struct Store {
    env: Arc<TestEnv>,
    email: IndexDefinition,
    city: IndexDefinition,
}

fn put(env: &TestEnv, label: &str, id: u64, props: &[(&str, Value)]) {
    let fields = env.fields_now();
    let mut record = NodeRecord::new(label);
    for (name, value) in props {
        record.set(fields.lookup(name).expect("field"), value.clone());
    }
    let mut txn = env.begin();
    LocalNodeStore
        .put(&mut txn, 1, NodeId::from_raw(id), &record)
        .expect("put");
    commit(&mut txn).expect("commit node");
}

fn user(env: &TestEnv, id: u64, email: &str) {
    let city = format!("c{}", id % 7);
    put(
        env,
        "User",
        id,
        &[
            ("email", Value::String(email.into())),
            ("city", Value::String(city)),
        ],
    );
}

/// Publish and build `descriptor`, and return it as it serves.
fn built(env: &Arc<TestEnv>, descriptor: IndexDescriptor) -> IndexDefinition {
    let store = LocalIndexStore::new(&env.engine);
    let mut descriptor = descriptor;
    descriptor.state = IndexState::Building {
        written: 0,
        estimated_total: 0,
    };
    let mut txn = env.begin();
    let def = store
        .publish_definition_txn(&mut txn, descriptor)
        .expect("publish");
    store
        .put_build_txn(
            &mut txn,
            &IndexBuildRecord::accepted(def.id, def.generation, BuildFailure::Withdraw),
            None,
        )
        .expect("admit build");
    commit(&mut txn).expect("commit publication");
    // Ended before the build starts, which waits for every transaction
    // opened before its index.
    drop(txn);
    env.registry
        .register_published(&env.engine, def.clone())
        .expect("register");
    let builds = env.service(1);
    builds.submit(def.generation).expect("submit");
    let outcome = builds.wait_here(def.generation);
    assert!(
        matches!(outcome, Some(IndexBuildOutcome::Published { .. })),
        "{outcome:?}"
    );
    builds.shutdown();
    store
        .load_definition(def.id)
        .expect("load")
        .expect("published")
}

fn store() -> Store {
    let env = TestEnv::open();
    let mut fields = FieldInterner::new();
    fields.intern("email");
    fields.intern("city");
    fields.intern("body");
    env.set_fields(&fields);
    for id in 1..=USERS {
        user(&env, id, &format!("u{id}@x"));
    }
    for id in 1..=LARGE {
        put(
            &env,
            "Large",
            1000 + id,
            &[("body", Value::String("z".repeat(64 * 1024)))],
        );
    }
    let email = built(
        &env,
        IndexDescriptor::btree("user_email", "User", "email").unique(),
    );
    let city = built(&env, IndexDescriptor::btree("user_city", "User", "city"));
    Store { env, email, city }
}

fn string_tuple(value: &str) -> Vec<u8> {
    encode_tuple(&[Value::String(value.into())]).expect("tuple")
}

/// Run one pass over `defs` and return what it found, sorted, per
/// generation.
fn pass(
    s: &Store,
    defs: &[&IndexDefinition],
    budget: &Budget,
) -> Result<Option<(Vec<Findings>, VerifyStats)>, VerifyError> {
    let fields = s.env.fields_now();
    let field_of = |name: &str| fields.lookup(name);
    let interpretations: Vec<_> = defs.iter().map(|d| d.interpretation(&field_of)).collect();
    let targets: Vec<Target<'_>> = defs
        .iter()
        .zip(&interpretations)
        .map(|(def, interpretation)| Target {
            def,
            interpretation,
        })
        .collect();
    let mut out = verify(&s.env.engine, 1, &targets, budget, &AtomicBool::new(false))?;
    if let Some((findings, _)) = out.as_mut() {
        for found in findings {
            found.mismatches.sort_by_key(|m| format!("{m:?}"));
        }
    }
    Ok(out)
}

fn unbounded() -> Budget {
    Budget {
        memory_bytes: usize::MAX,
        scratch: None,
    }
}

fn sorted(mut mismatches: Vec<Mismatch>) -> Vec<Mismatch> {
    mismatches.sort_by_key(|m| format!("{m:?}"));
    mismatches
}

/// Damage both indexes every way an entry can be wrong, and return what a
/// check must find: entries missing, entries no record holds, a unique entry
/// naming the wrong holder or none that decodes, and two records holding
/// one unique value.
fn damage(s: &Store) -> (Vec<Mismatch>, Vec<Mismatch>) {
    let engine = &s.env.engine;
    // Raw state planted under the stores, as a defect would leave it.
    let idx = coordinode_storage::engine::partition::Partition::Idx;
    let (email, city) = (s.email.generation, s.city.generation);
    // Plain: user 3's entry lost; an entry for node 999, which holds nothing.
    let c3 = string_tuple("c3");
    engine
        .delete(idx, &encode_entry_key(city, &c3, 3))
        .expect("lose");
    let ghost = string_tuple("nowhere");
    engine
        .put(idx, &encode_entry_key(city, &ghost, 999), &[])
        .expect("plant");
    // Unique: user 5's entry lost; user 6's value naming user 7; user 8's
    // value naming no node that decodes.
    let u5 = string_tuple("u5@x");
    engine
        .delete(idx, &encode_unique_entry_key(email, &u5))
        .expect("lose");
    let u6 = string_tuple("u6@x");
    engine
        .put(
            idx,
            &encode_unique_entry_key(email, &u6),
            &7u64.to_be_bytes(),
        )
        .expect("misattribute");
    let u8 = string_tuple("u8@x");
    engine
        .put(idx, &encode_unique_entry_key(email, &u8), &[1, 2, 3])
        .expect("garble");
    // Records: user 50 takes user 9's email behind the index's back.
    user(&s.env, 50, "u9@x");
    let u9 = string_tuple("u9@x");
    let c50 = string_tuple("c1");
    let email_found = vec![
        Mismatch::Missing {
            node: 5,
            valid_from: None,
            tuple: u5,
        },
        Mismatch::Missing {
            node: 6,
            valid_from: None,
            tuple: u6.clone(),
        },
        Mismatch::Extra {
            node: 7,
            valid_from: None,
            tuple: u6,
        },
        Mismatch::Missing {
            node: 8,
            valid_from: None,
            tuple: u8.clone(),
        },
        Mismatch::Extra {
            node: u64::MAX,
            valid_from: None,
            tuple: u8,
        },
        Mismatch::SourceDuplicate {
            nodes: [9, 50],
            tuple: u9,
        },
    ];
    let city_found = vec![
        Mismatch::Missing {
            node: 3,
            valid_from: None,
            tuple: c3,
        },
        Mismatch::Extra {
            node: 999,
            valid_from: None,
            tuple: ghost,
        },
        // User 50's own city entry was never written.
        Mismatch::Missing {
            node: 50,
            valid_from: None,
            tuple: c50,
        },
    ];
    (sorted(email_found), sorted(city_found))
}

/// A store that agrees with its records finds nothing. Both generations are
/// checked by one read of the records, and records of a label no checked
/// index covers are never decoded, however large.
#[test]
fn one_read_checks_every_generation_and_decodes_only_their_labels() {
    let s = store();
    let (findings, stats) = pass(&s, &[&s.email, &s.city], &unbounded())
        .expect("pass")
        .expect("not stopped");
    assert!(
        findings.iter().all(|f| f.mismatches.is_empty()),
        "{findings:?}"
    );
    assert_eq!(stats.source_rows, USERS + LARGE, "every row read once");
    assert_eq!(stats.decoded_rows, USERS, "only the users decoded");
    assert!(
        stats.decoded_bytes < stats.source_bytes / 10,
        "the large records were read, never decoded: {stats:?}"
    );
    for found in &findings {
        assert_eq!(found.checked, 2 * USERS, "every record and every entry");
    }
    assert_eq!(stats.expected_entries, 2 * USERS);
    assert_eq!(stats.stored_entries, 2 * USERS);
}

/// Each way an entry can disagree with its records is found, named as the
/// repairs need it, and attributed to its own generation.
#[test]
fn every_disagreement_is_found_in_its_own_generation() {
    let s = store();
    let (email, city) = damage(&s);
    let (findings, _) = pass(&s, &[&s.email, &s.city], &unbounded())
        .expect("pass")
        .expect("not stopped");
    assert_eq!(findings[0].mismatches, email);
    assert_eq!(findings[1].mismatches, city);
}

/// Checked one at a time, each generation finds exactly what the shared
/// pass found for it.
#[test]
fn a_shared_pass_finds_what_separate_passes_find() {
    let s = store();
    damage(&s);
    let (shared, _) = pass(&s, &[&s.email, &s.city], &unbounded())
        .expect("pass")
        .expect("not stopped");
    for (i, def) in [&s.email, &s.city].into_iter().enumerate() {
        let (alone, _) = pass(&s, &[def], &unbounded())
            .expect("pass")
            .expect("not stopped");
        assert_eq!(alone[0].mismatches, shared[i].mismatches);
        assert_eq!(alone[0].checked, shared[i].checked);
    }
}

/// Past its memory a pass spills sorted runs to scratch files and merges
/// them back: it finds exactly what a pass held in memory finds.
#[test]
fn a_pass_past_its_memory_spills_and_finds_the_same() {
    let s = store();
    damage(&s);
    let scratch = tempfile::tempdir().expect("scratch");
    let (held, _) = pass(&s, &[&s.email, &s.city], &unbounded())
        .expect("pass")
        .expect("not stopped");
    let (spilled, stats) = pass(
        &s,
        &[&s.email, &s.city],
        &Budget {
            memory_bytes: 256,
            scratch: Some(scratch.path().join("runs")),
        },
    )
    .expect("pass")
    .expect("not stopped");
    assert!(stats.spilled_runs > 1, "{stats:?}");
    assert!(stats.spilled_bytes > 0);
    for (held, spilled) in held.iter().zip(&spilled) {
        assert_eq!(held.mismatches, spilled.mismatches);
        assert_eq!(held.checked, spilled.checked);
    }
}

/// A pass that outgrows its memory with nowhere to spill, or whose scratch
/// files cannot be written, fails: it never concludes from a part of the
/// comparison.
#[test]
fn a_pass_that_cannot_spill_fails() {
    let s = store();
    let tiny = |scratch| Budget {
        memory_bytes: 256,
        scratch,
    };
    assert!(matches!(
        pass(&s, &[&s.email], &tiny(None)),
        Err(VerifyError::NoScratch { limit: 256 })
    ));
    // A scratch directory that cannot be made: its parent is a file.
    let dir = tempfile::tempdir().expect("dir");
    let file = dir.path().join("file");
    std::fs::write(&file, b"x").expect("file");
    assert!(matches!(
        pass(&s, &[&s.email], &tiny(Some(file.join("runs")))),
        Err(VerifyError::Scratch(_))
    ));
}

/// A pass told to stop ends without findings.
#[test]
fn a_stopped_pass_finds_nothing() {
    let s = store();
    damage(&s);
    let fields = s.env.fields_now();
    let field_of = |name: &str| fields.lookup(name);
    let interpretation = s.email.interpretation(&field_of);
    let targets = [Target {
        def: &s.email,
        interpretation: &interpretation,
    }];
    let out = verify(
        &s.env.engine,
        1,
        &targets,
        &unbounded(),
        &AtomicBool::new(true),
    )
    .expect("pass");
    assert!(out.is_none());
}

/// Writes after a pass pinned its snapshot are not part of what it read:
/// records and entries are compared as of one moment, so a write the index
/// maintains never shows as a disagreement.
#[test]
fn a_pass_reads_records_and_entries_at_one_snapshot() {
    let s = store();
    let (snapshot, _pin) = s.env.engine.pin_snapshot();
    // A record whose entry the index never got, written after the pin.
    user(&s.env, 60, "u60@x");
    let fields = s.env.fields_now();
    let field_of = |name: &str| fields.lookup(name);
    let interpretation = s.email.interpretation(&field_of);
    let mut expected = Expected::new(Vec::new(), Run::default());
    let mut findings = Findings::default();
    let mut stats = VerifyStats::default();
    compare(
        &s.env.engine,
        snapshot,
        &Target {
            def: &s.email,
            interpretation: &interpretation,
        },
        &mut expected,
        &mut findings,
        &mut stats,
    )
    .expect("compare");
    assert_eq!(
        stats.stored_entries, USERS,
        "the entries as of the pin, nothing written since"
    );
    let (now, _) = pass(&s, &[&s.email], &unbounded())
        .expect("pass")
        .expect("not stopped");
    assert_eq!(
        now[0].mismatches,
        vec![Mismatch::Missing {
            node: 60,
            valid_from: None,
            tuple: string_tuple("u60@x"),
        }],
        "a pass begun after the write reads it"
    );
}
