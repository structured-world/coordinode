//! An application embedding CoordiNode: it opens a database on disk, runs
//! parameterized statements, commits one transaction, rolls another back,
//! handles a typed refusal, closes the database and reopens the directory
//! to check what persisted.
//!
//! ```text
//! cargo run -- [DIRECTORY]
//! ```
//!
//! `DIRECTORY` (default `coordinode-quickstart-data` in the current
//! directory) belongs to this program: the first run creates the database
//! there, every later run adds to it under the next run number, and nothing
//! in it is ever deleted. Point it at a directory of its own, never at
//! another application's database.

use std::collections::HashMap;
use std::path::PathBuf;

use coordinode_core::graph::types::Value;
use coordinode_embed::{Database, DatabaseError};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let dir = std::env::args_os().nth(1).map_or_else(
        || PathBuf::from("coordinode-quickstart-data"),
        PathBuf::from,
    );

    let run = write(&dir)?;
    // `write` dropped its handle: the directory is free to open again.
    verify(&dir, run)?;
    println!(
        "run {run} in {}: committed rows persisted, rolled-back rows absent",
        dir.display()
    );
    Ok(())
}

/// One run's writes. Returns the run's number.
fn write(dir: &PathBuf) -> Result<i64> {
    // Created on the first run, opened on every later one.
    let mut db = Database::open(dir)?;

    // Schema statements are safe to repeat: an existing constraint is kept.
    db.execute_cypher(
        "CREATE CONSTRAINT account_email IF NOT EXISTS \
         FOR (a:Account) REQUIRE a.email IS UNIQUE",
    )?;

    // Each run is numbered after the runs before it, so a repeat run adds
    // its own rows instead of colliding with earlier ones.
    let rows = db.execute_cypher("MATCH (r:QuickstartRun) RETURN count(r) AS runs")?;
    let run = int(rows.first().and_then(|row| row.get("runs")))? + 1;
    db.execute_cypher_with_params(
        "CREATE (:QuickstartRun {run: $run})",
        params([("run", Value::Int(run))]),
    )?;

    // Two statements, one atomic commit: both land or neither does.
    let alice = Value::String(format!("alice-{run}@example.com"));
    let tx = db.begin_transaction()?;
    db.execute_in_transaction(
        tx,
        "CREATE (:Account {email: $email, run: $run})",
        Some(params([("email", alice.clone()), ("run", Value::Int(run))])),
    )?;
    db.execute_in_transaction(
        tx,
        "MATCH (a:Account {email: $email}) \
         CREATE (a)-[:OWNS]->(:Wallet {run: $run, balance: 100})",
        Some(params([("email", alice.clone()), ("run", Value::Int(run))])),
    )?;
    let receipt = db.commit_transaction(tx)?;
    println!("run {run}: committed at {}", receipt.commit_ts.as_raw());

    // Rolled back explicitly: nothing it wrote is kept.
    let tx = db.begin_transaction()?;
    db.execute_in_transaction(
        tx,
        "CREATE (:Account {email: $email, run: $run})",
        Some(params([
            ("email", Value::String(format!("bob-{run}@example.com"))),
            ("run", Value::Int(run)),
        ])),
    )?;
    db.rollback_transaction(tx)?;

    // An expected refusal arrives as a typed error to match on, never as
    // text to parse; the refused write changes nothing.
    match db.execute_cypher_with_params(
        "CREATE (:Account {email: $email, run: $run})",
        params([("email", alice), ("run", Value::Int(run))]),
    ) {
        Err(DatabaseError::UniqueViolation { index, .. }) => {
            println!("run {run}: a second account with the same email refused by {index}");
        }
        Ok(_) => return Err("a duplicate email was accepted".into()),
        Err(other) => return Err(other.into()),
    }

    // Dropping the database closes it: background work stops, memtables are
    // flushed, and the directory can be opened again.
    drop(db);
    Ok(run)
}

/// Reopen the directory and check what this run left in it.
fn verify(dir: &PathBuf, run: i64) -> Result<()> {
    let mut db = Database::open(dir)?;
    let committed = db.execute_cypher_with_params(
        "MATCH (a:Account {run: $run})-[:OWNS]->(w:Wallet) RETURN a.email AS email, w.balance AS balance",
        params([("run", Value::Int(run))]),
    )?;
    if committed.len() != 1 {
        return Err(format!(
            "expected the committed account and wallet, found {} rows",
            committed.len()
        )
        .into());
    }
    let rolled_back = db.execute_cypher_with_params(
        "MATCH (a:Account {email: $email}) RETURN a",
        params([("email", Value::String(format!("bob-{run}@example.com")))]),
    )?;
    if !rolled_back.is_empty() {
        return Err("the rolled-back account is in the database".into());
    }
    let accounts = db.execute_cypher_with_params(
        "MATCH (a:Account {run: $run}) RETURN count(a) AS n",
        params([("run", Value::Int(run))]),
    )?;
    if int(accounts.first().and_then(|row| row.get("n")))? != 1 {
        return Err("the refused duplicate left an account behind".into());
    }
    Ok(())
}

fn params<const N: usize>(pairs: [(&str, Value); N]) -> HashMap<String, Value> {
    pairs
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect()
}

fn int(value: Option<&Value>) -> Result<i64> {
    match value {
        Some(Value::Int(n)) => Ok(*n),
        other => Err(format!("expected an integer, got {other:?}").into()),
    }
}
