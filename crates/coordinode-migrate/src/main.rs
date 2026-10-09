//! `coordinode-migrate --data <dir> [--apply | --check]`: report, or rewrite,
//! what in a stopped server's data directory an earlier build wrote in a
//! shape this build no longer reads.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage: coordinode-migrate --data <dir> [--apply | --check | --journal-stats]\n\n\
    Reports what in the data directory an earlier build wrote in a shape this\n\
    build no longer reads and a migration knows. --apply rewrites it; every\n\
    replaced file is kept as a hard link under <dir>/migrate-backup/<run>/.\n\
    --check decodes every catalog record of the store and its checkpoints and\n\
    lists the ones this build cannot read, migrated or not. --journal-stats\n\
    totals what the store's journal holds, by kind of write and key prefix.\n\
    Stop the server first.";

enum Mode {
    Survey,
    Apply,
    Check,
    JournalStats,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut data = None;
    let mut mode = Mode::Survey;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--data" => data = iter.next().map(PathBuf::from),
            "--apply" => mode = Mode::Apply,
            "--check" => mode = Mode::Check,
            "--journal-stats" => mode = Mode::JournalStats,
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument {other}\n\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(data) = data else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let outcome = match mode {
        Mode::Survey => migrate(&data, false),
        Mode::Apply => migrate(&data, true),
        Mode::Check => check(&data),
        Mode::JournalStats => journal_stats(&data),
    };
    match outcome {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn migrate(data: &Path, apply: bool) -> anyhow::Result<bool> {
    let found = if apply {
        let run = format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs()
        );
        coordinode_migrate::apply(data, &run)?
    } else {
        coordinode_migrate::survey(data)?
    };
    let mut total = 0;
    for (migration, findings) in &found {
        println!("{}: {}", migration.name, migration.summary);
        if findings.is_empty() {
            println!("  nothing found");
        }
        for finding in findings {
            println!("  {}: {}", finding.path.display(), finding.detail);
        }
        total += findings.len();
    }
    match (total, apply) {
        (0, _) => println!("nothing a migration knows is left to rewrite"),
        (n, false) => println!("{n} places to rewrite; run again with --apply"),
        (n, true) => println!("{n} places rewritten"),
    }
    Ok(true)
}

fn journal_stats(data: &Path) -> anyhow::Result<bool> {
    let s = coordinode_migrate::journal_stats::journal_stats(data, 25)?;
    let hours = s.span.map_or(0.0, |(lo, hi)| {
        // HLC: the wall clock in milliseconds above a 16-bit logical counter.
        ((hi >> 16).saturating_sub(lo >> 16)) as f64 / 3_600_000.0
    });
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    println!(
        "{} records, {} proposals over {hours:.2} h; envelopes {:.1} MiB, frames {:.1} MiB",
        s.records,
        s.proposals,
        mib(s.envelope_bytes),
        mib(s.frame_bytes)
    );
    println!("class: writes, keys MiB, values MiB, distinct keys, most writes of one key");
    for (class, c) in &s.classes {
        println!(
            "  {class}: {}, {:.1}, {:.1}, {}, {}",
            c.writes,
            mib(c.key_bytes),
            mib(c.value_bytes),
            c.distinct_keys,
            c.max_per_key
        );
    }
    println!("keys written most often:");
    for (class, key, n) in &s.hottest {
        println!("  {n} x {class} {key}");
    }
    println!("node rewrites per label: rewrites, MiB written, MiB that changed");
    for (label, r) in &s.node_rewrites {
        println!(
            "  {label}: {}, {:.1}, {:.2}",
            r.rewrites,
            mib(r.bytes),
            mib(r.changed_bytes)
        );
        for (field, changes, size) in r.fields.iter().take(8) {
            println!("    {field}: {size} bytes, changed in {changes} rewrites");
        }
    }
    Ok(true)
}

fn check(data: &Path) -> anyhow::Result<bool> {
    let mut clean = true;
    for store in coordinode_migrate::check::check(data)? {
        println!("{}", store.path.display());
        let counts: Vec<String> = store
            .checked
            .iter()
            .map(|(what, n)| format!("{n} {what}"))
            .collect();
        println!("  checked: {}", counts.join(", "));
        for bad in &store.unreadable {
            clean = false;
            println!("  UNREADABLE {} {}: {}", bad.what, bad.key, bad.error);
            println!("    {}", bad.content);
        }
    }
    if clean {
        println!("every catalog record reads");
    }
    Ok(clean)
}
