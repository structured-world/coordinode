//! `coordinode-migrate --data <dir> [--apply]`: report, or rewrite, what in a
//! stopped server's data directory an earlier build wrote in a shape this
//! build no longer reads.

use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage: coordinode-migrate --data <dir> [--apply]\n\n\
    Reports what in the data directory an earlier build wrote in a shape this\n\
    build no longer reads. --apply rewrites it; every replaced file is kept as a\n\
    hard link under <dir>/migrate-backup/<run>/. Stop the server first.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut data = None;
    let mut apply = false;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--data" => data = iter.next().map(PathBuf::from),
            "--apply" => apply = true,
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
    match run(&data, apply) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(data: &std::path::Path, apply: bool) -> anyhow::Result<()> {
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
        (0, _) => println!("the directory is current"),
        (n, false) => println!("{n} files to rewrite; run again with --apply"),
        (n, true) => println!("{n} files rewritten"),
    }
    Ok(())
}
