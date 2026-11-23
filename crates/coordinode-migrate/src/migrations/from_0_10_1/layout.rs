//! A store's trees at the top of its directory move into their sets.
//!
//! 0.10.1 laid a store out as one directory per partition, and the columnar
//! tables beside them, at the top of the endpoint's directory. This build
//! opens the deployment's partitions under the system set and the others,
//! the tables among them, under the namespace's set, and refuses a store
//! with a tree left at the top. Each tree moves whole by a rename: nothing
//! in it is rewritten, so nothing goes to the backup, and a move is undone
//! by renaming the directory back.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, bail};
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::scope::{Scope, TABLES_DIR, TreeId};

use super::NAMESPACE;
use crate::{Backup, Finding, Migration, stores};

pub(super) const MIGRATION: Migration = Migration {
    name: "tree-sets",
    summary: "trees at the top of a store move into the system set and the namespace's set",
    survey,
    apply,
};

/// Each tree directory `store` keeps at its top, with where it goes.
fn moves(store: &Path) -> Vec<(PathBuf, PathBuf)> {
    let mut moves: Vec<(PathBuf, PathBuf)> = Partition::all()
        .iter()
        .map(|&part| {
            let tree = if part.is_deployment() {
                TreeId::system(part)
            } else {
                TreeId::namespace(NAMESPACE, part)
            };
            (store.join(part.name()), store.join(tree.dir()))
        })
        .collect();
    moves.push((
        store.join(TABLES_DIR),
        store
            .join(Scope::Namespace(NAMESPACE).dir())
            .join(TABLES_DIR),
    ));
    moves.retain(|(from, _)| from.is_dir());
    moves
}

fn survey(data: &Path) -> anyhow::Result<Vec<Finding>> {
    let mut findings = Vec::new();
    for store in stores(data)? {
        let n = moves(&store).len();
        if n > 0 {
            findings.push(Finding {
                path: store,
                detail: format!("{n} trees at the top of the store"),
            });
        }
    }
    Ok(findings)
}

fn apply(finding: &Finding, _backup: &Backup) -> anyhow::Result<()> {
    for (from, to) in moves(&finding.path) {
        if to.exists() {
            // A store holds one copy of each tree: two would leave no way to
            // tell which one the store's records describe.
            bail!(
                "{} and {} both hold the tree; move one of them aside",
                from.display(),
                to.display()
            );
        }
        let parent = to.parent().context("a tree directory has a parent")?;
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        std::fs::rename(&from, &to)
            .with_context(|| format!("move {} to {}", from.display(), to.display()))?;
        sync_dir(parent)?;
        sync_dir(&finding.path)?;
    }
    Ok(())
}

/// Make a rename into or out of `dir` durable. Windows has no directory
/// handle to sync; NTFS journals the rename itself.
fn sync_dir(dir: &Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    std::fs::File::open(dir)
        .and_then(|d| d.sync_all())
        .with_context(|| format!("sync {}", dir.display()))?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
