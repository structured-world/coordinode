//! Every stored version carries the commit that wrote it.
//!
//! This build keeps a record's revision in its value: every value of a key
//! outside the reserved namespace, in each partition whose records are
//! replaced rather than folded, ends with the timestamp of the commit that
//! wrote it. 0.10.1 kept none, and its physical sequence numbers do not name
//! the commit either: compaction settled old versions to zero, and an
//! installed copy wrote its rows at a sequence number of its own.
//!
//! So each tree is written anew from its whole history, every version and
//! merge operand at its own sequence number, with the commit it is known to
//! carry or, where that is not known, the unknown revision. A version is
//! known only when the journal holds the write itself: an entry at that
//! timestamp writing that key with those bytes, which a 0.10.1 commit applied
//! at its own timestamp. A record of unknown revision reads as before; a
//! write conditioned on its revision is refused until a write replaces it.
//!
//! This runs on the trees at the top of a 0.10.1 store, before they move
//! into their sets: until every tree here is written, the store keeps a
//! layout this build refuses to open. A tree is written beside the old one
//! and renamed into place; the old one's files are kept in the backup.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, bail};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::rebuild::{StoredVersion, rewrite_tree_values};

use crate::{Backup, Finding, Migration, oplog, stores};

pub(super) const MIGRATION: Migration = Migration {
    name: "revision-stamps",
    summary: "every stored version carries the commit that wrote it, or the unknown revision",
    survey,
    apply,
};

/// The file, at the top of a store, naming the trees already written with
/// revisions, one partition name a line: a run cut short goes on from it.
const DONE_FILE: &str = "REVISION_STAMPS";

/// Where a tree is written before it replaces the old one.
const STAGED_SUFFIX: &str = ".stamping";

/// Where the old tree waits between the two renames.
const REPLACED_SUFFIX: &str = ".replaced";

/// The revision of a version whose commit is not known.
const UNKNOWN: u64 = 0;

/// The partitions whose trees `store` holds at its top and has not written
/// with revisions yet.
fn pending(store: &Path) -> anyhow::Result<Vec<Partition>> {
    let done = done(store)?;
    Ok(Partition::all()
        .iter()
        .copied()
        .filter(|part| part.carries_revision())
        .filter(|part| {
            let top = store.join(part.name());
            let replaced = with_suffix(&top, REPLACED_SUFFIX).is_dir();
            // A tree written and recorded may still leave the old one to
            // remove.
            if done.contains(part.name()) {
                return replaced;
            }
            top.is_dir() || with_suffix(&top, STAGED_SUFFIX).is_dir() || replaced
        })
        .collect())
}

fn done(store: &Path) -> anyhow::Result<HashSet<String>> {
    let path = store.join(DONE_FILE);
    if !path.exists() {
        return Ok(HashSet::new());
    }
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    Ok(text.lines().map(str::to_owned).collect())
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Whether `store` still lays its trees out at its top: a store of 0.10.1.
fn flat(store: &Path) -> bool {
    Partition::all()
        .iter()
        .any(|part| store.join(part.name()).is_dir())
}

fn survey(data: &Path) -> anyhow::Result<Vec<Finding>> {
    let mut findings = Vec::new();
    for store in stores(data)? {
        let pending = pending(&store)?;
        // A store whose every tree is written but whose marker does not say
        // so yet: the run was cut short between the two.
        let unmarked = flat(&store) && !coordinode_storage::format::marker_has_revisions(&store)?;
        if !pending.is_empty() || unmarked {
            findings.push(Finding {
                path: store,
                detail: format!("{} trees without revisions", pending.len()),
            });
        }
    }
    Ok(findings)
}

fn apply(finding: &Finding, backup: &Backup) -> anyhow::Result<()> {
    let store = &finding.path;
    let known = journal_writes(store)?;
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        store,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    for part in pending(store)? {
        stamp_tree(store, &config, part, &known, backup)
            .with_context(|| format!("the {} tree of {}", part.name(), store.display()))?;
    }
    // Every tree written: the store records that its records carry their
    // revisions, which this build requires of a store it opens.
    let format = coordinode_storage::format::read_marker(store)?
        .with_context(|| format!("{} has no format marker", store.display()))?;
    backup.keep(&store.join(coordinode_storage::format::MARKER_FILE))?;
    coordinode_storage::format::write_marker(
        store,
        format,
        &coordinode_storage::engine::footprint::Footprints::default(),
    )?;
    Ok(())
}

/// The writes the journal of `store` holds, by partition, key and commit
/// timestamp, with the bytes each wrote: what proves a version's commit.
type Writes = HashMap<(Partition, Vec<u8>, u64), Vec<u8>>;

/// The writes the units of `store`'s own journal carry, each at the commit
/// timestamp its unit names. Only a put or a merge the unit states proves
/// anything: a command decided when it was applied names no write here, and
/// the journal of a checkpoint inside the store is that checkpoint's, not
/// this history's. A unit that does not read stops the run.
fn journal_writes(store: &Path) -> anyhow::Result<Writes> {
    use coordinode_core::txn::frame::{DecodeLimits, decode_proposal};
    use coordinode_core::txn::proposal::Mutation;
    use coordinode_storage::oplog::OplogOp;
    let mut writes = Writes::new();
    for path in oplog::segment_files(store)? {
        if in_other_store(store, &path) {
            continue;
        }
        let segment = oplog::read(&path)?;
        for entry in &segment.entries {
            for op in &entry.ops {
                let OplogOp::Unit { frame } = op else {
                    continue;
                };
                let unit = decode_proposal(frame, &DecodeLimits::DEFAULT).with_context(|| {
                    format!("a unit of log entry {} in {}", entry.index, path.display())
                })?;
                let at = unit.commit_ts.as_raw();
                for mutation in unit.mutations {
                    let (partition, key, bytes) = match mutation {
                        Mutation::Put {
                            partition,
                            key,
                            value,
                        } => (partition, key, value),
                        Mutation::Merge {
                            partition,
                            key,
                            operand,
                        } => (partition, key, operand),
                        _ => continue,
                    };
                    writes.insert((Partition::from(partition), key, at), bytes);
                }
            }
        }
    }
    Ok(writes)
}

/// Whether `path`, below `store`, belongs to another store inside it: a
/// directory between them carries an engine format marker.
fn in_other_store(store: &Path, path: &Path) -> bool {
    path.ancestors()
        .skip(1)
        .take_while(|dir| *dir != store)
        .any(|dir| dir.join(coordinode_storage::format::MARKER_FILE).is_file())
}

/// Write `part`'s tree of `store` anew with revisions and put it in place.
fn stamp_tree(
    store: &Path,
    config: &StorageConfig,
    part: Partition,
    known: &Writes,
    backup: &Backup,
) -> anyhow::Result<()> {
    let top = store.join(part.name());
    let staged = with_suffix(&top, STAGED_SUFFIX);
    let replaced = with_suffix(&top, REPLACED_SUFFIX);
    match (top.is_dir(), staged.is_dir(), replaced.is_dir()) {
        // Cut short while writing: written again.
        (true, true, false) => remove_dir(&staged)?,
        // Cut short between the renames: the written tree goes in place.
        (false, true, true) => {}
        // Cut short after them: the tree in place is the written one, since
        // the old one stays aside until the record of it is durable.
        (true, false, true) => {
            mark_done(store, part)?;
            return remove_dir(&replaced);
        }
        (true, false, false) => {}
        _ => bail!(
            "{} is in a state no run leaves: the tree, its staged copy and the \
             replaced one are not where a run puts them",
            top.display()
        ),
    }
    if top.is_dir() {
        let relative = |path: &Path| -> anyhow::Result<PathBuf> {
            Ok(path
                .strip_prefix(store)
                .context("a tree of the store")?
                .to_path_buf())
        };
        let mut rewrite = |version: StoredVersion<'_>| Ok(stamped(part, version, known));
        rewrite_tree_values(
            config,
            part,
            (&relative(&top)?, &relative(&staged)?),
            None,
            &mut rewrite,
        )?;
        keep_tree(&top, backup)?;
        std::fs::rename(&top, &replaced)
            .with_context(|| format!("move {} aside", top.display()))?;
        sync_dir(store)?;
    }
    std::fs::rename(&staged, &top).with_context(|| format!("put {} in place", top.display()))?;
    sync_dir(store)?;
    cut(Cut::AfterRenames)?;
    // Recorded before the old tree goes: while it is aside, the tree in
    // place is known to be the written one.
    mark_done(store, part)?;
    cut(Cut::AfterRecord)?;
    remove_dir(&replaced)
}

/// Where a test stops a run, as a crash there would.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Cut {
    AfterRenames,
    AfterRecord,
}

#[cfg(test)]
thread_local! {
    pub(super) static CUT: std::cell::Cell<Option<Cut>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn cut(at: Cut) -> anyhow::Result<()> {
    if CUT.with(|cut| cut.get()) == Some(at) {
        CUT.with(|cut| cut.set(None));
        bail!("cut short at {at:?}");
    }
    Ok(())
}

#[cfg(not(test))]
#[derive(Clone, Copy)]
enum Cut {
    AfterRenames,
    AfterRecord,
}

#[cfg(not(test))]
#[inline]
fn cut(_: Cut) -> anyhow::Result<()> {
    Ok(())
}

/// The bytes `version` of `part` is written with: its value with the commit
/// the journal proves it carries, or the unknown revision. A key of the
/// engine's reserved namespace carries none.
fn stamped(part: Partition, version: StoredVersion<'_>, known: &Writes) -> Vec<u8> {
    if version.key.first() == Some(&0x00) {
        return version.value.to_vec();
    }
    let proved = known
        .get(&(part, version.key.to_vec(), version.seqno))
        .is_some_and(|bytes| bytes.as_slice() == version.value);
    let revision = if proved && version.seqno != UNKNOWN {
        version.seqno
    } else {
        UNKNOWN
    };
    let mut value = Vec::with_capacity(version.value.len() + 8);
    value.extend_from_slice(version.value);
    value.extend_from_slice(&revision.to_le_bytes());
    value
}

/// Keep every file of the tree at `dir` in the backup.
fn keep_tree(dir: &Path, backup: &Backup) -> anyhow::Result<()> {
    let mut dirs = vec![dir.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).with_context(|| format!("list {}", dir.display()))? {
            let entry = entry.with_context(|| format!("list {}", dir.display()))?;
            let path = entry.path();
            if entry
                .file_type()
                .with_context(|| format!("stat {}", path.display()))?
                .is_dir()
            {
                dirs.push(path);
            } else {
                backup.keep(&path)?;
            }
        }
    }
    Ok(())
}

fn remove_dir(dir: &Path) -> anyhow::Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir).with_context(|| format!("remove {}", dir.display()))?;
    }
    Ok(())
}

/// Record `part` among the trees written, the whole record replaced at
/// once: a crash leaves the old record or the new one, never part of a line.
fn mark_done(store: &Path, part: Partition) -> anyhow::Result<()> {
    use std::io::Write as _;
    let mut done = done(store)?;
    if !done.insert(part.name().to_owned()) {
        return Ok(());
    }
    let mut names: Vec<&String> = done.iter().collect();
    names.sort();
    let path = store.join(DONE_FILE);
    let staged = with_suffix(&path, STAGED_SUFFIX);
    let mut file =
        std::fs::File::create(&staged).with_context(|| format!("create {}", staged.display()))?;
    for name in names {
        writeln!(file, "{name}").with_context(|| format!("write {}", staged.display()))?;
    }
    file.sync_all()
        .with_context(|| format!("sync {}", staged.display()))?;
    std::fs::rename(&staged, &path)
        .with_context(|| format!("rename {} over {}", staged.display(), path.display()))?;
    sync_dir(store)
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
