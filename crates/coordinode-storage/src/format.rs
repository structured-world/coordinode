//! The engine format of a data directory, and its migration on open.
//!
//! A directory records the engine format that wrote it in a marker file on
//! its primary endpoint. A release opens a directory of its own format, and
//! migrates one written by the format just before it, once and in one
//! direction: every step of [`MIGRATIONS`] that starts at the directory's
//! format runs, and the marker moves only once the whole open has succeeded
//! ([`settle`]), so a crash part way, or an open that refuses the directory
//! for another reason, leaves the old marker and repeats the step next time.
//! A directory of any other format is refused by name and left untouched.
//! A directory with data and no marker was written before the marker
//! existed, which is format 0.
//!
//! An offline restore claims empty directories with a second marker naming
//! the image it installs ([`begin_restore`]) and removes it once the image
//! is durable ([`finish_restore`]); meanwhile every other open is refused,
//! so a restore cut short is run again rather than served half installed.

use std::path::{Path, PathBuf};

use crate::engine::config::{Durability, StorageConfig};
use crate::error::{StorageError, StorageResult};

/// Name of the marker file in the primary endpoint's directory.
pub const MARKER_FILE: &str = "ENGINE_FORMAT";

/// One step of directory migration: turns a directory written by format
/// `from` into one of format `from + 1`. A step is idempotent, since an open
/// that ends before the marker moves runs it again.
pub struct MigrationStep {
    /// The format the step reads.
    pub from: u32,
    /// Rewrite the directory at the path in place.
    pub migrate: fn(&Path) -> StorageResult<()>,
}

/// Every migration this release ships, at most one per source format. None
/// leads into this release's format: membership records carry each member's
/// node identity, which no earlier directory records, and before the first
/// supported release an older directory is refused rather than converted.
pub static MIGRATIONS: &[MigrationStep] = &[
    // A test build runs up to two formats past this release's, so a suite
    // can take a directory through an intermediate version.
    #[cfg(feature = "test-format-bump")]
    MigrationStep {
        from: coordinode_core::version::ENGINE_FORMAT_VERSION,
        migrate: no_change,
    },
    #[cfg(feature = "test-format-bump")]
    MigrationStep {
        from: coordinode_core::version::ENGINE_FORMAT_VERSION + 1,
        migrate: no_change,
    },
];

#[cfg(feature = "test-format-bump")]
fn no_change(_: &Path) -> StorageResult<()> {
    Ok(())
}

/// What opening a directory did to its format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatOpen {
    /// No durable directory to mark: every endpoint is volatile.
    Volatile,
    /// A new directory, now marked with the current format.
    Created,
    /// Already in the current format.
    Current,
    /// Migrated from the named format to the current one; the marker moves
    /// when [`settle`] is called after the open succeeds.
    Migrated {
        /// The directory migrated.
        dir: PathBuf,
        /// The format the directory was in.
        from: u32,
        /// The format it is in now.
        to: u32,
    },
}

/// Record the format a migration reached, once the open that ran it has
/// succeeded, charging the marker to `footprints`. Nothing to do for any
/// other outcome.
pub fn settle(
    opened: &FormatOpen,
    footprints: &crate::engine::footprint::Footprints,
) -> StorageResult<()> {
    match opened {
        FormatOpen::Migrated { dir, to, .. } => write_marker(dir, *to, footprints),
        _ => Ok(()),
    }
}

/// Name of the file in each durable endpoint directory that whoever owns
/// the store there, an open engine or a restore, holds locked.
pub const LOCK_FILE: &str = "STORE_LOCK";

/// Exclusive holds on a configuration's durable endpoint directories: whoever
/// has them owns the store there until they are dropped. An open engine
/// keeps them for its lifetime.
#[derive(Debug, Default)]
pub struct DirectoryLocks(Vec<std::fs::File>);

/// What preparing the directories for an engine open found, and the holds
/// the engine keeps.
#[derive(Debug)]
pub struct Prepared {
    /// What happened to the primary directory's format.
    pub format: FormatOpen,
    /// The holds on every durable endpoint directory.
    pub locks: DirectoryLocks,
}

/// Bring the configured directories to an open: take every durable
/// endpoint's hold, refuse a directory holding an unfinished restore, and
/// bring the primary to the engine format this process runs (mark a new
/// one, migrate one of the previous format, refuse any other). Called first
/// thing by every engine open.
///
/// # Errors
///
/// [`StorageError::StoreInUse`] when another open or a restore holds a
/// directory; [`StorageError::RestoreIncomplete`]; a format refusal.
pub fn prepare(config: &StorageConfig) -> StorageResult<Prepared> {
    let dirs = durable_dirs(config);
    for &dir in &dirs {
        std::fs::create_dir_all(dir).map_err(|e| io_at("create", dir, e))?;
    }
    let resolved = canonical(&dirs)?;
    let mut locks = DirectoryLocks::default();
    // The lock files this open created: a refused open removes them, so a
    // store it refuses (another format, an unfinished restore) is left as
    // it found it.
    let mut created = Vec::new();
    for &dir in &dirs {
        let path = dir.join(LOCK_FILE);
        let existed = path.exists();
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(|e| io_at("open", &path, e))?;
        if !existed {
            created.push(path.clone());
        }
        match hold(dir, file) {
            Ok(file) => locks.0.push(file),
            Err(e) => return Err(release_created(locks, &created, e)),
        }
    }
    match prepare_held(config, &dirs, &resolved) {
        Ok(format) => Ok(Prepared { format, locks }),
        Err(e) => Err(release_created(locks, &created, e)),
    }
}

/// Let go of `locks` and remove the lock files of `created`, then hand back
/// `refusal`: what a refused open leaves behind is what it found.
fn release_created(
    locks: DirectoryLocks,
    created: &[PathBuf],
    refusal: StorageError,
) -> StorageError {
    drop(locks);
    for path in created {
        if let Err(e) = std::fs::remove_file(path) {
            tracing::warn!(path = %path.display(), error = %e, "a refused open left its lock file");
        }
    }
    refusal
}

/// The checks [`prepare`] makes holding every directory: an unfinished
/// restore refuses the open, a published one's secondary claims go, and the
/// primary is brought to the engine format.
fn prepare_held(
    config: &StorageConfig,
    dirs: &[&Path],
    resolved: &[PathBuf],
) -> StorageResult<FormatOpen> {
    // A partial image is never served: only the restore installing it opens
    // the directories until that restore has finished, which the primary's
    // claim tells. A secondary's claim outlives it by a moment and binds the
    // directory to its whole set: removed by an open of that set, refused
    // to any other.
    let primary_durable = config
        .endpoints
        .first()
        .is_some_and(|endpoint| endpoint.durability != Durability::Volatile);
    let unfinished = match dirs.first() {
        Some(&primary) if primary_durable => read_claim(primary)?,
        _ => None,
    };
    if let (Some(claim), Some(&primary)) = (&unfinished, dirs.first()) {
        return Err(StorageError::RestoreIncomplete {
            path: primary.display().to_string(),
            image: claim.image,
        });
    }
    // The claim writes the primary's format marker after the primary's
    // claim, so a secondary claim beside a marked, unclaimed primary is one
    // a published restore left; beside an unmarked primary it is a claim cut
    // short before the primary's.
    let published = match dirs.first() {
        Some(&primary) if primary_durable => read_marker(primary)?.is_some(),
        _ => false,
    };
    for &dir in dirs {
        match read_claim(dir)? {
            None => {}
            Some(claim) if claim.endpoints == resolved && published => remove_claim(dir)?,
            Some(claim) if claim.endpoints == resolved => {
                return Err(StorageError::RestoreIncomplete {
                    path: dir.display().to_string(),
                    image: claim.image,
                });
            }
            Some(_) => {
                return Err(StorageError::RestoreTarget {
                    path: dir.display().to_string(),
                    reason: "it belongs to a restore into another set of directories".into(),
                });
            }
        }
    }
    prepare_primary(config)
}

/// [`prepare`] for the open a restore makes of the directories `claim`
/// holds: the holds pass from the claim to the engine, and the unfinished
/// restore the claim is for does not refuse it.
///
/// # Errors
///
/// [`StorageError::RestoreTarget`] when `config` names other directories
/// than the claim covers; a format refusal.
pub fn prepare_restoring(config: &StorageConfig, claim: RestoreClaim) -> StorageResult<Prepared> {
    if canonical(&durable_dirs(config))? != claim.claim.endpoints {
        return Err(StorageError::RestoreTarget {
            path: config.data_dir().display().to_string(),
            reason: "the configuration names other directories than the restore claimed".into(),
        });
    }
    Ok(Prepared {
        format: prepare_primary(config)?,
        locks: claim.locks,
    })
}

/// The format half of [`prepare`], on the primary endpoint.
fn prepare_primary(config: &StorageConfig) -> StorageResult<FormatOpen> {
    let Some(primary) = config.endpoints.first() else {
        return Ok(FormatOpen::Volatile);
    };
    if primary.durability == Durability::Volatile {
        return Ok(FormatOpen::Volatile);
    }
    prepare_dir(
        &primary.path,
        coordinode_core::version::engine_format_version(),
        MIGRATIONS,
    )
}

/// The directories of the configuration's durable endpoints, in order.
fn durable_dirs(config: &StorageConfig) -> Vec<&Path> {
    config
        .endpoints
        .iter()
        .filter(|endpoint| endpoint.durability != Durability::Volatile)
        .map(|endpoint| endpoint.path.as_path())
        .collect()
}

/// Take a restore's hold on `dir`, which exists: its lock file is created
/// only in an empty directory. A directory with content but no lock file
/// holds something no store or restore of this engine made: `Ok(None)`, and
/// nothing in it is changed.
///
/// # Errors
///
/// [`StorageError::StoreInUse`] when another holder has it.
fn hold_for_restore(dir: &Path) -> StorageResult<Option<std::fs::File>> {
    let path = dir.join(LOCK_FILE);
    let mut options = std::fs::OpenOptions::new();
    options.write(true);
    if is_empty(dir)? {
        options.create(true).truncate(false);
    }
    match options.open(&path) {
        Ok(file) => hold(dir, file).map(Some),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(io_at("open", &path, e)),
    }
}

/// Lock `file`, the lock file of `dir`, exclusively without waiting.
fn hold(dir: &Path, file: std::fs::File) -> StorageResult<std::fs::File> {
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(std::fs::TryLockError::WouldBlock) => Err(StorageError::StoreInUse {
            path: dir.display().to_string(),
        }),
        Err(std::fs::TryLockError::Error(e)) => Err(io_at("lock", &dir.join(LOCK_FILE), e)),
    }
}

/// `dirs`, which exist, resolved: what a claim records them by. Two
/// endpoints resolving to one directory would hold each other's lock and
/// store their trees on top of each other, so they are refused.
fn canonical(dirs: &[&Path]) -> StorageResult<Vec<PathBuf>> {
    let resolved: Vec<PathBuf> = dirs
        .iter()
        .map(|dir| std::fs::canonicalize(dir).map_err(|e| io_at("resolve", dir, e)))
        .collect::<StorageResult<_>>()?;
    for (i, path) in resolved.iter().enumerate() {
        if resolved[..i].contains(path) {
            return Err(StorageError::InvalidConfig(format!(
                "two endpoints name the same directory {}",
                path.display()
            )));
        }
    }
    Ok(resolved)
}

/// [`prepare`] for one directory, a target format and a set of steps.
pub fn prepare_dir(dir: &Path, runs: u32, steps: &[MigrationStep]) -> StorageResult<FormatOpen> {
    let found = match read_marker(dir)? {
        Some(found) => found,
        None if is_empty(dir)? => {
            std::fs::create_dir_all(dir).map_err(|e| io_at("create", dir, e))?;
            // Before the footprints are measured: open charges the marker
            // with the rest of the directory.
            write_marker(dir, runs, &crate::engine::footprint::Footprints::default())?;
            return Ok(FormatOpen::Created);
        }
        None => 0,
    };
    if found == runs {
        return Ok(FormatOpen::Current);
    }
    let refuse = || StorageError::UnsupportedFormat {
        path: dir.display().to_string(),
        found,
        runs,
    };
    // Format numbering starts at 1, so `runs - 1` exists.
    if found != runs - 1 {
        return Err(refuse());
    }
    let step = steps.iter().find(|s| s.from == found).ok_or_else(refuse)?;
    tracing::info!(dir = %dir.display(), from = found, to = runs, "migrating the data directory");
    (step.migrate)(dir)?;
    Ok(FormatOpen::Migrated {
        dir: dir.to_path_buf(),
        from: found,
        to: runs,
    })
}

/// The format a directory's marker names, `None` without a marker.
pub fn read_marker(dir: &Path) -> StorageResult<Option<u32>> {
    let path = dir.join(MARKER_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io_at("read", &path, e)),
    };
    text.trim().parse().map(Some).map_err(|_| {
        StorageError::Io(format!(
            "{} does not name an engine format: {text:?}",
            path.display()
        ))
    })
}

/// Record `format` as the directory's format: written aside, made durable,
/// then renamed over the marker, so a crash leaves the old marker or the new
/// one and never a torn one. The marker is charged to `footprints` before it
/// is written.
pub fn write_marker(
    dir: &Path,
    format: u32,
    footprints: &crate::engine::footprint::Footprints,
) -> StorageResult<()> {
    write_atomically(dir, MARKER_FILE, &format!("{format}\n"), footprints)
}

/// Name of the file each claimed endpoint directory holds while a restore
/// into it is unfinished.
pub const RESTORE_FILE: &str = "RESTORE_INCOMPLETE";

/// What a restore claims: the image it installs and every directory it
/// installs into, resolved, in configuration order.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Claim {
    image: u64,
    endpoints: Vec<PathBuf>,
}

impl Claim {
    fn encode(&self) -> String {
        let mut text = format!("{:016x}\n", self.image);
        for endpoint in &self.endpoints {
            text.push_str(&endpoint.to_string_lossy());
            text.push('\n');
        }
        text
    }

    fn decode(text: &str) -> Option<Self> {
        let mut lines = text.lines();
        let image = u64::from_str_radix(lines.next()?, 16).ok()?;
        Some(Self {
            image,
            endpoints: lines.map(PathBuf::from).collect(),
        })
    }
}

/// The claim of the unfinished restore into `dir`, `None` when none is.
fn read_claim(dir: &Path) -> StorageResult<Option<Claim>> {
    let path = dir.join(RESTORE_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io_at("read", &path, e)),
    };
    Claim::decode(&text).map(Some).ok_or_else(|| {
        StorageError::Io(format!(
            "{} does not name a restore: {text:?}",
            path.display()
        ))
    })
}

/// A restore's ownership of its target directories, from the claim until it
/// is passed to the engine open that installs the image
/// ([`prepare_restoring`]).
#[derive(Debug)]
pub struct RestoreClaim {
    claim: Claim,
    locks: DirectoryLocks,
}

impl RestoreClaim {
    /// The checksum of the image the claim is for.
    #[must_use]
    pub fn image(&self) -> u64 {
        self.claim.image
    }
}

/// Claim the configured directories for the restore of the image with
/// checksum `image`, before anything is installed. Every durable endpoint
/// is held from before it is checked until the returned claim (or the
/// engine it is passed to) is dropped. Each must be empty, or the
/// unfinished restore of this same image into this same set of directories
/// is taken up again. From the claim until [`finish_restore`], only the open
/// the claim is passed to opens them, so a restore cut short is never
/// served half installed.
///
/// # Errors
///
/// [`StorageError::RestoreTarget`] when the primary endpoint is volatile, a
/// directory holds a store or something else, or the unfinished restore of
/// another image or into other directories; nothing is changed then.
/// [`StorageError::StoreInUse`] when an open or a restore holds one. A
/// write failure.
pub fn begin_restore(config: &StorageConfig, image: u64) -> StorageResult<RestoreClaim> {
    begin_restore_with(config, image, &mut || Ok(()))
}

/// [`begin_restore`], calling `step` once the target is checked and after
/// each write that claims it; an error from `step` ends the claim there, as
/// a crash at that point would.
pub(crate) fn begin_restore_with(
    config: &StorageConfig,
    image: u64,
    step: &mut dyn FnMut() -> StorageResult<()>,
) -> StorageResult<RestoreClaim> {
    let refuse = |path: &Path, reason: &str| StorageError::RestoreTarget {
        path: path.display().to_string(),
        reason: reason.into(),
    };
    // The primary holds the engine format and the claim that covers the
    // rest: a configuration whose first endpoint keeps nothing has no place
    // for either.
    let primary = match config.endpoints.first() {
        Some(first) if first.durability != Durability::Volatile => first.path.as_path(),
        Some(first) => {
            return Err(refuse(
                &first.path,
                "the primary endpoint is volatile; a restore installs into durable directories",
            ));
        }
        None => return Err(refuse(Path::new(""), "no endpoint is configured")),
    };
    let dirs = durable_dirs(config);
    // Content no store or restore of this engine made is refused before any
    // directory is created; the check under the holds below decides.
    for &dir in &dirs {
        if !is_empty(dir)? && !dir.join(LOCK_FILE).exists() {
            return Err(refuse(
                dir,
                "it is not empty; a restore installs only into empty directories",
            ));
        }
    }
    for &dir in &dirs {
        if !dir.exists() {
            std::fs::create_dir_all(dir).map_err(|e| io_at("create", dir, e))?;
        }
    }
    let resolved = canonical(&dirs)?;
    let mut locks = DirectoryLocks::default();
    for &dir in &dirs {
        match hold_for_restore(dir)? {
            Some(file) => locks.0.push(file),
            None => {
                return Err(refuse(
                    dir,
                    "it is not empty; a restore installs only into empty directories",
                ));
            }
        }
    }
    let ours = Claim {
        image,
        endpoints: resolved,
    };
    // The primary's claim is written last, so it standing means every
    // directory it names was claimed: those are taken up whatever their own
    // claim or content. Without it, each directory is empty or carries this
    // claim from a cut-short one.
    match read_claim(primary)? {
        Some(found) if found == ours => {}
        Some(found) => return Err(refuse(primary, &other_claim(&found, &ours))),
        None => {
            for &dir in &dirs {
                match read_claim(dir)? {
                    Some(found) if found == ours => {}
                    Some(found) => return Err(refuse(dir, &other_claim(&found, &ours))),
                    None if is_empty(dir)? => {}
                    None => {
                        return Err(refuse(
                            dir,
                            "it is not empty; a restore installs only into empty directories",
                        ));
                    }
                }
            }
        }
    }
    step()?;
    let none = crate::engine::footprint::Footprints::default();
    let encoded = ours.encode();
    // Secondaries first, the primary last, then the format the primary's
    // store is opened in.
    for &dir in dirs.iter().skip(1).chain(std::iter::once(&primary)) {
        if read_claim(dir)?.as_ref() != Some(&ours) {
            write_atomically(dir, RESTORE_FILE, &encoded, &none)?;
            step()?;
        }
    }
    if read_marker(primary)?.is_none() {
        write_marker(
            primary,
            coordinode_core::version::engine_format_version(),
            &none,
        )?;
        step()?;
    }
    Ok(RestoreClaim { claim: ours, locks })
}

/// Why `found` keeps a restore claiming `ours` out.
fn other_claim(found: &Claim, ours: &Claim) -> String {
    if found.image == ours.image {
        "it holds an unfinished restore into another set of directories; run that one again, \
         or empty the directories"
            .into()
    } else {
        format!(
            "it holds the unfinished restore of image {:#018x}, not of {:#018x}; finish that \
             one, or empty the directories",
            found.image, ours.image
        )
    }
}

/// End the restore into `config`'s directories, once its image is installed
/// and durable, while the engine the claim was passed to still holds them:
/// from then on the store opens as usual. The primary's claim goes first:
/// a stop before it leaves a restore that is taken up again, a stop after it
/// a published store whose secondaries stay bound to it.
///
/// # Errors
///
/// A claim cannot be removed.
pub fn finish_restore(config: &StorageConfig) -> StorageResult<()> {
    finish_restore_with(config, &mut || Ok(()))
}

/// [`finish_restore`], calling `step` after each claim it removes; an error
/// from `step` ends it there, as a crash at that point would.
pub(crate) fn finish_restore_with(
    config: &StorageConfig,
    step: &mut dyn FnMut() -> StorageResult<()>,
) -> StorageResult<()> {
    let dirs = durable_dirs(config);
    // The primary's claim first: its removal is the one point the store is
    // published at. The secondaries' claims left after it still name their
    // whole set, so no other store takes one of them, and an open of this
    // configuration removes them.
    for &dir in dirs.first().into_iter().chain(dirs.iter().skip(1)) {
        remove_claim(dir)?;
        step()?;
    }
    Ok(())
}

/// Remove the claim in `dir`, durably; none there is nothing to do.
fn remove_claim(dir: &Path) -> StorageResult<()> {
    let path = dir.join(RESTORE_FILE);
    match std::fs::remove_file(&path) {
        Ok(()) => sync_dir(dir),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(io_at("remove", &path, e)),
    }
}

/// Write `text` to `name` in `dir`: written aside, made durable, then
/// renamed over the file, so a crash leaves the old content or the new and
/// never a torn one. Charged to `footprints` before it is written.
fn write_atomically(
    dir: &Path,
    name: &str,
    text: &str,
    footprints: &crate::engine::footprint::Footprints,
) -> StorageResult<()> {
    use std::io::Write;
    let path = dir.join(name);
    let staged: PathBuf = dir.join(format!("{name}.tmp"));
    if let Some(charge) = footprints.file(&staged) {
        charge.charge_to(text.len() as u64)?;
    }
    let mut file = std::fs::File::create(&staged).map_err(|e| io_at("create", &staged, e))?;
    file.write_all(text.as_bytes())
        .and_then(|()| file.sync_all())
        .map_err(|e| io_at("write", &staged, e))?;
    drop(file);
    std::fs::rename(&staged, &path).map_err(|e| io_at("rename", &path, e))?;
    // Within one directory: one endpoint, nothing to refuse.
    footprints
        .renamed(&staged, &path)
        .map_err(|e| io_at("charge", &path, e))?;
    sync_dir(dir)
}

/// Make a rename in `dir` durable. Windows has no directory handle to sync;
/// NTFS journals the rename itself.
fn sync_dir(dir: &Path) -> StorageResult<()> {
    #[cfg(unix)]
    {
        std::fs::File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(|e| io_at("sync", dir, e))?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// Whether `dir` holds nothing but what taking it leaves: its lock file and
/// a claim written aside and never renamed in.
fn is_empty(dir: &Path) -> StorageResult<bool> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(true),
        Err(e) => return Err(io_at("list", dir, e)),
    };
    let staged_claim = format!("{RESTORE_FILE}.tmp");
    for entry in entries {
        let name = entry.map_err(|e| io_at("list", dir, e))?.file_name();
        if name != LOCK_FILE && name.to_str() != Some(staged_claim.as_str()) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn io_at(what: &str, path: &Path, e: std::io::Error) -> StorageError {
    StorageError::Io(format!("{what} {}: {e}", path.display()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
