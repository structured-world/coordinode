//! This member's second copy of its consensus promise, and the incarnation
//! that binds it to this store.
//!
//! The promise (the consensus engine's vote) lives in the Raft tree through
//! the member-local journal. A damaged tree loses it, and a member that forgot
//! its vote can grant a second one in the same term. So the promise is kept
//! twice: in the tree, and in a pair of small files beside the consensus log
//! that no tree repair, compaction, checkpoint or journal purge touches.
//!
//! The files are numbered slots, written in place and never renamed. A
//! record carries a sequence number and a checksum; the newest whole record
//! is the copy. What a slot reads after a failed sync may never have reached
//! the disk, so a write is placed by what this process proved durable, never
//! by what the slots read. A process that has proved nothing yet (it just
//! opened) writes a new slot and leaves every existing one alone, since any
//! of them may be the only durable record. Once a write has succeeded, the
//! slots beyond the one holding it and one more are removed, and the next
//! write goes to that other one. The directory is synced until a sync in
//! this process has made the slot's name durable. Each record carries the
//! incarnation of the
//! store it belongs to, so a file brought from another store is never taken
//! for this member's promise.
//!
//! The incarnation is drawn once, when the store is first bound to a
//! consensus log, and recorded in the log's marker file. Every file is
//! charged to the endpoint's footprint.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use lsm_tree::fs::{Fs, FsOpenOptions};

use crate::engine::footprint::Footprints;
use crate::error::{StorageError, StorageResult};

/// The name the second copy's slots share, in the root of the endpoint that
/// holds the consensus log: `<name>.<number>`.
pub const PROMISE_FILE: &str = "consensus-vote";

/// Identifies one store bound to a consensus log.
pub type Incarnation = [u8; 16];

const MAGIC: &[u8; 8] = b"CNPROM02";
/// Magic, incarnation, sequence number, payload length.
const HEADER: usize = MAGIC.len() + 16 + 8 + 4;
const CHECKSUM: usize = 4;

fn slot_path(dir: &Path, slot: u32) -> PathBuf {
    dir.join(format!("{PROMISE_FILE}.{slot}"))
}

/// The slot files of the second copy in `dir`, by number.
///
/// # Errors
///
/// `dir` exists and cannot be listed.
pub fn copy_files(fs: &dyn Fs, dir: &Path) -> StorageResult<Vec<(u32, PathBuf)>> {
    let entries = match fs.read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == lsm_tree::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(io_at("list", dir, e.into())),
    };
    let mut slots: Vec<(u32, PathBuf)> = entries
        .into_iter()
        .filter(|entry| !entry.is_dir)
        .filter_map(|entry| {
            let slot = entry
                .file_name
                .strip_prefix(PROMISE_FILE)?
                .strip_prefix('.')?
                .parse()
                .ok()?;
            Some((slot, entry.path))
        })
        .collect();
    slots.sort_unstable_by_key(|(slot, _)| *slot);
    Ok(slots)
}

/// What one process proved about the copy's slots: kept across its writes,
/// empty when the store opens.
#[derive(Debug, Default)]
pub struct CopyWriter {
    /// The last record this process wrote durably, and its slot.
    durable: Option<(u32, Incarnation, Vec<u8>)>,
    /// The slot a write failed on, written again by the next attempt.
    failed: Option<u32>,
    /// Slots whose names a directory sync in this process made durable.
    named: Vec<u32>,
}

impl CopyWriter {
    /// Whether the newest record is `payload` for `incarnation`, written
    /// durably by this process with no failed write since.
    #[must_use]
    pub fn holds(&self, incarnation: &Incarnation, payload: &[u8]) -> bool {
        self.failed.is_none()
            && self
                .durable
                .as_ref()
                .is_some_and(|(_, held, written)| held == incarnation && written == payload)
    }

    /// The slot the next record goes to, given what each slot reads.
    fn target(&self, slots: &[(u32, Slot)]) -> u32 {
        // Never the durable record's slot, so any slot is safe; the same one
        // keeps a process whose writes keep failing at one new file.
        if let Some(failed) = self.failed {
            return failed;
        }
        if let Some((durable, _, _)) = self.durable {
            // Oldest first: a slot without a whole record, then the lowest
            // sequence.
            if let Some((slot, _)) = slots
                .iter()
                .filter(|(slot, _)| *slot != durable)
                .min_by_key(|(_, read)| read.sequence().map_or(0, |n| n + 1))
            {
                return *slot;
            }
        }
        // A new slot: the lowest number no file has. Slot numbers stay a few
        // above the count of files, far below u32::MAX.
        let mut free = 0;
        for (slot, _) in slots {
            if *slot != free {
                break;
            }
            free += 1;
        }
        free
    }
}

/// One slot file as read.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Slot {
    /// A whole record.
    Record {
        sequence: u64,
        incarnation: Incarnation,
        payload: Vec<u8>,
    },
    /// A file that holds no whole record: torn, damaged or of another layout.
    Damaged(String),
}

impl Slot {
    fn sequence(&self) -> Option<u64> {
        match self {
            Self::Record { sequence, .. } => Some(*sequence),
            Self::Damaged(_) => None,
        }
    }
}

/// What the second copy holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromiseCopy {
    /// No file: never written, or lost.
    Absent,
    /// An intact copy written for `incarnation`; `payload` is the consensus
    /// engine's encoding of the promise.
    Held {
        incarnation: Incarnation,
        payload: Vec<u8>,
    },
    /// A file that cannot be this store's copy: torn, damaged, or of an
    /// unknown layout.
    Damaged(String),
}

/// Read the copy in `dir`.
///
/// # Errors
///
/// The file exists and cannot be read; a damaged file is not an error but
/// [`PromiseCopy::Damaged`].
pub fn read_copy(fs: &dyn Fs, dir: &Path) -> StorageResult<PromiseCopy> {
    let slots = read_slots(fs, dir)?;
    let newest = slots
        .iter()
        .filter_map(|(_, slot)| match slot {
            Slot::Record {
                sequence,
                incarnation,
                payload,
            } => Some((*sequence, incarnation, payload)),
            Slot::Damaged(_) => None,
        })
        .max_by_key(|(sequence, _, _)| *sequence);
    Ok(match newest {
        Some((_, incarnation, payload)) => PromiseCopy::Held {
            incarnation: *incarnation,
            payload: payload.clone(),
        },
        None => match slots.iter().find_map(|(_, slot)| match slot {
            Slot::Damaged(why) => Some(why.clone()),
            Slot::Record { .. } => None,
        }) {
            Some(why) => PromiseCopy::Damaged(why),
            None => PromiseCopy::Absent,
        },
    })
}

/// Every slot file in `dir`, by number, as read. A file removed between
/// listing and reading is left out.
fn read_slots(fs: &dyn Fs, dir: &Path) -> StorageResult<Vec<(u32, Slot)>> {
    let mut slots = Vec::new();
    for (slot, path) in copy_files(fs, dir)? {
        if let Some(bytes) = read_file(fs, &path)? {
            slots.push((slot, decode(&bytes)));
        }
    }
    Ok(slots)
}

/// The whole of the file at `path`, `None` when there is none.
pub(crate) fn read_file(fs: &dyn Fs, path: &Path) -> StorageResult<Option<Vec<u8>>> {
    let mut file = match fs.open(path, &FsOpenOptions::new().read(true)) {
        Ok(file) => file,
        Err(e) if e.kind() == lsm_tree::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io_at("open", path, e.into())),
    };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| io_at("read", path, e))?;
    Ok(Some(bytes))
}

fn decode(bytes: &[u8]) -> Slot {
    let damaged = |why: &str| Slot::Damaged(why.to_string());
    if bytes.len() < HEADER + CHECKSUM {
        return damaged("shorter than its header");
    }
    if &bytes[..MAGIC.len()] != MAGIC {
        return damaged("not a promise copy");
    }
    let (body, sum) = bytes.split_at(bytes.len() - CHECKSUM);
    let mut stored = [0u8; CHECKSUM];
    stored.copy_from_slice(sum);
    if crc32c::crc32c(body) != u32::from_le_bytes(stored) {
        return damaged("checksum mismatch");
    }
    let at = MAGIC.len();
    let mut incarnation = [0u8; 16];
    incarnation.copy_from_slice(&body[at..at + 16]);
    let mut sequence = [0u8; 8];
    sequence.copy_from_slice(&body[at + 16..at + 24]);
    let mut len = [0u8; 4];
    len.copy_from_slice(&body[at + 24..HEADER]);
    let payload = &body[HEADER..];
    if usize::try_from(u32::from_le_bytes(len)).ok() != Some(payload.len()) {
        return damaged("length does not match its payload");
    }
    Slot::Record {
        sequence: u64::from_le_bytes(sequence),
        incarnation,
        payload: payload.to_vec(),
    }
}

/// Write `payload` for `incarnation` as the newest record of the copy in
/// `dir`, through `writer`, which keeps what this process proved about the
/// slots between writes: durable when this returns. The slot of the last
/// durable record is never written, so a write cut short, at any point,
/// leaves it.
///
/// # Errors
///
/// The endpoint has no room for the file, or a write or sync fails. The
/// last durable record then stays the copy until a later write succeeds.
pub fn write_copy(
    fs: &dyn Fs,
    dir: &Path,
    writer: &mut CopyWriter,
    incarnation: &Incarnation,
    payload: &[u8],
    footprints: &Footprints,
) -> StorageResult<()> {
    let len = u32::try_from(payload.len()).map_err(|_| {
        StorageError::InvalidConfig(format!(
            "a consensus promise of {} bytes does not fit its copy",
            payload.len()
        ))
    })?;
    let slots = read_slots(fs, dir)?;
    let target = writer.target(&slots);
    // Above every record any slot reads, so the new one is the copy once
    // whole. Sequence numbers count writes of one member's copy: far below
    // u64::MAX.
    let sequence = slots
        .iter()
        .filter_map(|(_, read)| read.sequence())
        .max()
        .map_or(0, |n| n + 1);
    let mut bytes = Vec::with_capacity(HEADER + payload.len() + CHECKSUM);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(incarnation);
    bytes.extend_from_slice(&sequence.to_le_bytes());
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes.extend_from_slice(payload);
    let sum = crc32c::crc32c(&bytes);
    bytes.extend_from_slice(&sum.to_le_bytes());

    let path = &slot_path(dir, target);
    if let Some(charge) = footprints.file(path) {
        charge.charge_to(bytes.len() as u64)?;
    }
    writer.failed = Some(target);
    let mut file = fs
        .open(
            path,
            &FsOpenOptions::new().write(true).create(true).truncate(true),
        )
        .map_err(|e| io_at("open", path, e.into()))?;
    file.write_all(&bytes)
        .map_err(|e| io_at("write", path, e))?;
    file.sync_all().map_err(|e| io_at("sync", path, e.into()))?;
    drop(file);
    // A slot is durable only once its name is, and a name this process did
    // not sync may be one an earlier, failed write created.
    if !writer.named.contains(&target) {
        fs.sync_directory(dir)
            .map_err(|e| io_at("sync", dir, e.into()))?;
        writer.named = slots.iter().map(|(slot, _)| *slot).collect();
        writer.named.push(target);
    }
    writer.failed = None;
    writer.durable = Some((target, *incarnation, payload.to_vec()));
    remove_spent_slots(fs, dir, writer, target, &slots, footprints);
    Ok(())
}

/// Remove the slots beyond the durable record's and one to write next: every
/// record in them is older than the durable one. A slot that cannot be
/// removed stays and is removed after a later write.
fn remove_spent_slots(
    fs: &dyn Fs,
    dir: &Path,
    writer: &mut CopyWriter,
    durable: u32,
    slots: &[(u32, Slot)],
    footprints: &Footprints,
) {
    let others: Vec<u32> = slots
        .iter()
        .map(|(slot, _)| *slot)
        .filter(|slot| *slot != durable)
        .collect();
    // The one kept to write next: a slot whose name is durable, so the next
    // write needs no directory sync.
    let Some(next) = others
        .iter()
        .copied()
        .find(|slot| writer.named.contains(slot))
        .or_else(|| others.first().copied())
    else {
        return;
    };
    for slot in others.into_iter().filter(|slot| *slot != next) {
        let path = slot_path(dir, slot);
        match fs.remove_file(&path) {
            Ok(()) => {
                footprints.removed(&path);
                writer.named.retain(|named| *named != slot);
            }
            Err(e) => tracing::warn!(
                path = %path.display(),
                error = %e,
                "a spent slot of the consensus promise copy stays until a later write"
            ),
        }
    }
}

/// Write `bytes` aside, make them durable, rename them over `name` in `dir`
/// and make the rename durable, so a crash leaves the old file or the new one
/// and never a torn one. Charged to `footprints` before anything is written.
pub(crate) fn replace_file(
    fs: &dyn Fs,
    dir: &Path,
    name: &str,
    bytes: &[u8],
    footprints: &Footprints,
) -> StorageResult<()> {
    let path = dir.join(name);
    let staged = dir.join(format!("{name}.tmp"));
    if let Some(charge) = footprints.file(&staged) {
        charge.charge_to(bytes.len() as u64)?;
    }
    let mut file = fs
        .open(
            &staged,
            &FsOpenOptions::new().write(true).create(true).truncate(true),
        )
        .map_err(|e| io_at("create", &staged, e.into()))?;
    file.write_all(bytes)
        .map_err(|e| io_at("write", &staged, e))?;
    file.sync_all()
        .map_err(|e| io_at("sync", &staged, e.into()))?;
    drop(file);
    fs.rename(&staged, &path)
        .map_err(|e| io_at("rename", &path, e.into()))?;
    // Within one directory: one endpoint, nothing to refuse.
    footprints
        .renamed(&staged, &path)
        .map_err(|e| io_at("charge", &path, e))?;
    fs.sync_directory(dir)
        .map_err(|e| io_at("sync", dir, e.into()))
}

/// A fresh incarnation for a store first bound to a consensus log.
pub fn new_incarnation() -> Incarnation {
    use std::hash::{BuildHasher, Hasher};
    // Each RandomState is keyed from the operating system's randomness; the
    // incarnation only has to differ from every other store's, not resist an
    // adversary, who could copy the file whole anyway.
    let half = |salt: u64| {
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_u64(salt);
        hasher.finish().to_le_bytes()
    };
    let mut incarnation = [0u8; 16];
    incarnation[..8].copy_from_slice(&half(0));
    incarnation[8..].copy_from_slice(&half(1));
    incarnation
}

/// The incarnation `text` names in a marker, `None` for a marker written
/// before markers carried one (an empty file).
pub(crate) fn parse_incarnation(text: &str) -> StorageResult<Option<Incarnation>> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    let invalid = || StorageError::Io(format!("the consensus log marker holds {text:?}"));
    // Checked before any slicing: a byte index into text that is not all
    // ASCII can fall inside a character.
    if text.len() != 32 || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(invalid());
    }
    let mut incarnation = [0u8; 16];
    for (i, byte) in incarnation.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).map_err(|_| invalid())?;
    }
    Ok(Some(incarnation))
}

/// How a marker records `incarnation`.
pub(crate) fn format_incarnation(incarnation: &Incarnation) -> String {
    let mut text = String::with_capacity(33);
    for byte in incarnation {
        text.push_str(&format!("{byte:02x}"));
    }
    text.push('\n');
    text
}

fn io_at(what: &str, path: &Path, e: std::io::Error) -> StorageError {
    StorageError::Io(format!("{what} {}: {e}", path.display()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
