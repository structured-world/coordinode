//! This member's second copy of its consensus promise, and the incarnation
//! that binds it to this store.
//!
//! The promise (the consensus engine's vote) lives in the Raft tree through
//! the member-local journal. A damaged tree loses it, and a member that forgot
//! its vote can grant a second one in the same term. So the promise is kept
//! twice: in the tree, and in a small file beside the consensus log that no
//! tree repair, compaction, checkpoint or journal purge touches. The file is
//! replaced atomically (written aside, synced, renamed, directory synced) and
//! carries a checksum and the incarnation of the store it belongs to, so a
//! torn write, a damaged block or a file brought from another store is never
//! taken for this member's promise.
//!
//! The incarnation is drawn once, when the store is first bound to a
//! consensus log, and recorded in the log's marker file. Both files are
//! charged to the endpoint's footprint.

use std::io::Write;
use std::path::Path;

use crate::engine::footprint::Footprints;
use crate::error::{StorageError, StorageResult};

/// The file that holds the second copy, in the root of the endpoint that
/// holds the consensus log.
pub const PROMISE_FILE: &str = "consensus-vote";

/// Identifies one store bound to a consensus log.
pub type Incarnation = [u8; 16];

const MAGIC: &[u8; 8] = b"CNPROM01";
/// Magic, incarnation, payload length.
const HEADER: usize = MAGIC.len() + 16 + 4;
const CHECKSUM: usize = 4;

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
pub fn read_copy(dir: &Path) -> StorageResult<PromiseCopy> {
    let path = dir.join(PROMISE_FILE);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(PromiseCopy::Absent),
        Err(e) => return Err(io_at("read", &path, e)),
    };
    Ok(decode(&bytes))
}

fn decode(bytes: &[u8]) -> PromiseCopy {
    let damaged = |why: &str| PromiseCopy::Damaged(why.to_string());
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
    let mut incarnation = [0u8; 16];
    incarnation.copy_from_slice(&body[MAGIC.len()..MAGIC.len() + 16]);
    let mut len = [0u8; 4];
    len.copy_from_slice(&body[MAGIC.len() + 16..HEADER]);
    let payload = &body[HEADER..];
    if usize::try_from(u32::from_le_bytes(len)).ok() != Some(payload.len()) {
        return damaged("length does not match its payload");
    }
    PromiseCopy::Held {
        incarnation,
        payload: payload.to_vec(),
    }
}

/// Replace the copy in `dir` with `payload` for `incarnation`: durable,
/// whole or not at all, when this returns.
///
/// # Errors
///
/// The endpoint has no room for the file, or a write, sync or rename fails.
/// The previous copy is then still in place.
pub fn write_copy(
    dir: &Path,
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
    let mut bytes = Vec::with_capacity(HEADER + payload.len() + CHECKSUM);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(incarnation);
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes.extend_from_slice(payload);
    let sum = crc32c::crc32c(&bytes);
    bytes.extend_from_slice(&sum.to_le_bytes());
    replace_file(dir, PROMISE_FILE, &bytes, footprints)
}

/// Write `bytes` aside, make them durable, rename them over `name` in `dir`
/// and make the rename durable, so a crash leaves the old file or the new one
/// and never a torn one. Charged to `footprints` before anything is written.
pub(crate) fn replace_file(
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
    let mut file = std::fs::File::create(&staged).map_err(|e| io_at("create", &staged, e))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| io_at("write", &staged, e))?;
    drop(file);
    std::fs::rename(&staged, &path).map_err(|e| io_at("rename", &path, e))?;
    // Within one directory: one endpoint, nothing to refuse.
    footprints
        .renamed(&staged, &path)
        .map_err(|e| io_at("charge", &path, e))?;
    lsm_tree::fs::Fs::sync_directory(&lsm_tree::fs::StdFs, dir)
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
    if text.len() != 32 {
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
