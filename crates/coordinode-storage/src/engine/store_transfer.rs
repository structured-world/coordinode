//! A copy of the whole store for another member, made from pinned versions of
//! its trees.
//!
//! [`StorageEngine::pin_store`] pins every replicated tree (each partition but
//! the node-local Raft one, and every `STORAGE COLUMNAR` table) while no Raft
//! entry applies, and returns a [`StoreTransfer`]: a byte stream of known
//! length that reads the pinned files in place as it is consumed. Nothing is
//! copied, linked or synced to take it, so its cost does not grow with the
//! store, and the files stay unchanged until the transfer drops.
//!
//! [`receive_store`] reads such a stream into a directory, rebuilds each tree
//! there from its files and its memtable contents, and leaves a directory that
//! opens with [`StorageEngine::open_checkpoint`]. Installing it into a live
//! store is the receiver's business; the engine's own install reads it as rows.
//!
//! The stream, all integers little-endian:
//!
//! ```text
//! [magic "CNST"][version u8]
//! [tree_count u32]
//! tree*:
//!   [name: u16 len + UTF-8]            "<partition>" or "tables/<table id>"
//!   [position: u8 flag (+ next u64, payload u32 len + bytes)]
//!   [layout: u32 len + bytes]
//!   [file_count u32]
//!   file*: [kind u8][id u64][attestation: u8 flag (+ u32 len + bytes)][len u64][bytes]
//!   [entry_count u64]
//!   entry*: [key u32 len + bytes][value u32 len + bytes][seqno u64][value type u8]
//!   [range_tombstone_count u32]
//!   tombstone*: [start u32 len + bytes][end u32 len + bytes][seqno u64]
//! [inventory: u8 flag (+ u32 len + bytes)]
//! [checksum u64: xxh3 of every byte before it]
//! ```
//!
//! A tree's position is present only for a partition installed from a peer
//! further along the Raft log than the applies: it holds every entry below
//! that position, not only the ones the transfer's own position names.

use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use lsm_tree::{AbstractTree, PinnedEntry, PinnedFile, PinnedRangeTombstone, PinnedVersion};
use xxhash_rust::xxh3::Xxh3;

use crate::engine::core::{RaftApplyState, RaftPosition, StorageEngine};
use crate::engine::coverage::{Domain, TreeCoverage};
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// What a transfer stream starts with.
const MAGIC: [u8; 4] = *b"CNST";
/// The stream's shape after its magic; a receiver refuses any other.
const VERSION: u8 = 1;
/// The name prefix of a `STORAGE COLUMNAR` table's tree.
const TABLE_PREFIX: &str = "tables/";

/// Which folder of its tree a pinned file belongs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FileKind {
    Table,
    Blob,
    Dictionary,
}

impl FileKind {
    fn tag(self) -> u8 {
        match self {
            Self::Table => 0,
            Self::Blob => 1,
            Self::Dictionary => 2,
        }
    }

    fn from_tag(tag: u8) -> StorageResult<Self> {
        match tag {
            0 => Ok(Self::Table),
            1 => Ok(Self::Blob),
            2 => Ok(Self::Dictionary),
            other => Err(malformed(format!("file kind {other}"))),
        }
    }

    /// The folder `Config::restore_pinned` expects the file in.
    fn folder(self) -> &'static str {
        match self {
            Self::Table => "tables",
            Self::Blob => "blobs",
            Self::Dictionary => "dicts",
        }
    }
}

/// Which tree of the store a pin is of.
#[derive(Clone, Debug, PartialEq, Eq)]
enum TreeName {
    Partition(Partition),
    Table(String),
}

impl TreeName {
    fn encode(&self) -> String {
        match self {
            Self::Partition(part) => part.name().to_owned(),
            Self::Table(id) => format!("{TABLE_PREFIX}{id}"),
        }
    }

    /// The tree a received name stands for. A table id must be one path
    /// component: the stream decides where its files land.
    fn decode(name: &str) -> StorageResult<Self> {
        if let Some(id) = name.strip_prefix(TABLE_PREFIX) {
            let valid =
                !id.is_empty() && id != "." && id != ".." && !id.contains(['/', '\\', '\0']);
            return if valid {
                Ok(Self::Table(id.to_owned()))
            } else {
                Err(malformed(format!("table name {name:?}")))
            };
        }
        Partition::all()
            .iter()
            .copied()
            .find(|part| *part != Partition::Raft && part.name() == name)
            .map(Self::Partition)
            .ok_or_else(|| malformed(format!("tree name {name:?}")))
    }

    /// Where the tree lies in a directory laid out as a checkpoint.
    fn dir(&self, root: &Path) -> PathBuf {
        match self {
            Self::Partition(part) => root.join(part.name()),
            Self::Table(id) => root.join("tables").join(id),
        }
    }
}

/// One pinned tree of a transfer.
struct PinnedTree {
    name: TreeName,
    /// Where the tree stands when it is past the transfer's position.
    position: Option<RaftPosition>,
    pin: PinnedVersion,
}

impl PinnedTree {
    fn files(&self) -> impl Iterator<Item = (FileKind, &PinnedFile)> {
        let pin = &self.pin;
        pin.tables()
            .iter()
            .map(|f| (FileKind::Table, f))
            .chain(pin.blob_files().iter().map(|f| (FileKind::Blob, f)))
            .chain(pin.dictionaries().iter().map(|f| (FileKind::Dictionary, f)))
    }
}

/// One piece of the stream, in order.
enum Part {
    /// Bytes encoded when the transfer was taken.
    Bytes(Vec<u8>),
    /// A pinned file, `len` bytes, read when reached.
    File { tree: usize, file: usize, len: u64 },
    /// A tree's memtable entries, `len` bytes once encoded, encoded when
    /// reached: they are held once, by the pin, until then.
    Entries { tree: usize, len: u64 },
    /// The checksum of everything before it.
    Checksum,
}

/// What the stream is handing out now.
enum Cursor {
    Buffer {
        bytes: Vec<u8>,
        at: usize,
        hashed: bool,
    },
    File {
        file: Box<dyn lsm_tree::fs::FsFile>,
        left: u64,
    },
}

/// The whole replicated store, pinned, as one byte stream of
/// [`Self::len`] bytes; see the [module docs](self) for its shape.
///
/// The pinned files stay on disk unchanged, whatever flushes and compactions
/// do meanwhile, until this drops. Reads block on file I/O.
pub struct StoreTransfer {
    trees: Vec<PinnedTree>,
    parts: Vec<Part>,
    next_part: usize,
    cursor: Option<Cursor>,
    /// Boxed: its state is most of the transfer's size, which moves around
    /// with the snapshot that holds it.
    hasher: Box<Xxh3>,
    len: u64,
}

impl std::fmt::Debug for StoreTransfer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoreTransfer")
            .field("trees", &self.trees.len())
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl StoreTransfer {
    fn new(trees: Vec<PinnedTree>, inventory: Option<Vec<u8>>) -> StorageResult<Self> {
        let mut parts = Vec::new();
        let mut head = Vec::new();
        head.extend_from_slice(&MAGIC);
        head.push(VERSION);
        put_u32(&mut head, trees.len())?;
        for (t, tree) in trees.iter().enumerate() {
            let name = tree.name.encode();
            put_u16(&mut head, name.len())?;
            head.extend_from_slice(name.as_bytes());
            match &tree.position {
                Some(position) => {
                    head.push(1);
                    head.extend_from_slice(&position.next.to_le_bytes());
                    put_bytes(&mut head, &position.payload)?;
                }
                None => head.push(0),
            }
            put_bytes(&mut head, &tree.pin.layout()?)?;
            let files: Vec<(FileKind, &PinnedFile)> = tree.files().collect();
            put_u32(&mut head, files.len())?;
            for (f, (kind, file)) in files.into_iter().enumerate() {
                head.push(kind.tag());
                head.extend_from_slice(&file.id().to_le_bytes());
                match file.heal_attestation()? {
                    Some(path) => {
                        head.push(1);
                        put_bytes(&mut head, &read_whole(file, &path)?)?;
                    }
                    None => head.push(0),
                }
                let len = file.file_size()?;
                head.extend_from_slice(&len.to_le_bytes());
                parts.push(Part::Bytes(std::mem::take(&mut head)));
                parts.push(Part::File {
                    tree: t,
                    file: f,
                    len,
                });
            }
            let len = entries_len(tree.pin.memtable_entries())?;
            parts.push(Part::Bytes(std::mem::take(&mut head)));
            parts.push(Part::Entries { tree: t, len });
            let tombstones = tree.pin.memtable_range_tombstones();
            put_u32(&mut head, tombstones.len())?;
            for tombstone in &tombstones {
                put_bytes(&mut head, &tombstone.start)?;
                put_bytes(&mut head, &tombstone.end)?;
                head.extend_from_slice(&tombstone.seqno.to_le_bytes());
            }
        }
        match &inventory {
            Some(bytes) => {
                head.push(1);
                put_bytes(&mut head, bytes)?;
            }
            None => head.push(0),
        }
        parts.push(Part::Bytes(head));
        parts.push(Part::Checksum);

        let len = parts
            .iter()
            .map(|part| match part {
                Part::Bytes(bytes) => bytes.len() as u64,
                Part::File { len, .. } | Part::Entries { len, .. } => *len,
                Part::Checksum => 8,
            })
            .sum();
        Ok(Self {
            trees,
            parts,
            next_part: 0,
            cursor: None,
            hasher: Box::new(Xxh3::new()),
            len,
        })
    }

    /// The stream's length in bytes.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the stream is empty; it never is, it carries at least its
    /// header and checksum.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Start the next part of the stream; `false` once there is none.
    fn advance(&mut self) -> io::Result<bool> {
        let Some(part) = self.parts.get_mut(self.next_part) else {
            return Ok(false);
        };
        self.next_part += 1;
        self.cursor = Some(match part {
            Part::Bytes(bytes) => Cursor::Buffer {
                bytes: std::mem::take(bytes),
                at: 0,
                hashed: true,
            },
            Part::File { tree, file, len } => {
                let pinned = self
                    .trees
                    .get(*tree)
                    .and_then(|t| t.files().nth(*file))
                    .map(|(_, f)| f)
                    .ok_or_else(|| io::Error::other("a transfer part names no pinned file"))?;
                Cursor::File {
                    file: pinned.open().map_err(io::Error::other)?,
                    left: *len,
                }
            }
            Part::Entries { tree, len } => {
                let pin = &self
                    .trees
                    .get(*tree)
                    .ok_or_else(|| io::Error::other("a transfer part names no pinned tree"))?
                    .pin;
                let bytes = encode_entries(pin.memtable_entries())?;
                if bytes.len() as u64 != *len {
                    return Err(io::Error::other(
                        "a pinned memtable changed between its sizing and its encoding",
                    ));
                }
                Cursor::Buffer {
                    bytes,
                    at: 0,
                    hashed: true,
                }
            }
            Part::Checksum => Cursor::Buffer {
                bytes: self.hasher.digest().to_le_bytes().to_vec(),
                at: 0,
                hashed: false,
            },
        });
        Ok(true)
    }
}

impl Read for StoreTransfer {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        loop {
            match &mut self.cursor {
                Some(Cursor::Buffer { bytes, at, hashed }) if *at < bytes.len() => {
                    let n = buf.len().min(bytes.len() - *at);
                    let chunk = &bytes[*at..*at + n];
                    buf[..n].copy_from_slice(chunk);
                    if *hashed {
                        self.hasher.update(chunk);
                    }
                    *at += n;
                    return Ok(n);
                }
                Some(Cursor::File { file, left }) if *left > 0 => {
                    // Below `buf.len()` whenever it is the smaller, so a usize.
                    let want = if *left < buf.len() as u64 {
                        *left as usize
                    } else {
                        buf.len()
                    };
                    let n = file.read(&mut buf[..want])?;
                    if n == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "a pinned file is shorter than when it was pinned",
                        ));
                    }
                    self.hasher.update(&buf[..n]);
                    *left -= n as u64;
                    return Ok(n);
                }
                _ => {
                    if !self.advance()? {
                        return Ok(0);
                    }
                }
            }
        }
    }
}

impl StorageEngine {
    /// Pin the whole replicated store for a transfer to another member:
    /// every partition but the node-local Raft one, and every `STORAGE
    /// COLUMNAR` table, with the inventory of its index copies.
    ///
    /// On a store a Raft state machine runs over, the trees are pinned with
    /// the applies paused, so they hold exactly the applied entries, and a
    /// partition installed from a peer further along carries its own
    /// position. Call it where the caller knows the applies' position (the
    /// state machine, between two applies); it costs no file I/O on the
    /// partition trees.
    ///
    /// Blocks; call it off the async runtime.
    ///
    /// # Errors
    ///
    /// A tree, a coverage record or the index inventory cannot be read, or a
    /// header field exceeds the stream's limits.
    pub fn pin_store(&self) -> StorageResult<StoreTransfer> {
        let (trees, inventory) = match self.raft_fence() {
            Some(fence) => {
                let mut pinned = None;
                fence.with_applies_paused(&mut |applies, _| {
                    pinned = Some(self.pin_trees(Some(applies))?);
                    Ok(())
                })?;
                pinned.ok_or_else(|| {
                    StorageError::InvalidConfig("the Raft apply fence never ran the pin".into())
                })?
            }
            None => self.pin_trees(None)?,
        };
        StoreTransfer::new(trees, inventory)
    }

    /// Pin every replicated tree and take the sealed index inventory, with
    /// the applies at `applies` when a Raft state machine runs.
    fn pin_trees(
        &self,
        applies: Option<&RaftApplyState>,
    ) -> StorageResult<(Vec<PinnedTree>, Option<Vec<u8>>)> {
        let mut trees = Vec::with_capacity(Partition::all().len());
        for &part in Partition::all() {
            if part == Partition::Raft {
                continue;
            }
            let tree = self.tree(part)?;
            let position = match applies {
                Some(applies) if applies.floor(part) > applies.next() => {
                    TreeCoverage::read(tree, Domain::Raft)?
                        .base()
                        .map(|(next, payload)| RaftPosition {
                            next,
                            payload: payload.to_vec(),
                        })
                }
                _ => None,
            };
            trees.push(PinnedTree {
                name: TreeName::Partition(part),
                position,
                pin: tree.pin(),
            });
        }
        #[cfg(feature = "columnar")]
        for (table_id, tree) in self.columnar_table_trees() {
            trees.push(PinnedTree {
                name: TreeName::Table(table_id),
                position: None,
                pin: tree.pin(),
            });
        }
        // Taken while nothing applies, as the trees are, and sealed at once:
        // the receiver keeps the evidence of what this member proved.
        let inventory = match self.capture_index_inventory()? {
            Some(captured) => self.seal_index_inventory(&captured)?,
            None => None,
        };
        Ok((trees, inventory))
    }
}

/// A store received with [`receive_store`]: a directory that opens with
/// [`StorageEngine::open_checkpoint`], removed when this drops.
#[derive(Debug)]
pub struct ReceivedStore {
    dir: PathBuf,
    inventory: Option<Vec<u8>>,
    positions: Vec<(Partition, RaftPosition)>,
}

impl ReceivedStore {
    /// The directory holding the received trees.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The sender's inventory of the index copies the store carries; `None`
    /// leaves every copy unknown to the receiver.
    #[must_use]
    pub fn index_inventory(&self) -> Option<&[u8]> {
        self.inventory.as_deref()
    }

    /// The partitions that stand past the transfer's position, each with
    /// its own.
    #[must_use]
    pub fn positions(&self) -> &[(Partition, RaftPosition)] {
        &self.positions
    }
}

impl Drop for ReceivedStore {
    fn drop(&mut self) {
        remove_quietly(&self.dir);
    }
}

/// One tree read from the stream, its files already in place.
struct ReceivedTree {
    name: TreeName,
    position: Option<RaftPosition>,
    layout: Vec<u8>,
    entries: Vec<PinnedEntry>,
    tombstones: Vec<PinnedRangeTombstone>,
}

/// Read a [`StoreTransfer`] stream from `reader` into `dir`, which must not
/// exist, and rebuild every tree it carries there. The stream's checksum is
/// verified before any tree is rebuilt, and the reader must end right after
/// it. A failure removes `dir`.
///
/// Blocks; call it off the async runtime.
///
/// # Errors
///
/// `dir` exists or cannot be written, the stream is malformed, truncated or
/// fails its checksum, or a tree cannot be rebuilt from it.
pub fn receive_store(reader: &mut impl Read, dir: &Path) -> StorageResult<ReceivedStore> {
    crate::engine::core::claim_checkpoint_dir(dir)?;
    let mut received = ReceivedStore {
        dir: dir.to_path_buf(),
        inventory: None,
        positions: Vec::new(),
    };
    let mut hashing = HashingReader {
        inner: reader,
        hasher: Xxh3::new(),
    };
    let trees = read_trees(&mut hashing, dir)?;
    received.inventory = match read_u8(&mut hashing)? {
        0 => None,
        1 => Some(read_bytes(&mut hashing)?),
        other => return Err(malformed(format!("inventory flag {other}"))),
    };
    let actual = hashing.hasher.digest();
    let mut checksum = [0u8; 8];
    hashing
        .inner
        .read_exact(&mut checksum)
        .map_err(read_error)?;
    if u64::from_le_bytes(checksum) != actual {
        return Err(malformed("checksum mismatch".into()));
    }
    if hashing.inner.read(&mut [0u8; 1]).map_err(read_error)? != 0 {
        return Err(malformed("bytes past the checksum".into()));
    }

    for tree in trees {
        if let (TreeName::Partition(part), Some(position)) = (&tree.name, &tree.position) {
            received.positions.push((*part, position.clone()));
        }
        restore_tree(dir, tree)?;
    }
    // A received store is measured by the engine that opens it; the transfer
    // writes beside the member's store unheld by its footprint.
    crate::format::write_marker(
        dir,
        coordinode_core::version::engine_format_version(),
        &crate::engine::footprint::Footprints::default(),
    )?;
    Ok(received)
}

/// Read every tree of the stream, writing its files into place under `dir`.
fn read_trees(reader: &mut impl Read, dir: &Path) -> StorageResult<Vec<ReceivedTree>> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(read_error)?;
    if magic != MAGIC {
        return Err(malformed("not a store transfer".into()));
    }
    let version = read_u8(reader)?;
    if version != VERSION {
        return Err(malformed(format!("version {version}")));
    }
    let count = read_u32(reader)?;
    let mut trees: Vec<ReceivedTree> = Vec::new();
    for _ in 0..count {
        let name = String::from_utf8(read_short(reader)?)
            .map_err(|_| malformed("a tree name is not UTF-8".into()))?;
        let name = TreeName::decode(&name)?;
        if trees.iter().any(|t| t.name == name) {
            return Err(malformed(format!("tree {} twice", name.encode())));
        }
        let position = match read_u8(reader)? {
            0 => None,
            1 => Some(RaftPosition {
                next: read_u64(reader)?,
                payload: read_bytes(reader)?,
            }),
            other => return Err(malformed(format!("position flag {other}"))),
        };
        let layout = read_bytes(reader)?;
        let root = name.dir(dir);
        for _ in 0..read_u32(reader)? {
            let kind = FileKind::from_tag(read_u8(reader)?)?;
            let id = read_u64(reader)?;
            let folder = root.join(kind.folder());
            std::fs::create_dir_all(&folder)
                .map_err(|e| StorageError::Io(format!("create {folder:?}: {e}")))?;
            let path = folder.join(id.to_string());
            match read_u8(reader)? {
                0 => {}
                1 => {
                    let attestation = read_bytes(reader)?;
                    let mut name = path.clone().into_os_string();
                    name.push(".heal-attest");
                    write_new(&PathBuf::from(name), &mut attestation.as_slice(), None)?;
                }
                other => return Err(malformed(format!("attestation flag {other}"))),
            }
            let len = read_u64(reader)?;
            write_new(&path, reader, Some(len))?;
        }
        let entries = read_entries(reader)?;
        let mut tombstones = Vec::new();
        for _ in 0..read_u32(reader)? {
            tombstones.push(PinnedRangeTombstone {
                start: read_bytes(reader)?.into(),
                end: read_bytes(reader)?.into(),
                seqno: read_u64(reader)?,
            });
        }
        trees.push(ReceivedTree {
            name,
            position,
            layout,
            entries,
            tombstones,
        });
    }
    Ok(trees)
}

fn read_entries(reader: &mut impl Read) -> StorageResult<Vec<PinnedEntry>> {
    let count = read_u64(reader)?;
    let mut entries = Vec::new();
    for _ in 0..count {
        let key = read_bytes(reader)?;
        let value = read_bytes(reader)?;
        let seqno = read_u64(reader)?;
        let value_type = lsm_tree::ValueType::try_from(read_u8(reader)?)
            .map_err(|_| malformed("an entry's value type".into()))?;
        entries.push(PinnedEntry {
            key: key.into(),
            value: value.into(),
            seqno,
            value_type,
        });
    }
    Ok(entries)
}

/// Rebuild `tree` in its folder under `root`, configured as the engine opens
/// that tree.
fn restore_tree(root: &Path, tree: ReceivedTree) -> StorageResult<()> {
    let seqno =
        lsm_tree::SharedSequenceNumberGenerator::from(lsm_tree::SequenceNumberCounter::default());
    let config = match &tree.name {
        TreeName::Partition(part) => {
            use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
            StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                root,
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )])
            .relocated()
            .to_tree_config(*part, seqno)
        }
        // A columnar table's memtable contents land as row tables beside its
        // columnar ones; its projected scan reads both.
        TreeName::Table(_) => {
            lsm_tree::Config::new_with_generators(tree.name.dir(root), seqno.clone(), seqno)
        }
    };
    config
        .restore_pinned(&tree.layout, tree.entries, tree.tombstones)
        .map_err(|e| StorageError::Io(format!("rebuild {}: {e}", tree.name.encode())))?;
    Ok(())
}

/// The encoded size of `entries`.
fn entries_len(entries: impl Iterator<Item = PinnedEntry>) -> StorageResult<u64> {
    let mut len = 8u64;
    for entry in entries {
        check_u32(entry.key.len())?;
        check_u32(entry.value.len())?;
        len += 4 + entry.key.len() as u64 + 4 + entry.value.len() as u64 + 8 + 1;
    }
    Ok(len)
}

fn encode_entries(entries: impl Iterator<Item = PinnedEntry>) -> io::Result<Vec<u8>> {
    let entries: Vec<PinnedEntry> = entries.collect();
    let mut out = Vec::new();
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for entry in &entries {
        put_bytes(&mut out, &entry.key).map_err(io::Error::other)?;
        put_bytes(&mut out, &entry.value).map_err(io::Error::other)?;
        out.extend_from_slice(&entry.seqno.to_le_bytes());
        out.push(u8::from(entry.value_type));
    }
    Ok(out)
}

/// The whole of a small file lying beside a pinned one.
fn read_whole(file: &PinnedFile, path: &Path) -> StorageResult<Vec<u8>> {
    std::fs::read(path).map_err(|e| {
        StorageError::Io(format!(
            "read the attestation of pinned file {}: {e}",
            file.id()
        ))
    })
}

/// Write a new file at `path` from `reader`: exactly `len` bytes, or all of
/// it when `len` is `None`. Not synced: the rebuild syncs every file before
/// its tree's manifest names it.
fn write_new(path: &Path, reader: &mut impl Read, len: Option<u64>) -> StorageResult<()> {
    let io = |e: io::Error| StorageError::Io(format!("write {path:?}: {e}"));
    let mut file = std::fs::File::create_new(path).map_err(io)?;
    let copied = match len {
        Some(len) => io::copy(&mut reader.take(len), &mut file).map_err(io)?,
        None => io::copy(reader, &mut file).map_err(io)?,
    };
    if len.is_some_and(|len| copied != len) {
        return Err(malformed(format!("{path:?} is truncated")));
    }
    file.flush().map_err(io)
}

/// Remove `dir` and everything under it; a failure only leaves it behind.
fn remove_quietly(dir: &Path) {
    if let Err(e) = std::fs::remove_dir_all(dir)
        && e.kind() != io::ErrorKind::NotFound
    {
        tracing::warn!(?dir, %e, "could not remove a received store");
    }
}

fn malformed(what: String) -> StorageError {
    StorageError::Serialization(format!("store transfer: {what}"))
}

fn read_error(e: io::Error) -> StorageError {
    if e.kind() == io::ErrorKind::UnexpectedEof {
        malformed("truncated".into())
    } else {
        StorageError::Io(format!("read a store transfer: {e}"))
    }
}

fn check_u32(n: usize) -> StorageResult<u32> {
    u32::try_from(n).map_err(|_| malformed(format!("a field of {n} bytes exceeds u32")))
}

fn put_u16(out: &mut Vec<u8>, n: usize) -> StorageResult<()> {
    let n = u16::try_from(n).map_err(|_| malformed(format!("a name of {n} bytes")))?;
    out.extend_from_slice(&n.to_le_bytes());
    Ok(())
}

fn put_u32(out: &mut Vec<u8>, n: usize) -> StorageResult<()> {
    out.extend_from_slice(&check_u32(n)?.to_le_bytes());
    Ok(())
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> StorageResult<()> {
    put_u32(out, bytes.len())?;
    out.extend_from_slice(bytes);
    Ok(())
}

fn read_u8(reader: &mut impl Read) -> StorageResult<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf).map_err(read_error)?;
    Ok(buf[0])
}

fn read_u32(reader: &mut impl Read) -> StorageResult<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).map_err(read_error)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(reader: &mut impl Read) -> StorageResult<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).map_err(read_error)?;
    Ok(u64::from_le_bytes(buf))
}

/// A length-prefixed byte string. The buffer grows with the bytes actually
/// read rather than trusting the length up front: a corrupt length ends in a
/// truncation error, not a huge allocation.
fn read_sized(reader: &mut impl Read, len: u64) -> StorageResult<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take(len)
        .read_to_end(&mut bytes)
        .map_err(read_error)?;
    if bytes.len() as u64 != len {
        return Err(malformed("truncated".into()));
    }
    Ok(bytes)
}

fn read_bytes(reader: &mut impl Read) -> StorageResult<Vec<u8>> {
    let len = read_u32(reader)?;
    read_sized(reader, u64::from(len))
}

fn read_short(reader: &mut impl Read) -> StorageResult<Vec<u8>> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf).map_err(read_error)?;
    read_sized(reader, u64::from(u16::from_le_bytes(buf)))
}

/// Hands `inner`'s bytes through, folding each into the stream checksum.
struct HashingReader<'a, R> {
    inner: &'a mut R,
    hasher: Xxh3,
}

impl<R: Read> Read for HashingReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
