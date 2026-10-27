//! BlobStore identities: the logical content of a chunk, the immutable
//! representations that hold its bytes, the catalog that publishes them, and
//! the owners that reference content.
//!
//! Large values are split into chunks. A chunk's [`ContentId`] is what it
//! says (digest algorithm, digest and length), not where its bytes are: a
//! [`BlobRef`] lists contents in object order, and reading resolves each one
//! through the catalog to its eligible [`Representation`]. Identical content
//! within one [`SharingScope`] shares one representation.
//!
//! Three identities stay apart. The content is logical. A representation is
//! one physical copy of it, written once under a generation no other copy of
//! the same content ever reuses, so a cleanup aimed at a retired copy can
//! never reach a newer one of the same bytes. An owner (a direct blob, an S3
//! object, a node property) is what holds references, and two owners of the
//! same bytes are two owners.
//!
//! ## Storage layout
//!
//! ```text
//! Blob partition
//!   blob:<scope><content><generation>        → encoded chunk bytes
//! BlobRef partition
//!   blobcat:<scope><content>                 → CatalogEntry (eligible copy)
//!   blobmeta:<owner id>                      → OwnerRecord (direct blob)
//!   s3:<path>                                → OwnerRecord (S3 object)
//!   blobref:<node id><property id>           → OwnerRecord (node property)
//!   blobstage:<staging id>                   → StagingRecord
//!   blobcand:<release µs><scope><content><owner><incarnation>
//!                                            → (release candidate)
//!   blobretired:<scope><content><generation> → (payload awaiting cleanup)
//!   blobgc:compact                           → oldest cleanup awaiting compaction (µs)
//!   blobgc:state                             → collector record
//! Counter partition
//!   counter:blob_refs:<scope>:<content hex>  → owners referencing the content
//!   counter:blob_pins:<scope>:<content hex>  → uploads holding the content
//!   counter:blob_gc:<measure>                → what the collector reclaimed
//! ```
//!
//! ## Inline threshold
//!
//! Values smaller than 4KB are stored directly as node properties.
//! Values >= 4KB are chunked and stored via BlobStore.

use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::node::NodeId;

/// Default chunk size: 256KB.
pub const DEFAULT_CHUNK_SIZE: usize = 256 * 1024;

/// Maximum chunk size: 16MB.
pub const MAX_CHUNK_SIZE: usize = 16 * 1024 * 1024;

/// Inline threshold: values below this are stored as node properties.
pub const INLINE_THRESHOLD: usize = 4 * 1024;

/// The digest algorithm a content identity is computed with. Part of the
/// identity: equal digests under different algorithms are different content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DigestAlgorithm {
    /// SHA-256 over the raw chunk bytes.
    Sha256,
}

impl DigestAlgorithm {
    /// The byte that names the algorithm in keys.
    pub const fn tag(self) -> u8 {
        match self {
            Self::Sha256 => 1,
        }
    }

    /// The algorithm a key byte names.
    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::Sha256),
            _ => None,
        }
    }
}

/// The logical content of one chunk: its digest under a named algorithm and
/// its verified raw length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ContentId {
    algorithm: DigestAlgorithm,
    digest: [u8; 32],
    len: u32,
}

impl ContentId {
    /// Bytes a content identity takes in a key: algorithm, digest, length.
    pub const ENCODED_LEN: usize = 1 + 32 + 4;

    /// The identity of `data`.
    ///
    /// # Panics
    ///
    /// When `data` is longer than [`MAX_CHUNK_SIZE`]: a chunk is never that
    /// long, so such a call is a programming error.
    pub fn of(data: &[u8]) -> Self {
        assert!(data.len() <= MAX_CHUNK_SIZE, "a chunk is at most 16MB");
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&Sha256::digest(data));
        Self {
            algorithm: DigestAlgorithm::Sha256,
            digest,
            // At most MAX_CHUNK_SIZE, asserted above.
            len: data.len() as u32,
        }
    }

    /// An identity from its parts.
    pub const fn from_parts(algorithm: DigestAlgorithm, digest: [u8; 32], len: u32) -> Self {
        Self {
            algorithm,
            digest,
            len,
        }
    }

    /// The digest algorithm.
    pub const fn algorithm(&self) -> DigestAlgorithm {
        self.algorithm
    }

    /// The digest.
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// The raw length in bytes.
    pub const fn len(&self) -> u32 {
        self.len
    }

    /// Whether the content is empty. A chunk never is; present for the
    /// length accessor's sake.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether `data` is this content: same length and same digest.
    pub fn matches(&self, data: &[u8]) -> bool {
        data.len() == self.len as usize && Self::of(data).digest == self.digest
    }

    /// Append the key form: algorithm tag, digest, length big-endian.
    pub fn encode_into(&self, out: &mut Vec<u8>) {
        out.push(self.algorithm.tag());
        out.extend_from_slice(&self.digest);
        out.extend_from_slice(&self.len.to_be_bytes());
    }

    /// Read the key form [`Self::encode_into`] writes from the front of
    /// `bytes`.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let (&tag, rest) = bytes.split_first()?;
        let digest: [u8; 32] = rest.get(..32)?.try_into().ok()?;
        let len = u32::from_be_bytes(rest.get(32..36)?.try_into().ok()?);
        Some(Self {
            algorithm: DigestAlgorithm::from_tag(tag)?,
            digest,
            len,
        })
    }

    /// Lowercase hex of the key form, for counter keys and logs.
    pub fn to_hex(&self) -> String {
        let mut raw = Vec::with_capacity(Self::ENCODED_LEN);
        self.encode_into(&mut raw);
        hex(&raw)
    }

    /// The digest alone in lowercase hex, as clients see a chunk.
    pub fn digest_hex(&self) -> String {
        hex(&self.digest)
    }
}

impl core::fmt::Display for ContentId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for b in &self.digest {
            write!(f, "{b:02x}")?;
        }
        write!(f, "/{}", self.len)
    }
}

fn hex(bytes: &[u8]) -> String {
    use core::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        // Writing to a String cannot fail.
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// The scope within which identical content may share one representation.
/// Content is looked up only after its scope is resolved: a digest alone is
/// never authority to share bytes across scopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SharingScope(u64);

impl SharingScope {
    /// The store's own scope: every owner of a store that has no narrower
    /// sharing rule shares within it.
    pub const STORE: Self = Self(0);

    /// A scope from its number.
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The scope's number.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// Content within its sharing scope: what the catalog, the counters and the
/// claims are keyed by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ScopedContent {
    /// The scope the content is shared in.
    pub scope: SharingScope,
    /// The content.
    pub content: ContentId,
}

impl ScopedContent {
    /// `content` in `scope`.
    pub const fn new(scope: SharingScope, content: ContentId) -> Self {
        Self { scope, content }
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.scope.0.to_be_bytes());
        self.content.encode_into(out);
    }

    /// Bytes the scoped content takes in a key.
    pub const ENCODED_LEN: usize = 8 + ContentId::ENCODED_LEN;

    /// Read the key form from the front of `bytes`.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let scope = u64::from_be_bytes(bytes.get(..8)?.try_into().ok()?);
        Some(Self {
            scope: SharingScope(scope),
            content: ContentId::decode(bytes.get(8..)?)?,
        })
    }
}

/// How a representation's bytes encode its content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Codec {
    /// The raw chunk bytes; the content digest is their integrity evidence.
    Raw,
}

/// One immutable physical copy of a content, as the catalog publishes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Representation {
    /// The generation its payload key carries. Unique per content for the
    /// life of the store: a retired generation is never reused.
    pub generation: u64,
    /// How the bytes encode the content.
    pub codec: Codec,
    /// Length of the encoded bytes.
    pub encoded_len: u32,
}

/// The catalog's record of one scoped content, kept while the content has a
/// representation readers and new references resolve to. A retired
/// representation leaves the catalog and waits for cleanup under its own
/// [`retired_key`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// The representation readers and new references resolve to.
    pub eligible: Representation,
}

impl CatalogEntry {
    /// MessagePack encoding.
    pub fn to_msgpack(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(self)
    }

    /// Decode [`Self::to_msgpack`]'s output.
    pub fn from_msgpack(data: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(data)
    }
}

/// An object's content, in order: the chunks, repeated where the object
/// repeats them, that reassemble it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlobRef {
    /// The scope the contents resolve in.
    pub scope: SharingScope,
    /// The chunks in object order, with their multiplicity.
    pub chunks: Vec<ContentId>,
    /// Total size of the object in bytes.
    pub total_size: u64,
}

impl BlobRef {
    /// A reference to `chunks` in `scope`, `total_size` bytes in all.
    pub fn new(scope: SharingScope, chunks: Vec<ContentId>, total_size: u64) -> Self {
        Self {
            scope,
            chunks,
            total_size,
        }
    }

    /// Number of chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// The distinct contents the object references, each once, sorted.
    /// Ownership is counted per owner and distinct content, whatever the
    /// multiplicity; sorted so membership is a binary search.
    pub fn distinct(&self) -> Vec<ScopedContent> {
        let mut distinct: Vec<ScopedContent> = self
            .chunks
            .iter()
            .map(|content| ScopedContent::new(self.scope, *content))
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        distinct
    }

    /// Serialize to MessagePack.
    pub fn to_msgpack(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(self)
    }

    /// Deserialize from MessagePack.
    pub fn from_msgpack(data: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(data)
    }
}

/// What an owner holds: its incarnation and the object it references.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OwnerRecord {
    /// Distinguishes this publication from every earlier and later one at
    /// the same owner key, so a release and its candidates name the exact
    /// holding they end.
    pub incarnation: u64,
    /// The object.
    pub blob: BlobRef,
}

impl OwnerRecord {
    /// MessagePack encoding.
    pub fn to_msgpack(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(self)
    }

    /// Decode [`Self::to_msgpack`]'s output.
    pub fn from_msgpack(data: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(data)
    }
}

/// One content an upload holds while it is staged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagedContent {
    /// The content.
    pub content: ScopedContent,
    /// The generation the upload uses: an eligible one it shares, or a new
    /// one it writes.
    pub generation: u64,
    /// Whether the upload writes the payload under `generation` itself.
    pub writes: bool,
}

/// An upload between admission and publication. Its contents are pinned
/// while the record exists, and a publication requires the record, so an
/// upload whose record was fenced cannot publish afterwards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagingRecord {
    /// When the upload was admitted, in microseconds since the epoch.
    pub admitted_at_micros: i64,
    /// The distinct contents it holds.
    pub contents: Vec<StagedContent>,
}

impl StagingRecord {
    /// MessagePack encoding.
    pub fn to_msgpack(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(self)
    }

    /// Decode [`Self::to_msgpack`]'s output.
    pub fn from_msgpack(data: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(data)
    }
}

// -- Key encoding --

/// Key prefix of payload in the Blob partition.
pub const PAYLOAD_PREFIX: &[u8] = b"blob:";
/// Key prefix of catalog entries.
pub const CATALOG_PREFIX: &[u8] = b"blobcat:";
/// Key prefix of direct blob owners.
pub const DIRECT_OWNER_PREFIX: &[u8] = b"blobmeta:";
/// Key prefix of S3 object owners.
pub const S3_OWNER_PREFIX: &[u8] = b"s3:";
/// Key prefix of node property owners.
pub const PROPERTY_OWNER_PREFIX: &[u8] = b"blobref:";
/// Key prefix of staged uploads.
pub const STAGING_PREFIX: &[u8] = b"blobstage:";
/// Key prefix of release candidates.
pub const CANDIDATE_PREFIX: &[u8] = b"blobcand:";
/// Key prefix of retired representations awaiting cleanup.
pub const RETIRED_PREFIX: &[u8] = b"blobretired:";
/// The key whose presence says payload was cleaned up and the compaction
/// that returns its space has not run since. Its value is when the oldest
/// such cleanup ran (microseconds since the epoch, big-endian); its version
/// is the latest one's commit, which the compaction must pass.
pub const COMPACTION_PENDING_KEY: &[u8] = b"blobgc:compact";
/// Counter of the payload bytes cleaned up since the last compaction the
/// collector ran.
pub const UNCOMPACTED_BYTES_KEY: &[u8] = b"counter:blob_gc:uncompacted_bytes";
/// The collector's own record: when accounting was last reconciled, and why
/// collection is stopped if it is.
pub const COLLECTOR_STATE_KEY: &[u8] = b"blobgc:state";
/// Key prefix of the owner reference counters.
pub const REFS_COUNTER_PREFIX: &[u8] = b"counter:blob_refs:";
/// Key prefix of the upload pin counters.
pub const PINS_COUNTER_PREFIX: &[u8] = b"counter:blob_pins:";
/// Counter of the payload bytes the collector deleted.
pub const RECLAIMED_BYTES_KEY: &[u8] = b"counter:blob_gc:reclaimed_bytes";
/// Counter of the payload chunks the collector deleted.
pub const RECLAIMED_CHUNKS_KEY: &[u8] = b"counter:blob_gc:reclaimed_chunks";
/// Counter of the bytes the Blob partition shrank by in the compactions the
/// collector ran.
pub const RECLAIMED_DISK_BYTES_KEY: &[u8] = b"counter:blob_gc:reclaimed_disk_bytes";

/// The payload key of `content`'s representation `generation`.
pub fn payload_key(content: &ScopedContent, generation: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(PAYLOAD_PREFIX.len() + ScopedContent::ENCODED_LEN + 8);
    key.extend_from_slice(PAYLOAD_PREFIX);
    content.encode_into(&mut key);
    key.extend_from_slice(&generation.to_be_bytes());
    key
}

/// The catalog key of `content`.
pub fn catalog_key(content: &ScopedContent) -> Vec<u8> {
    let mut key = Vec::with_capacity(CATALOG_PREFIX.len() + ScopedContent::ENCODED_LEN);
    key.extend_from_slice(CATALOG_PREFIX);
    content.encode_into(&mut key);
    key
}

/// The counter key of the owners referencing `content`.
pub fn refs_counter_key(content: &ScopedContent) -> Vec<u8> {
    counter_key(REFS_COUNTER_PREFIX, content)
}

/// The counter key of the uploads holding `content`.
pub fn pins_counter_key(content: &ScopedContent) -> Vec<u8> {
    counter_key(PINS_COUNTER_PREFIX, content)
}

fn counter_key(prefix: &[u8], content: &ScopedContent) -> Vec<u8> {
    let mut key = prefix.to_vec();
    key.extend_from_slice(
        alloc::format!("{}:{}", content.scope.as_raw(), content.content.to_hex()).as_bytes(),
    );
    key
}

/// The content a counter key under `prefix` ([`REFS_COUNTER_PREFIX`] or
/// [`PINS_COUNTER_PREFIX`]) counts, or `None` for a key that is not one.
pub fn decode_counter_key(prefix: &[u8], key: &[u8]) -> Option<ScopedContent> {
    let rest = core::str::from_utf8(key.strip_prefix(prefix)?).ok()?;
    let (scope, content) = rest.split_once(':')?;
    let scope = scope.parse::<u64>().ok()?;
    if content.len() != ContentId::ENCODED_LEN * 2 {
        return None;
    }
    let mut raw = [0u8; ContentId::ENCODED_LEN];
    for (byte, pair) in raw.iter_mut().zip(content.as_bytes().chunks(2)) {
        *byte = u8::from_str_radix(core::str::from_utf8(pair).ok()?, 16).ok()?;
    }
    Some(ScopedContent::new(
        SharingScope(scope),
        ContentId::decode(&raw)?,
    ))
}

/// The key of the staged upload `staging_id`.
pub fn staging_key(staging_id: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(STAGING_PREFIX.len() + 8);
    key.extend_from_slice(STAGING_PREFIX);
    key.extend_from_slice(&staging_id.to_be_bytes());
    key
}

/// The staging id a staging key names.
pub fn decode_staging_key(key: &[u8]) -> Option<u64> {
    let id = key.strip_prefix(STAGING_PREFIX)?;
    Some(u64::from_be_bytes(id.try_into().ok()?))
}

/// The candidate key recording that `owner_key` at `incarnation` released
/// `content` at `released_at_micros`. One per release, so concurrent releases
/// never write one key; ordered by release time first, so a collector reads
/// them oldest first and stops at the first one still in its grace.
pub fn candidate_key(
    released_at_micros: i64,
    content: &ScopedContent,
    owner_key: &[u8],
    incarnation: u64,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(CANDIDATE_PREFIX.len() + 8 + ScopedContent::ENCODED_LEN + 40);
    key.extend_from_slice(CANDIDATE_PREFIX);
    key.extend_from_slice(&micros_key(released_at_micros));
    content.encode_into(&mut key);
    key.extend_from_slice(&Sha256::digest(owner_key));
    key.extend_from_slice(&incarnation.to_be_bytes());
    key
}

/// A release candidate as its key records it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    /// When the owner released the content, in microseconds since the epoch.
    pub released_at_micros: i64,
    /// The content released.
    pub content: ScopedContent,
}

/// The candidate a candidate key records.
pub fn decode_candidate_key(key: &[u8]) -> Option<Candidate> {
    let rest = key.strip_prefix(CANDIDATE_PREFIX)?;
    let released = u64::from_be_bytes(rest.get(..8)?.try_into().ok()?);
    Some(Candidate {
        // `micros_key` wrote a non-negative time, so it fits back.
        released_at_micros: i64::try_from(released).ok()?,
        content: ScopedContent::decode(rest.get(8..)?)?,
    })
}

/// The smallest candidate key released after `micros`: a scan of candidates
/// whose grace has passed stops here.
pub fn candidates_released_after(micros: i64) -> Vec<u8> {
    let mut key = CANDIDATE_PREFIX.to_vec();
    // Nothing is released after the last representable time: the bound
    // stays there.
    key.extend_from_slice(&micros_key(micros.saturating_add(1)));
    key
}

/// A time in the order keys sort by. Times are taken from a clock that reads
/// the epoch when set before it, so they are never negative; one that is
/// sorts as the epoch.
fn micros_key(micros: i64) -> [u8; 8] {
    u64::try_from(micros).unwrap_or(0).to_be_bytes()
}

/// The key recording that `content`'s representation `generation` is
/// retired and its payload awaits cleanup.
pub fn retired_key(content: &ScopedContent, generation: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(RETIRED_PREFIX.len() + ScopedContent::ENCODED_LEN + 8);
    key.extend_from_slice(RETIRED_PREFIX);
    content.encode_into(&mut key);
    key.extend_from_slice(&generation.to_be_bytes());
    key
}

/// The content and generation a retired key names.
pub fn decode_retired_key(key: &[u8]) -> Option<(ScopedContent, u64)> {
    decode_content_generation(key.strip_prefix(RETIRED_PREFIX)?)
}

/// The content and generation a payload key names.
pub fn decode_payload_key(key: &[u8]) -> Option<(ScopedContent, u64)> {
    decode_content_generation(key.strip_prefix(PAYLOAD_PREFIX)?)
}

fn decode_content_generation(rest: &[u8]) -> Option<(ScopedContent, u64)> {
    if rest.len() != ScopedContent::ENCODED_LEN + 8 {
        return None;
    }
    let content = ScopedContent::decode(rest)?;
    let generation = u64::from_be_bytes(rest.get(ScopedContent::ENCODED_LEN..)?.try_into().ok()?);
    Some((content, generation))
}

/// The owner key of the direct blob `owner_id`.
pub fn direct_owner_key(owner_id: &[u8; 16]) -> Vec<u8> {
    let mut key = Vec::with_capacity(DIRECT_OWNER_PREFIX.len() + 16);
    key.extend_from_slice(DIRECT_OWNER_PREFIX);
    key.extend_from_slice(owner_id);
    key
}

/// The owner key of the S3 object at `path`.
pub fn s3_owner_key(path: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(S3_OWNER_PREFIX.len() + path.len());
    key.extend_from_slice(S3_OWNER_PREFIX);
    key.extend_from_slice(path.as_bytes());
    key
}

/// The owner key of a node property: `blobref:<node_id BE>:<prop_id u32 BE>`.
pub fn encode_blobref_key(node_id: NodeId, prop_id: u32) -> Vec<u8> {
    let mut key = Vec::with_capacity(PROPERTY_OWNER_PREFIX.len() + 8 + 1 + 4);
    key.extend_from_slice(PROPERTY_OWNER_PREFIX);
    key.extend_from_slice(&node_id.as_raw().to_be_bytes());
    key.push(b':');
    key.extend_from_slice(&prop_id.to_be_bytes());
    key
}

/// Whether `key` in the BlobRef partition names an owner.
pub fn is_owner_key(key: &[u8]) -> bool {
    key.starts_with(DIRECT_OWNER_PREFIX)
        || key.starts_with(S3_OWNER_PREFIX)
        || key.starts_with(PROPERTY_OWNER_PREFIX)
}

// -- Chunking --

/// Split data into chunks, each with its content identity.
///
/// Identical chunks get identical identities, which is what lets them share
/// one representation.
pub fn chunk_data(data: &[u8], chunk_size: usize) -> Vec<(ContentId, &[u8])> {
    assert!(chunk_size > 0 && chunk_size <= MAX_CHUNK_SIZE);
    data.chunks(chunk_size)
        .map(|chunk| (ContentId::of(chunk), chunk))
        .collect()
}

/// Check if a value should be stored inline (< threshold) or via BlobStore.
pub fn should_inline(data: &[u8]) -> bool {
    data.len() < INLINE_THRESHOLD
}

/// The reference to `data` in `scope`, cut into default-size chunks, and the
/// chunks themselves.
pub fn create_blob(scope: SharingScope, data: &[u8]) -> (BlobRef, Vec<(ContentId, &[u8])>) {
    create_blob_with_chunk_size(scope, data, DEFAULT_CHUNK_SIZE)
}

/// [`create_blob`] with a chosen chunk size.
pub fn create_blob_with_chunk_size(
    scope: SharingScope,
    data: &[u8],
    chunk_size: usize,
) -> (BlobRef, Vec<(ContentId, &[u8])>) {
    let chunks = chunk_data(data, chunk_size);
    let ids: Vec<ContentId> = chunks.iter().map(|(id, _)| *id).collect();
    (BlobRef::new(scope, ids, data.len() as u64), chunks)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
