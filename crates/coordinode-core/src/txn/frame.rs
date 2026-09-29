//! Compact, self-contained encoding of one application unit (a
//! [`RaftProposal`]): the bytes the journal persists, replication carries and
//! recovery replays.
//!
//! A frame removes representation redundancy without changing what it
//! carries. Keys are front-coded against the previous key of the same
//! partition, so the prefixes a transaction repeats (a node's `node:<shard>:`,
//! an index's name) are written once. A value or operand identical to one
//! already in the frame is a reference to it. Every reference addresses
//! validated material of the same frame; decoding needs no earlier message,
//! dictionary or lookup, and expands to exactly the mutations that were
//! encoded, in their order.
//!
//! ```text
//! frame  = version:u8  id:varint  commit_ts:varint  start_ts:varint  flags:u8
//!          op_count:varint  op*  crc32c:u32le
//! op     = tag:u8  body            tag = kind | partition << 3
//! key    = shared:varint  suffix_len:varint  suffix
//! bytes  = v:varint  (v even: literal of v/2 bytes follows;
//!                     v odd: the (v/2)-th literal of at least MIN_SHARED bytes)
//! ```
//!
//! Decoding checks the version, the checksum, every length and reference and
//! the expansion budget before it allocates for them.

use rustc_hash::FxHashMap;

use super::proposal::{
    DerivedIndexWork, DerivedSource, IndexBinding, Mutation, PartitionId, ProposalId, RaftProposal,
};
use super::timestamp::Timestamp;

/// The frame layout this build writes and reads.
pub const FRAME_VERSION: u8 = 1;

/// Shortest literal a later identical slice may refer to: below this a
/// reference saves nothing.
const MIN_SHARED: usize = 8;

const KIND_PUT: u8 = 0;
const KIND_DELETE: u8 = 1;
const KIND_MERGE: u8 = 2;
const KIND_REMOVE_RANGE: u8 = 3;
const KIND_COMMAND: u8 = 4;
const KIND_DERIVE: u8 = 5;
const KIND_MASK: u8 = 0b111;

const SOURCE_UNIT_RECORD: u8 = 0;
const SOURCE_VALUES: u8 = 1;

const FLAG_BYPASS_RATE_LIMITER: u8 = 1;

/// Every partition in its frame number, the same numbering the snapshot and
/// the write buffer use.
const PARTITIONS: [PartitionId; 10] = [
    PartitionId::Node,
    PartitionId::Adj,
    PartitionId::EdgeProp,
    PartitionId::Blob,
    PartitionId::BlobRef,
    PartitionId::Schema,
    PartitionId::Idx,
    PartitionId::Counter,
    PartitionId::VectorF32,
    PartitionId::Registry,
];

fn partition_number(partition: PartitionId) -> u8 {
    match partition {
        PartitionId::Node => 0,
        PartitionId::Adj => 1,
        PartitionId::EdgeProp => 2,
        PartitionId::Blob => 3,
        PartitionId::BlobRef => 4,
        PartitionId::Schema => 5,
        PartitionId::Idx => 6,
        PartitionId::Counter => 7,
        PartitionId::VectorF32 => 8,
        PartitionId::Registry => 9,
    }
}

/// Bounds a decoder holds a frame to, checked before the work they bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeLimits {
    /// Largest encoded frame accepted.
    pub max_frame_bytes: usize,
    /// Most operations one frame may carry.
    pub max_ops: usize,
    /// Most bytes of keys, values and commands the frame may expand to.
    pub max_expanded_bytes: usize,
}

impl DecodeLimits {
    /// The bounds replication and recovery apply: far above any unit a
    /// transaction or a backfill page produces, far below what would exhaust
    /// a member.
    pub const DEFAULT: Self = Self {
        max_frame_bytes: 256 << 20,
        max_ops: 4 << 20,
        max_expanded_bytes: 1 << 30,
    };
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Why bytes are not a frame this build accepts.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FrameError {
    /// The frame ends before a field it declares.
    #[error("frame truncated at byte {at}")]
    Truncated {
        /// Offset where the missing field starts.
        at: usize,
    },
    /// The checksum does not match the frame's bytes.
    #[error("frame checksum mismatch")]
    Checksum,
    /// A layout version this build does not read.
    #[error("frame version {0} is not supported")]
    UnsupportedVersion(u8),
    /// An operation tag names no kind or partition.
    #[error("frame operation tag {0:#04x} is unknown")]
    UnknownTag(u8),
    /// Flags this build does not know.
    #[error("frame flags {0:#04x} are unknown")]
    UnknownFlags(u8),
    /// A key shares more bytes with its predecessor than the predecessor has.
    #[error("key shares {shared} bytes with a predecessor of {available}")]
    BadPrefix {
        /// Bytes the key claims to share.
        shared: usize,
        /// Bytes the predecessor has.
        available: usize,
    },
    /// A reference to a slice the frame has not carried.
    #[error("reference to slice {index} of {count}")]
    BadReference {
        /// The slice referred to.
        index: usize,
        /// Slices carried so far.
        count: usize,
    },
    /// A varint longer than a u64.
    #[error("varint at byte {at} overflows")]
    VarintOverflow {
        /// Offset where the varint starts.
        at: usize,
    },
    /// The frame exceeds a decode bound.
    #[error("frame exceeds its {what} bound of {limit}")]
    OverLimit {
        /// The bound exceeded.
        what: &'static str,
        /// Its value.
        limit: usize,
    },
    /// A metadata command whose bytes do not decode.
    #[error("frame command does not decode: {0}")]
    Command(String),
    /// A DERIVED binding or input whose bytes do not decode.
    #[error("frame index derivation does not decode: {0}")]
    Derive(String),
    /// A DERIVED binding whose interpretation this build cannot derive.
    #[error(transparent)]
    Unsupported(#[from] crate::index::derive::UnsupportedInterpretation),
    /// A DERIVED source naming no earlier node record put of the unit.
    #[error("derivation source {ordinal} is not an earlier node record put")]
    BadSource {
        /// The position named.
        ordinal: u32,
    },
    /// Bytes after the last operation and before the checksum.
    #[error("{0} bytes follow the last operation")]
    Trailing(usize),
}

fn put_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Encode `proposal` as one frame.
///
/// # Errors
///
/// A metadata command that does not serialize, a unit beyond the default
/// [`DecodeLimits`], or DERIVED work no member could derive (an unsupported
/// interpretation, or a record source naming no earlier node record put):
/// such a unit never reaches a log, where every member would refuse it.
pub fn encode_proposal(proposal: &RaftProposal) -> Result<Vec<u8>, FrameError> {
    encode_within(proposal, &DecodeLimits::DEFAULT)
}

/// Encode the operations of a unit a journal records outside a proposal, at
/// `commit_ts`: the frame [`decode_unit`] reads back.
///
/// # Errors
///
/// As [`encode_proposal`].
pub fn encode_unit(mutations: &[Mutation], commit_ts: Timestamp) -> Result<Vec<u8>, FrameError> {
    encode_within(
        UnitRef {
            id: 0,
            commit_ts: commit_ts.as_raw(),
            start_ts: 0,
            bypass_rate_limiter: false,
            mutations,
        },
        &DecodeLimits::DEFAULT,
    )
}

/// The operations of a frame, under the default bounds.
///
/// # Errors
///
/// As [`decode_proposal`].
pub fn decode_unit(frame: &[u8]) -> Result<Vec<Mutation>, FrameError> {
    decode_proposal(frame, &DecodeLimits::DEFAULT).map(|unit| unit.mutations)
}

/// A unit's header fields and operations, borrowed from wherever they live.
#[derive(Clone, Copy)]
struct UnitRef<'a> {
    id: u64,
    commit_ts: u64,
    start_ts: u64,
    bypass_rate_limiter: bool,
    mutations: &'a [Mutation],
}

impl<'a> From<&'a RaftProposal> for UnitRef<'a> {
    fn from(proposal: &'a RaftProposal) -> Self {
        Self {
            id: proposal.id.as_raw(),
            commit_ts: proposal.commit_ts.as_raw(),
            start_ts: proposal.start_ts.as_raw(),
            bypass_rate_limiter: proposal.bypass_rate_limiter,
            mutations: &proposal.mutations,
        }
    }
}

/// [`encode_proposal`], refusing a unit a decoder bound by `limits` would
/// refuse.
fn encode_within<'a>(
    unit: impl Into<UnitRef<'a>>,
    limits: &DecodeLimits,
) -> Result<Vec<u8>, FrameError> {
    let unit = unit.into();
    if unit.mutations.len() > limits.max_ops {
        return Err(FrameError::OverLimit {
            what: "operation count",
            limit: limits.max_ops,
        });
    }
    let mut out = Vec::with_capacity(64 + unit.mutations.len() * 48);
    out.push(FRAME_VERSION);
    put_varint(unit.id, &mut out);
    put_varint(unit.commit_ts, &mut out);
    put_varint(unit.start_ts, &mut out);
    out.push(if unit.bypass_rate_limiter {
        FLAG_BYPASS_RATE_LIMITER
    } else {
        0
    });
    put_varint(unit.mutations.len() as u64, &mut out);

    let mut encoder = Encoder::default();
    for (at, mutation) in unit.mutations.iter().enumerate() {
        if let Mutation::Derive(work) = mutation {
            check_derived(work, &unit.mutations[..at])?;
        }
        encoder.op(mutation, &mut out)?;
    }
    if encoder.expanded > limits.max_expanded_bytes {
        return Err(FrameError::OverLimit {
            what: "expanded size",
            limit: limits.max_expanded_bytes,
        });
    }
    let crc = crc32c::crc32c(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    if out.len() > limits.max_frame_bytes {
        return Err(FrameError::OverLimit {
            what: "frame size",
            limit: limits.max_frame_bytes,
        });
    }
    Ok(out)
}

/// Encoding state: the last key per partition and the literals a later
/// identical slice may refer to, whether borrowed from a mutation or encoded
/// here (a DERIVED binding or input), numbered in one sequence; and the bytes
/// a decoder expands the frame to, counted as its budget counts them.
#[derive(Default)]
struct Encoder<'a> {
    last_key: [&'a [u8]; PARTITIONS.len()],
    literals: FxHashMap<&'a [u8], u64>,
    encoded: FxHashMap<Vec<u8>, u64>,
    literal_count: u64,
    // Each addend is the length of a slice held in memory for this unit, so
    // the sum fits a usize.
    expanded: usize,
}

impl<'a> Encoder<'a> {
    fn op(&mut self, mutation: &'a Mutation, out: &mut Vec<u8>) -> Result<(), FrameError> {
        match mutation {
            Mutation::Put {
                partition,
                key,
                value,
            } => {
                let p = self.tag(KIND_PUT, *partition, out);
                self.key(p, key, out);
                self.bytes(value, out);
            }
            Mutation::Delete { partition, key } => {
                let p = self.tag(KIND_DELETE, *partition, out);
                self.key(p, key, out);
            }
            Mutation::Merge {
                partition,
                key,
                operand,
            } => {
                let p = self.tag(KIND_MERGE, *partition, out);
                self.key(p, key, out);
                self.bytes(operand, out);
            }
            Mutation::RemoveRange {
                partition,
                start,
                end,
            } => {
                let p = self.tag(KIND_REMOVE_RANGE, *partition, out);
                self.key(p, start, out);
                // The end is coded against the start it bounds.
                self.key(p, end, out);
            }
            Mutation::Command(command) => {
                out.push(KIND_COMMAND);
                let bytes =
                    rmp_serde::to_vec(command).map_err(|e| FrameError::Command(e.to_string()))?;
                self.expanded += bytes.len();
                put_varint(bytes.len() as u64, out);
                out.extend_from_slice(&bytes);
            }
            Mutation::Derive(work) => {
                out.push(KIND_DERIVE);
                // Every effect of one index in a unit carries the same
                // binding: the second and later are references.
                self.encoded_bytes(to_msgpack(&work.binding)?, out);
                put_varint(work.node_id, out);
                self.encoded_bytes(to_msgpack(&work.old)?, out);
                match &work.new {
                    DerivedSource::UnitRecord(ordinal) => {
                        out.push(SOURCE_UNIT_RECORD);
                        put_varint(u64::from(*ordinal), out);
                    }
                    DerivedSource::Values(values) => {
                        out.push(SOURCE_VALUES);
                        self.encoded_bytes(to_msgpack(values)?, out);
                    }
                }
            }
        }
        Ok(())
    }

    /// [`Self::bytes`] for a slice encoded here rather than borrowed.
    fn encoded_bytes(&mut self, bytes: Vec<u8>, out: &mut Vec<u8>) {
        self.expanded += bytes.len();
        if bytes.len() >= MIN_SHARED {
            if let Some(&index) = self.encoded.get(&bytes) {
                put_varint((index << 1) | 1, out);
                return;
            }
            put_varint((bytes.len() as u64) << 1, out);
            out.extend_from_slice(&bytes);
            self.encoded.insert(bytes, self.literal_count);
            self.literal_count += 1;
            return;
        }
        put_varint((bytes.len() as u64) << 1, out);
        out.extend_from_slice(&bytes);
    }

    fn tag(&self, kind: u8, partition: PartitionId, out: &mut Vec<u8>) -> usize {
        let p = partition_number(partition);
        out.push(kind | (p << 3));
        usize::from(p)
    }

    fn key(&mut self, partition: usize, key: &'a [u8], out: &mut Vec<u8>) {
        let last = self.last_key[partition];
        let shared = last.iter().zip(key).take_while(|(a, b)| a == b).count();
        self.expanded += key.len();
        put_varint(shared as u64, out);
        put_varint((key.len() - shared) as u64, out);
        out.extend_from_slice(&key[shared..]);
        self.last_key[partition] = key;
    }

    fn bytes(&mut self, bytes: &'a [u8], out: &mut Vec<u8>) {
        self.expanded += bytes.len();
        if bytes.len() >= MIN_SHARED {
            if let Some(&index) = self.literals.get(bytes) {
                put_varint((index << 1) | 1, out);
                return;
            }
            self.literals.insert(bytes, self.literal_count);
            self.literal_count += 1;
        }
        put_varint((bytes.len() as u64) << 1, out);
        out.extend_from_slice(bytes);
    }
}

/// Whether `proposal` encodes as a frame, checked without encoding it far
/// from the frame bound. A pipeline runs this before handing a unit to the
/// log: the log store encodes the unit and has no way to refuse it there.
///
/// # Errors
///
/// The [`FrameError`] [`encode_proposal`] would return.
pub fn check_proposal(proposal: &RaftProposal) -> Result<(), FrameError> {
    check_within(proposal, &DecodeLimits::DEFAULT)
}

/// Largest header: version, three varints, flags, the operation count and
/// the checksum.
const FRAME_HEADER_MAX: usize = 1 + 3 * 10 + 1 + 10 + 4;

/// Most bytes of tags, lengths and varints one operation adds to the bytes
/// it carries: a DERIVED operation's tag, three slice lengths, node id,
/// source tag and ordinal.
const MAX_OP_OVERHEAD: usize = 1 + 3 * 10 + 10 + 1 + 10;

/// [`check_proposal`] against `limits`.
fn check_within<'a>(unit: impl Into<UnitRef<'a>>, limits: &DecodeLimits) -> Result<(), FrameError> {
    let unit = unit.into();
    let ops = unit.mutations.len();
    if ops > limits.max_ops {
        return Err(FrameError::OverLimit {
            what: "operation count",
            limit: limits.max_ops,
        });
    }
    // Each addend is the length of a slice held in memory for this unit.
    let mut expanded = 0usize;
    for (at, mutation) in unit.mutations.iter().enumerate() {
        expanded += match mutation {
            Mutation::Put { key, value, .. } => key.len() + value.len(),
            Mutation::Delete { key, .. } => key.len(),
            Mutation::Merge { key, operand, .. } => key.len() + operand.len(),
            Mutation::RemoveRange { start, end, .. } => start.len() + end.len(),
            Mutation::Command(command) => msgpack_len(command).map_err(FrameError::Command)?,
            Mutation::Derive(work) => {
                check_derived(work, &unit.mutations[..at])?;
                let new = match &work.new {
                    DerivedSource::UnitRecord(_) => 0,
                    DerivedSource::Values(values) => {
                        msgpack_len(values).map_err(FrameError::Derive)?
                    }
                };
                msgpack_len(&work.binding).map_err(FrameError::Derive)?
                    + msgpack_len(&work.old).map_err(FrameError::Derive)?
                    + new
            }
        };
    }
    if expanded > limits.max_expanded_bytes {
        return Err(FrameError::OverLimit {
            what: "expanded size",
            limit: limits.max_expanded_bytes,
        });
    }
    // A frame writes each carried byte at most once, plus its header and
    // each operation's overhead: within the bound, it fits. Near the bound,
    // shared prefixes and slices decide, and only encoding tells.
    let most = ops
        .checked_mul(MAX_OP_OVERHEAD)
        .and_then(|overhead| overhead.checked_add(FRAME_HEADER_MAX))
        .and_then(|fixed| fixed.checked_add(expanded));
    match most {
        Some(most) if most <= limits.max_frame_bytes => Ok(()),
        _ => encode_within(unit, limits).map(drop),
    }
}

/// The length of `value`'s MessagePack encoding, counted without keeping it.
fn msgpack_len<T: serde::Serialize + ?Sized>(value: &T) -> Result<usize, String> {
    struct Counter(usize);
    // no-std: a counting `rmp::encode::RmpWrite` in place of `std::io::Write`
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 += bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter(0);
    rmp_serde::encode::write(&mut counter, value).map_err(|e| e.to_string())?;
    Ok(counter.0)
}

/// Refuse DERIVED work no member could derive: an interpretation this build
/// does not support, or a record source that does not name a node record put
/// among the `earlier` operations of its unit.
fn check_derived(work: &DerivedIndexWork, earlier: &[Mutation]) -> Result<(), FrameError> {
    work.binding.interpretation.check_supported()?;
    match work.new {
        DerivedSource::UnitRecord(ordinal) if !names_record(earlier, ordinal) => {
            Err(FrameError::BadSource { ordinal })
        }
        _ => Ok(()),
    }
}

/// Whether the operation at `ordinal` of `earlier` puts a node record.
fn names_record(earlier: &[Mutation], ordinal: u32) -> bool {
    matches!(
        earlier.get(ordinal as usize),
        Some(Mutation::Put {
            partition: PartitionId::Node,
            ..
        })
    )
}

fn to_msgpack<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, FrameError> {
    rmp_serde::to_vec(value).map_err(|e| FrameError::Derive(e.to_string()))
}

/// Decode a frame into the proposal it encodes.
///
/// # Errors
///
/// [`FrameError`] for bytes this build does not accept as a frame, or that
/// exceed `limits`.
pub fn decode_proposal(frame: &[u8], limits: &DecodeLimits) -> Result<RaftProposal, FrameError> {
    if frame.len() > limits.max_frame_bytes {
        return Err(FrameError::OverLimit {
            what: "frame size",
            limit: limits.max_frame_bytes,
        });
    }
    let Some(body_len) = frame.len().checked_sub(4) else {
        return Err(FrameError::Truncated { at: 0 });
    };
    let (body, crc) = frame.split_at(body_len);
    let mut stored = [0u8; 4];
    stored.copy_from_slice(crc);
    if crc32c::crc32c(body) != u32::from_le_bytes(stored) {
        return Err(FrameError::Checksum);
    }

    let mut r = Reader { bytes: body, at: 0 };
    let version = r.u8()?;
    if version != FRAME_VERSION {
        return Err(FrameError::UnsupportedVersion(version));
    }
    let id = ProposalId::from_raw(r.varint()?);
    let commit_ts = Timestamp::from_raw(r.varint()?);
    let start_ts = Timestamp::from_raw(r.varint()?);
    let flags = r.u8()?;
    if flags & !FLAG_BYPASS_RATE_LIMITER != 0 {
        return Err(FrameError::UnknownFlags(flags));
    }
    let op_count = r.len_at_most(limits.max_ops, "operation count")?;
    // Every operation takes at least two bytes: no count the body cannot hold.
    if op_count > r.remaining() / 2 {
        return Err(FrameError::Truncated { at: r.at });
    }

    let mut decoder = Decoder {
        reader: r,
        last_key: Default::default(),
        literals: Vec::new(),
        expanded: 0,
        limit: limits.max_expanded_bytes,
    };
    let mut mutations = Vec::with_capacity(op_count);
    for _ in 0..op_count {
        let op = decoder.op(&mutations)?;
        mutations.push(op);
    }
    let left = decoder.reader.remaining();
    if left != 0 {
        return Err(FrameError::Trailing(left));
    }
    Ok(RaftProposal {
        id,
        mutations,
        commit_ts,
        start_ts,
        bypass_rate_limiter: flags & FLAG_BYPASS_RATE_LIMITER != 0,
    })
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.at
    }

    fn u8(&mut self) -> Result<u8, FrameError> {
        let b = *self
            .bytes
            .get(self.at)
            .ok_or(FrameError::Truncated { at: self.at })?;
        self.at += 1;
        Ok(b)
    }

    fn varint(&mut self) -> Result<u64, FrameError> {
        let start = self.at;
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.u8()?;
            let low = u64::from(b & 0x7F);
            // The tenth byte may carry only the top bit of a u64.
            if shift == 63 && low > 1 {
                return Err(FrameError::VarintOverflow { at: start });
            }
            value |= low << shift;
            if b & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(FrameError::VarintOverflow { at: start })
    }

    /// A varint used as a length or count, refused above `limit`.
    fn len_at_most(&mut self, limit: usize, what: &'static str) -> Result<usize, FrameError> {
        let value = self.varint()?;
        match usize::try_from(value) {
            Ok(n) if n <= limit => Ok(n),
            _ => Err(FrameError::OverLimit { what, limit }),
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], FrameError> {
        if n > self.remaining() {
            return Err(FrameError::Truncated { at: self.at });
        }
        let slice = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(slice)
    }
}

struct Decoder<'a> {
    reader: Reader<'a>,
    last_key: [Vec<u8>; PARTITIONS.len()],
    literals: Vec<&'a [u8]>,
    expanded: usize,
    limit: usize,
}

impl<'a> Decoder<'a> {
    /// Count `n` more expanded bytes against the budget, before they are
    /// allocated.
    fn spend(&mut self, n: usize) -> Result<(), FrameError> {
        match self.expanded.checked_add(n) {
            Some(total) if total <= self.limit => {
                self.expanded = total;
                Ok(())
            }
            _ => Err(FrameError::OverLimit {
                what: "expanded size",
                limit: self.limit,
            }),
        }
    }

    fn op(&mut self, earlier: &[Mutation]) -> Result<Mutation, FrameError> {
        let tag = self.reader.u8()?;
        let kind = tag & KIND_MASK;
        if kind == KIND_DERIVE {
            if tag != KIND_DERIVE {
                return Err(FrameError::UnknownTag(tag));
            }
            return self.derive(earlier).map(Mutation::Derive);
        }
        if kind == KIND_COMMAND {
            if tag != KIND_COMMAND {
                return Err(FrameError::UnknownTag(tag));
            }
            let len = self.reader.len_at_most(self.limit, "command size")?;
            self.spend(len)?;
            let bytes = self.reader.take(len)?;
            let command =
                rmp_serde::from_slice(bytes).map_err(|e| FrameError::Command(e.to_string()))?;
            return Ok(Mutation::Command(command));
        }
        let p = usize::from(tag >> 3);
        let partition = *PARTITIONS.get(p).ok_or(FrameError::UnknownTag(tag))?;
        match kind {
            KIND_PUT => Ok(Mutation::Put {
                partition,
                key: self.key(p)?,
                value: self.bytes()?,
            }),
            KIND_DELETE => Ok(Mutation::Delete {
                partition,
                key: self.key(p)?,
            }),
            KIND_MERGE => Ok(Mutation::Merge {
                partition,
                key: self.key(p)?,
                operand: self.bytes()?,
            }),
            KIND_REMOVE_RANGE => Ok(Mutation::RemoveRange {
                partition,
                start: self.key(p)?,
                end: self.key(p)?,
            }),
            _ => Err(FrameError::UnknownTag(tag)),
        }
    }

    fn key(&mut self, partition: usize) -> Result<Vec<u8>, FrameError> {
        let shared = self.reader.len_at_most(self.limit, "key size")?;
        let available = self.last_key[partition].len();
        if shared > available {
            return Err(FrameError::BadPrefix { shared, available });
        }
        let suffix_len = self.reader.len_at_most(self.limit, "key size")?;
        let suffix = self.reader.take(suffix_len)?;
        self.spend(shared + suffix_len)?;
        let mut key = Vec::with_capacity(shared + suffix_len);
        key.extend_from_slice(&self.last_key[partition][..shared]);
        key.extend_from_slice(suffix);
        self.last_key[partition].clone_from(&key);
        Ok(key)
    }

    fn derive(&mut self, earlier: &[Mutation]) -> Result<DerivedIndexWork, FrameError> {
        let binding: IndexBinding = from_msgpack(self.slice()?)?;
        binding.interpretation.check_supported()?;
        let node_id = self.reader.varint()?;
        let old = from_msgpack(self.slice()?)?;
        let new = match self.reader.u8()? {
            SOURCE_UNIT_RECORD => {
                let raw = self.reader.varint()?;
                let ordinal =
                    u32::try_from(raw).map_err(|_| FrameError::BadSource { ordinal: u32::MAX })?;
                if !names_record(earlier, ordinal) {
                    return Err(FrameError::BadSource { ordinal });
                }
                DerivedSource::UnitRecord(ordinal)
            }
            SOURCE_VALUES => DerivedSource::Values(from_msgpack(self.slice()?)?),
            other => return Err(FrameError::UnknownTag(other)),
        };
        Ok(DerivedIndexWork {
            binding,
            node_id,
            old,
            new,
        })
    }

    fn bytes(&mut self) -> Result<Vec<u8>, FrameError> {
        self.slice().map(<[u8]>::to_vec)
    }

    /// A literal or a reference to one, counted against the budget.
    fn slice(&mut self) -> Result<&'a [u8], FrameError> {
        let v = self.reader.varint()?;
        if v & 1 == 1 {
            let index = usize::try_from(v >> 1).unwrap_or(usize::MAX);
            let count = self.literals.len();
            let slice = *self
                .literals
                .get(index)
                .ok_or(FrameError::BadReference { index, count })?;
            self.spend(slice.len())?;
            return Ok(slice);
        }
        let len = usize::try_from(v >> 1).map_err(|_| FrameError::OverLimit {
            what: "value size",
            limit: self.limit,
        })?;
        let slice = self.reader.take(len)?;
        self.spend(len)?;
        if len >= MIN_SHARED {
            self.literals.push(slice);
        }
        Ok(slice)
    }
}

fn from_msgpack<'a, T: serde::Deserialize<'a>>(bytes: &'a [u8]) -> Result<T, FrameError> {
    rmp_serde::from_slice(bytes).map_err(|e| FrameError::Derive(e.to_string()))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
