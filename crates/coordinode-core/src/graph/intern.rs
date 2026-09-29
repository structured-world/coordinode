//! Field name interning: bidirectional mapping between property names and u32 IDs.
//!
//! Property graph storage repeats field names across millions of nodes.
//! Interning replaces string keys with varint-encoded integer IDs, achieving
//! ~80% reduction in property key storage.
//!
//! A binding, once published, is immutable: stored data carries only the id,
//! so the id's meaning must never change and never be reassigned. The
//! authoritative dictionary lives in the Schema partition under
//! [`FIELD_NAME_KEY_PREFIX`] and [`FIELD_ID_KEY_PREFIX`], one write-once record
//! per direction per binding; new bindings are assigned only by the ordered
//! application of a registration (see [`FieldRegistrar`]).
//!
//! ## Varint encoding
//!
//! IDs 0-127 → 1 byte, IDs 128-16383 → 2 bytes, IDs 16384+ → up to 5 bytes.
//! Uses standard unsigned LEB128 (same as protobuf varint).

use alloc::sync::Arc;
// no-std: hashbrown::HashMap
use std::collections::HashMap;

/// Schema key prefix of the name → id record of a binding.
pub const FIELD_NAME_KEY_PREFIX: &[u8] = b"ids:field:n:";

/// Schema key prefix of the id → name record of a binding.
pub const FIELD_ID_KEY_PREFIX: &[u8] = b"ids:field:i:";

/// Longest field name, in bytes, a binding accepts.
pub const MAX_FIELD_NAME_BYTES: usize = u16::MAX as usize;

/// Most names one registration carries.
pub const MAX_REGISTRATION_BATCH: usize = 4096;

/// Schema key of the name → id record for `name`.
pub fn field_name_key(name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(FIELD_NAME_KEY_PREFIX.len() + name.len());
    key.extend_from_slice(FIELD_NAME_KEY_PREFIX);
    key.extend_from_slice(name.as_bytes());
    key
}

/// Schema key of the id → name record for `id`. Big-endian, so the records
/// sort by id and the last one holds the allocation frontier.
pub fn field_id_key(id: u32) -> Vec<u8> {
    let mut key = Vec::with_capacity(FIELD_ID_KEY_PREFIX.len() + 4);
    key.extend_from_slice(FIELD_ID_KEY_PREFIX);
    key.extend_from_slice(&id.to_be_bytes());
    key
}

/// The id an id → name record key names, or `None` for any other key.
pub fn decode_field_id_key(key: &[u8]) -> Option<u32> {
    let raw: [u8; 4] = key.strip_prefix(FIELD_ID_KEY_PREFIX)?.try_into().ok()?;
    Some(u32::from_be_bytes(raw))
}

/// The value of a name → id record.
pub fn encode_field_id(id: u32) -> [u8; 4] {
    id.to_be_bytes()
}

/// The id a name → id record holds, or `None` when it is not one.
pub fn decode_field_id(value: &[u8]) -> Option<u32> {
    Some(u32::from_be_bytes(value.try_into().ok()?))
}

/// Why a set of bindings is not a dictionary, or a registration failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DictionaryError {
    /// Id 0 is reserved and never names a field.
    #[error("field id 0 is reserved, but {name:?} is bound to it")]
    ReservedId {
        /// The name bound to the reserved id.
        name: String,
    },
    /// One name, two ids.
    #[error("field {name:?} is bound to both {first} and {second}")]
    NameConflict {
        /// The name bound twice.
        name: String,
        /// The id it had.
        first: u32,
        /// The other id it was given.
        second: u32,
    },
    /// One id, two names.
    #[error("field id {id} is bound to both {first:?} and {second:?}")]
    IdConflict {
        /// The id bound twice.
        id: u32,
        /// The name it had.
        first: String,
        /// The other name it was given.
        second: String,
    },
    /// Stored or transferred dictionary bytes that do not decode.
    #[error("malformed field dictionary: {0}")]
    Malformed(String),
    /// A name longer than [`MAX_FIELD_NAME_BYTES`].
    #[error("field name of {len} bytes is longer than the {max}-byte limit")]
    NameTooLong {
        /// Its length in bytes.
        len: usize,
        /// The limit.
        max: usize,
    },
    /// More names in one registration than [`MAX_REGISTRATION_BATCH`].
    #[error("registration of {len} names is larger than the {max}-name limit")]
    BatchTooLarge {
        /// Names asked for.
        len: usize,
        /// The limit.
        max: usize,
    },
    /// The 32-bit id space has no room for the names asked for.
    #[error("the field id space is exhausted")]
    Exhausted,
    /// The authority did not publish the bindings asked for: the proposal
    /// failed, or its application rejected them.
    #[error("field registration failed: {0}")]
    Registration(String),
    /// A stored id no binding names: data the dictionary cannot interpret.
    #[error("field id {0} is stored but has no binding")]
    Unbound(u32),
}

/// Check that a registration may carry `names` at all.
///
/// # Errors
///
/// [`DictionaryError::BatchTooLarge`] or [`DictionaryError::NameTooLong`].
pub fn validate_registration(names: &[&str]) -> Result<(), DictionaryError> {
    if names.len() > MAX_REGISTRATION_BATCH {
        return Err(DictionaryError::BatchTooLarge {
            len: names.len(),
            max: MAX_REGISTRATION_BATCH,
        });
    }
    match names.iter().find(|n| n.len() > MAX_FIELD_NAME_BYTES) {
        Some(n) => Err(DictionaryError::NameTooLong {
            len: n.len(),
            max: MAX_FIELD_NAME_BYTES,
        }),
        None => Ok(()),
    }
}

/// Bindings published together and never changed afterwards.
#[derive(Debug)]
struct Published {
    name_to_id: HashMap<String, u32>,
    ids: IdTable,
    /// The largest bound id; 0 when nothing is bound.
    frontier: u32,
}

/// Id → name. Registration hands ids out densely, so the table is normally
/// an array indexed by id (the per-property decode path). Ids adopted from
/// elsewhere can be sparse; indexing those would allocate for every gap.
#[derive(Debug)]
enum IdTable {
    /// Indexed by id; `None` for an id no binding names.
    Dense(Vec<Option<String>>),
    Sparse(HashMap<u32, String>),
}

/// Gap an array index may carry before a sparse map is cheaper.
const DENSE_SLACK: usize = 1024;

impl Published {
    fn empty() -> Self {
        Self {
            name_to_id: HashMap::new(),
            ids: IdTable::Dense(Vec::new()),
            frontier: 0,
        }
    }

    /// Build from validated bindings.
    fn build(name_to_id: HashMap<String, u32>, mut id_to_name: HashMap<u32, String>) -> Self {
        let frontier = id_to_name.keys().copied().max().unwrap_or(0);
        let slots = frontier as usize + 1;
        let ids = if slots <= 2 * id_to_name.len() + DENSE_SLACK {
            let mut dense = vec![None; slots];
            for (id, name) in id_to_name.drain() {
                dense[id as usize] = Some(name);
            }
            IdTable::Dense(dense)
        } else {
            IdTable::Sparse(id_to_name)
        };
        Self {
            name_to_id,
            ids,
            frontier,
        }
    }

    #[inline]
    fn resolve(&self, id: u32) -> Option<&str> {
        match &self.ids {
            IdTable::Dense(dense) => dense.get(id as usize)?.as_deref(),
            IdTable::Sparse(sparse) => sparse.get(&id).map(String::as_str),
        }
    }

    /// Every binding, id → name, as a map to extend.
    fn id_map(&self) -> HashMap<u32, String> {
        match &self.ids {
            IdTable::Dense(dense) => dense
                .iter()
                .enumerate()
                .filter_map(|(id, name)| Some((id as u32, name.clone()?)))
                .collect(),
            IdTable::Sparse(sparse) => sparse.clone(),
        }
    }
}

/// Bidirectional field name ↔ u32 ID mapping.
///
/// A handle holds an immutable published view, shared by every clone, plus
/// the bindings added through this handle alone. Cloning is a reference-count
/// bump, so each query takes its own view without copying the dictionary and
/// without holding a lock while it runs. IDs start at 1 (0 is reserved).
#[derive(Debug, Clone)]
pub struct FieldInterner {
    published: Arc<Published>,
    local_names: HashMap<String, u32>,
    local_ids: HashMap<u32, String>,
    /// Next id [`Self::intern`] hands out, above every binding this handle
    /// knows; `None` once the id space is used up.
    next_id: Option<u32>,
}

impl FieldInterner {
    /// Reserved ID representing "no field" / uninitialized.
    pub const RESERVED_ID: u32 = 0;

    /// Create a new empty interner.
    pub fn new() -> Self {
        Self {
            published: Arc::new(Published::empty()),
            local_names: HashMap::new(),
            local_ids: HashMap::new(),
            next_id: Some(1),
        }
    }

    /// A dictionary holding exactly `bindings`.
    ///
    /// # Errors
    ///
    /// Id 0, a name bound twice, or an id bound twice.
    pub fn from_bindings(
        bindings: impl IntoIterator<Item = (String, u32)>,
    ) -> Result<Self, DictionaryError> {
        Self::new().extended(bindings)
    }

    /// This view with `bindings` added, as a new published view shared by
    /// its clones. Bindings this view already holds are accepted again.
    ///
    /// # Errors
    ///
    /// A binding that contradicts one this view holds, or id 0.
    pub fn extended(
        &self,
        bindings: impl IntoIterator<Item = (String, u32)>,
    ) -> Result<Self, DictionaryError> {
        let mut name_to_id = self.published.name_to_id.clone();
        let mut id_to_name = self.published.id_map();
        for (name, &id) in &self.local_names {
            insert_checked(&mut name_to_id, &mut id_to_name, name.clone(), id)?;
        }
        for (name, id) in bindings {
            insert_checked(&mut name_to_id, &mut id_to_name, name, id)?;
        }
        let published = Published::build(name_to_id, id_to_name);
        let next_id = published.frontier.checked_add(1);
        Ok(Self {
            published: Arc::new(published),
            local_names: HashMap::new(),
            local_ids: HashMap::new(),
            next_id,
        })
    }

    /// Record a binding the authority published, in this handle only.
    ///
    /// # Errors
    ///
    /// A binding that contradicts one this handle holds, or id 0.
    pub fn insert_binding(&mut self, name: &str, id: u32) -> Result<(), DictionaryError> {
        if id == Self::RESERVED_ID {
            return Err(DictionaryError::ReservedId {
                name: name.to_owned(),
            });
        }
        match self.lookup(name) {
            Some(existing) if existing == id => return Ok(()),
            Some(existing) => {
                return Err(DictionaryError::NameConflict {
                    name: name.to_owned(),
                    first: existing,
                    second: id,
                });
            }
            None => {}
        }
        if let Some(existing) = self.resolve(id) {
            return Err(DictionaryError::IdConflict {
                id,
                first: existing.to_owned(),
                second: name.to_owned(),
            });
        }
        self.local_names.insert(name.to_owned(), id);
        self.local_ids.insert(id, name.to_owned());
        if self.next_id.is_some_and(|next| id >= next) {
            self.next_id = id.checked_add(1);
        }
        Ok(())
    }

    /// Intern a field name in this handle alone, returning its ID.
    ///
    /// For a dictionary this handle is the only authority of (a test, an
    /// in-memory tool). A database registers names through its
    /// [`FieldRegistrar`] instead, so every binding it uses is durable.
    ///
    /// # Panics
    ///
    /// When all 2^32 - 1 ids are taken, which needs more names than memory.
    #[expect(
        clippy::expect_used,
        reason = "2^32 - 1 distinct names in one in-memory table exhaust memory first"
    )]
    pub fn intern(&mut self, name: &str) -> u32 {
        if let Some(id) = self.lookup(name) {
            return id;
        }
        let id = self
            .next_id
            .expect("more distinct field names than a 32-bit id space holds");
        self.next_id = id.checked_add(1);
        self.local_names.insert(name.to_owned(), id);
        self.local_ids.insert(id, name.to_owned());
        id
    }

    /// Look up an ID without interning. Returns `None` if not interned.
    #[inline]
    pub fn lookup(&self, name: &str) -> Option<u32> {
        if let Some(&id) = self.published.name_to_id.get(name) {
            return Some(id);
        }
        if self.local_names.is_empty() {
            return None;
        }
        self.local_names.get(name).copied()
    }

    /// Resolve an ID back to its field name.
    ///
    /// Returns `None` if the ID is out of range or is the reserved ID (0).
    #[inline]
    pub fn resolve(&self, id: u32) -> Option<&str> {
        if id == Self::RESERVED_ID {
            return None;
        }
        if let Some(name) = self.published.resolve(id) {
            return Some(name);
        }
        self.local_ids.get(&id).map(String::as_str)
    }

    /// The largest id this handle holds a binding for; 0 when it holds none.
    pub fn frontier(&self) -> u32 {
        self.local_ids
            .keys()
            .copied()
            .fold(self.published.frontier, u32::max)
    }

    /// Number of interned field names (excluding reserved ID 0).
    pub fn len(&self) -> usize {
        self.published.name_to_id.len() + self.local_names.len()
    }

    /// Whether the interner has no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterator over all (name, id) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> {
        self.published
            .name_to_id
            .iter()
            .chain(self.local_names.iter())
            .map(|(name, &id)| (name.as_str(), id))
    }

    /// Serialize the dictionary to bytes, for a backup that carries it.
    ///
    /// Format: count (u32 LE) followed by (id: u32 LE, name_len: u16 LE, name
    /// bytes) per binding, in id order.
    ///
    /// # Errors
    ///
    /// [`DictionaryError::NameTooLong`] for a name only a local
    /// [`Self::intern`] could have admitted.
    pub fn to_bytes(&self) -> Result<Vec<u8>, DictionaryError> {
        let mut pairs: Vec<(&str, u32)> = self.iter().collect();
        pairs.sort_unstable_by_key(|&(_, id)| id);
        let mut buf = Vec::with_capacity(4 + pairs.len() * 16);
        // Ids are distinct u32 values other than 0, so the count fits.
        buf.extend_from_slice(&(pairs.len() as u32).to_le_bytes());
        for (name, id) in pairs {
            let name_len = u16::try_from(name.len()).map_err(|_| DictionaryError::NameTooLong {
                len: name.len(),
                max: MAX_FIELD_NAME_BYTES,
            })?;
            buf.extend_from_slice(&id.to_le_bytes());
            buf.extend_from_slice(&name_len.to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
        }
        Ok(buf)
    }

    /// Deserialize a dictionary written by [`Self::to_bytes`].
    ///
    /// # Errors
    ///
    /// Truncated or trailing bytes, a name that is not UTF-8, id 0, or a
    /// name or id bound twice.
    pub fn from_bytes(data: &[u8]) -> Result<Self, DictionaryError> {
        let malformed = |what: &str| DictionaryError::Malformed(what.to_owned());
        let (count, mut rest) = data
            .split_first_chunk::<4>()
            .ok_or_else(|| malformed("no binding count"))?;
        let count = u32::from_le_bytes(*count) as usize;
        // Each binding takes at least 6 bytes, so a count the data cannot hold
        // is refused before anything is allocated for it.
        if count > rest.len() / 6 {
            return Err(malformed("binding count exceeds the data"));
        }
        let mut bindings = Vec::with_capacity(count);
        for _ in 0..count {
            let (id, after) = rest
                .split_first_chunk::<4>()
                .ok_or_else(|| malformed("truncated binding id"))?;
            let (len, after) = after
                .split_first_chunk::<2>()
                .ok_or_else(|| malformed("truncated name length"))?;
            let len = usize::from(u16::from_le_bytes(*len));
            if after.len() < len {
                return Err(malformed("truncated name"));
            }
            let (name, after) = after.split_at(len);
            let name = std::str::from_utf8(name).map_err(|_| malformed("name is not UTF-8"))?;
            bindings.push((name.to_owned(), u32::from_le_bytes(*id)));
            rest = after;
        }
        if !rest.is_empty() {
            return Err(malformed("bytes past the last binding"));
        }
        Self::from_bindings(bindings)
    }
}

/// Add one binding to a table being built, refusing any contradiction.
fn insert_checked(
    name_to_id: &mut HashMap<String, u32>,
    id_to_name: &mut HashMap<u32, String>,
    name: String,
    id: u32,
) -> Result<(), DictionaryError> {
    if id == FieldInterner::RESERVED_ID {
        return Err(DictionaryError::ReservedId { name });
    }
    if let Some(existing) = id_to_name.get(&id) {
        if *existing == name {
            return Ok(());
        }
        return Err(DictionaryError::IdConflict {
            id,
            first: existing.clone(),
            second: name,
        });
    }
    if let Some(&existing) = name_to_id.get(&name) {
        return Err(DictionaryError::NameConflict {
            name,
            first: existing,
            second: id,
        });
    }
    id_to_name.insert(id, name.clone());
    name_to_id.insert(name, id);
    Ok(())
}

impl Default for FieldInterner {
    fn default() -> Self {
        Self::new()
    }
}

/// The authority that publishes field bindings.
///
/// A database implements this over its ordered write path: a registration
/// is a proposal whose application assigns the ids, and a call returns only
/// once the bindings are durable, so data encoded with them never outlives
/// their meaning.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot publish field bindings",
    label = "not a field registrar",
    note = "an embedded `Database` provides one through `Database::field_registrar`"
)]
pub trait FieldRegistrar: Send + Sync {
    /// The ids of `names`, in order, registering the ones that have none.
    /// Every returned binding is durable when this returns.
    ///
    /// # Errors
    ///
    /// The batch is invalid, the id space is exhausted, or the registration
    /// could not be published.
    fn register(&self, names: &[&str]) -> Result<Vec<u32>, DictionaryError>;

    /// Publish exactly `bindings`, ids included, for data already encoded
    /// with them (a restore). Bindings the dictionary already holds are
    /// accepted again.
    ///
    /// # Errors
    ///
    /// A binding contradicts one already published, or the registration
    /// could not be published.
    fn adopt(&self, bindings: &FieldInterner) -> Result<(), DictionaryError>;

    /// The current verified view: every binding durable so far.
    ///
    /// # Errors
    ///
    /// The stored dictionary could not be read or is inconsistent.
    fn view(&self) -> Result<FieldInterner, DictionaryError>;
}

/// Encode a u32 as unsigned LEB128 varint.
///
/// Returns number of bytes written (1-5).
pub fn encode_varint(value: u32, buf: &mut [u8; 5]) -> usize {
    let mut v = value;
    let mut i = 0;
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            buf[i] = byte;
            return i + 1;
        }
        buf[i] = byte | 0x80;
        i += 1;
    }
}

/// Decode an unsigned LEB128 varint from bytes.
///
/// Returns `(value, bytes_consumed)` or `None` if malformed.
pub fn decode_varint(data: &[u8]) -> Option<(u32, usize)> {
    let mut result: u32 = 0;
    let mut shift = 0;
    for (i, &byte) in data.iter().enumerate() {
        if shift >= 35 {
            return None; // overflow
        }
        result |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            return Some((result, i + 1));
        }
        shift += 7;
    }
    None // incomplete
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
