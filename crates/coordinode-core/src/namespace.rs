//! The namespace a piece of data belongs to.
//!
//! A namespace is the tenancy unit of a deployment: its data lives in a set
//! of trees of its own, and keys carry no namespace. What tells a write's
//! namespace is the unit it travels in, which names it before the writes it
//! holds of that namespace.

use alloc::string::String;
use alloc::vec::Vec;

/// Bytes of the token a namespace creation carries.
pub const NAMESPACE_TOKEN_LEN: usize = 16;

/// The name of the one namespace a store of one namespace creates for
/// itself when it is initialized as its own store.
pub const DEFAULT_NAMESPACE: &str = "default";

/// The longest name a namespace may have, in bytes.
pub const MAX_NAMESPACE_NAME_LEN: usize = 63;

/// The deployment catalog's record of the next namespace id it hands out.
/// Ids start at 1 and are never handed out twice.
pub const NAMESPACE_ALLOCATOR_KEY: &[u8] = b"ns:next";

/// Prefix of the catalog's record of each namespace, by id.
pub const NAMESPACE_RECORD_PREFIX: &[u8] = b"ns:id:";

/// Prefix of the catalog's index of live namespaces by name.
pub const NAMESPACE_NAME_PREFIX: &[u8] = b"ns:name:";

/// The catalog key of namespace `id`'s record: ids in big-endian order, so
/// the records list by id.
#[must_use]
pub fn namespace_record_key(id: NamespaceId) -> Vec<u8> {
    let mut key = Vec::with_capacity(NAMESPACE_RECORD_PREFIX.len() + 8);
    key.extend_from_slice(NAMESPACE_RECORD_PREFIX);
    key.extend_from_slice(&id.as_raw().to_be_bytes());
    key
}

/// The id a record key of [`namespace_record_key`] names; `None` for any
/// other key.
#[must_use]
pub fn decode_namespace_record_key(key: &[u8]) -> Option<NamespaceId> {
    let id = key.strip_prefix(NAMESPACE_RECORD_PREFIX)?;
    Some(NamespaceId(u64::from_be_bytes(id.try_into().ok()?)))
}

/// Whether `key` of the deployment's records is a record of the namespace
/// catalog: its id allocator, a namespace's record or its name index.
#[must_use]
pub fn is_namespace_catalog_key(key: &[u8]) -> bool {
    key == NAMESPACE_ALLOCATOR_KEY
        || key.starts_with(NAMESPACE_RECORD_PREFIX)
        || key.starts_with(NAMESPACE_NAME_PREFIX)
}

/// The catalog key naming the live namespace called `name`.
#[must_use]
pub fn namespace_name_key(name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(NAMESPACE_NAME_PREFIX.len() + name.len());
    key.extend_from_slice(NAMESPACE_NAME_PREFIX);
    key.extend_from_slice(name.as_bytes());
    key
}

/// Why a name is not one a namespace may have.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NamespaceNameError {
    /// The name is empty.
    #[error("a namespace name cannot be empty")]
    Empty,
    /// The name is longer than [`MAX_NAMESPACE_NAME_LEN`] bytes.
    #[error("a namespace name holds at most {MAX_NAMESPACE_NAME_LEN} bytes, not {len}")]
    TooLong {
        /// Its length in bytes.
        len: usize,
    },
    /// The name does not start with a lowercase ASCII letter.
    #[error("a namespace name starts with a lowercase letter, not {first:?}")]
    Start {
        /// Its first character.
        first: char,
    },
    /// The name holds a character other than a lowercase ASCII letter, a
    /// digit, `_` or `-`.
    #[error("a namespace name holds lowercase letters, digits, '_' and '-', not {found:?}")]
    Character {
        /// The first character not admitted.
        found: char,
    },
}

/// Admit `name` as a namespace's name: a lowercase ASCII letter, then
/// lowercase letters, digits, `_` and `-`, at most [`MAX_NAMESPACE_NAME_LEN`]
/// bytes. One spelling per name, so two names never differ only in case.
///
/// # Errors
///
/// The [`NamespaceNameError`] the name breaks.
pub fn validate_namespace_name(name: &str) -> Result<(), NamespaceNameError> {
    let Some(first) = name.chars().next() else {
        return Err(NamespaceNameError::Empty);
    };
    if name.len() > MAX_NAMESPACE_NAME_LEN {
        return Err(NamespaceNameError::TooLong { len: name.len() });
    }
    if !first.is_ascii_lowercase() {
        return Err(NamespaceNameError::Start { first });
    }
    match name
        .chars()
        .find(|c| !(c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_' || *c == '-'))
    {
        Some(found) => Err(NamespaceNameError::Character { found }),
        None => Ok(()),
    }
}

/// Where a namespace stands in the deployment's catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamespaceState {
    /// It serves its data.
    Live,
    /// It was dropped: nothing reads or writes it, and its trees are removed;
    /// the record stays until they are gone, so its id is never handed out
    /// again and its leftovers are known for what they are.
    Dropping,
}

/// A codec for a namespace's data blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BlockCodec {
    /// Uncompressed blocks.
    None,
    /// LZ4: fast decode, moderate ratio.
    Lz4,
    /// Zstd at the given level.
    Zstd(i32),
}

/// The zstd levels a table can record: the negative fast levels, 0 for the
/// codec's default, and 1-22.
pub const ZSTD_LEVELS: core::ops::RangeInclusive<i32> = -128..=22;

/// The zstd level `zstd` alone names: the codec's own default, the knee of
/// its ratio and speed.
pub const DEFAULT_ZSTD_LEVEL: i32 = 3;

impl BlockCodec {
    /// The codec `text` names: `none`, `lz4`, `zstd` (at
    /// [`DEFAULT_ZSTD_LEVEL`]) or `zstd:<level>`, case-insensitive.
    ///
    /// # Errors
    ///
    /// Why `text` names no codec, for a person reading the refusal.
    pub fn parse(text: &str) -> Result<Self, String> {
        let lower = text.trim().to_ascii_lowercase();
        match lower.as_str() {
            "none" => return Ok(Self::None),
            "lz4" => return Ok(Self::Lz4),
            "zstd" => return Ok(Self::Zstd(DEFAULT_ZSTD_LEVEL)),
            _ => {}
        }
        let Some(level) = lower.strip_prefix("zstd:") else {
            return Err(alloc::format!(
                "'{text}' is not a codec: none, lz4, zstd or zstd:<level>"
            ));
        };
        let level: i32 = level
            .parse()
            .map_err(|_| alloc::format!("'{text}' names no zstd level"))?;
        if !ZSTD_LEVELS.contains(&level) {
            return Err(alloc::format!(
                "zstd level {level} is outside {}..={}",
                ZSTD_LEVELS.start(),
                ZSTD_LEVELS.end()
            ));
        }
        Ok(Self::Zstd(level))
    }
}

impl core::fmt::Display for BlockCodec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Lz4 => f.write_str("lz4"),
            Self::Zstd(level) => write!(f, "zstd:{level}"),
        }
    }
}

/// A namespace's data-block storage settings. Each unset field inherits the
/// deployment's: the hot and cold bands its codec per partition, the
/// dictionary switch its own.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NamespaceStorage {
    /// The codec of the hot levels, below the deployment's cold threshold.
    pub compression: Option<BlockCodec>,
    /// The codec of the cold levels, at and above the threshold.
    pub cold_compression: Option<BlockCodec>,
    /// Whether eligible zstd output uses the per-label dictionaries.
    pub dictionaries: Option<bool>,
}

impl NamespaceStorage {
    /// These settings with every field `change` names replaced by its value,
    /// the others kept: what `ALTER NAMESPACE ... SET STORAGE` makes.
    #[must_use]
    pub fn with(self, change: Self) -> Self {
        Self {
            compression: change.compression.or(self.compression),
            cold_compression: change.cold_compression.or(self.cold_compression),
            dictionaries: change.dictionaries.or(self.dictionaries),
        }
    }

    /// The settings `pairs` name, as a statement spells them: `compression`
    /// and `cold_compression` take a codec ([`BlockCodec::parse`]),
    /// `dictionaries` a boolean.
    ///
    /// # Errors
    ///
    /// An unknown setting, a setting named twice, or a value it does not
    /// take.
    pub fn from_settings<'a>(
        pairs: impl IntoIterator<Item = (&'a str, StorageSetting<'a>)>,
    ) -> Result<Self, String> {
        let mut out = Self::default();
        for (key, value) in pairs {
            let twice = || alloc::format!("storage setting '{key}' is named twice");
            match (key.to_ascii_lowercase().as_str(), value) {
                ("compression", StorageSetting::Text(text)) => {
                    if out.compression.is_some() {
                        return Err(twice());
                    }
                    out.compression = Some(BlockCodec::parse(text)?);
                }
                ("cold_compression", StorageSetting::Text(text)) => {
                    if out.cold_compression.is_some() {
                        return Err(twice());
                    }
                    out.cold_compression = Some(BlockCodec::parse(text)?);
                }
                ("dictionaries", StorageSetting::Bool(on)) => {
                    if out.dictionaries.is_some() {
                        return Err(twice());
                    }
                    out.dictionaries = Some(on);
                }
                ("compression" | "cold_compression", StorageSetting::Bool(_)) => {
                    return Err(alloc::format!("storage setting '{key}' takes a codec name"));
                }
                ("dictionaries", StorageSetting::Text(_)) => {
                    return Err("storage setting 'dictionaries' takes true or false".into());
                }
                _ => {
                    return Err(alloc::format!(
                        "'{key}' is not a storage setting: compression, cold_compression, \
                         dictionaries"
                    ));
                }
            }
        }
        Ok(out)
    }

    /// The bytes of the settings, fixed width: a codec as a tag and a level,
    /// the switch as unset, off or on.
    fn encode_into(&self, bytes: &mut Vec<u8>) {
        for codec in [self.compression, self.cold_compression] {
            let (tag, level) = match codec {
                None => (0u8, 0i32),
                Some(BlockCodec::None) => (1, 0),
                Some(BlockCodec::Lz4) => (2, 0),
                Some(BlockCodec::Zstd(level)) => (3, level),
            };
            bytes.push(tag);
            bytes.extend_from_slice(&level.to_be_bytes());
        }
        bytes.push(match self.dictionaries {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        });
    }

    /// Bytes [`Self::encode_into`] writes.
    const ENCODED_LEN: usize = 2 * 5 + 1;

    /// The settings `bytes` hold; `None` for bytes no encoder wrote.
    fn decode(bytes: &[u8; Self::ENCODED_LEN]) -> Option<Self> {
        let codec = |at: usize| -> Option<Option<BlockCodec>> {
            let level = i32::from_be_bytes(bytes[at + 1..at + 5].try_into().ok()?);
            match bytes[at] {
                0 => Some(None),
                1 => Some(Some(BlockCodec::None)),
                2 => Some(Some(BlockCodec::Lz4)),
                3 if ZSTD_LEVELS.contains(&level) => Some(Some(BlockCodec::Zstd(level))),
                _ => None,
            }
        };
        Some(Self {
            compression: codec(0)?,
            cold_compression: codec(5)?,
            dictionaries: match bytes[10] {
                0 => None,
                1 => Some(false),
                2 => Some(true),
                _ => return None,
            },
        })
    }
}

/// A storage setting's value as a statement spells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageSetting<'a> {
    /// A quoted text: a codec name.
    Text(&'a str),
    /// `true` or `false`.
    Bool(bool),
}

/// The deployment catalog's record of one namespace incarnation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceRecord {
    /// Its name while it was live.
    pub name: String,
    /// Where it stands.
    pub state: NamespaceState,
    /// The token of the creation that made it.
    pub token: [u8; NAMESPACE_TOKEN_LEN],
    /// Its data-block storage settings.
    pub storage: NamespaceStorage,
}

impl NamespaceRecord {
    /// The record's bytes: its state, its token, its storage settings, then
    /// its name.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            1 + NAMESPACE_TOKEN_LEN + NamespaceStorage::ENCODED_LEN + self.name.len(),
        );
        bytes.push(match self.state {
            NamespaceState::Live => 0,
            NamespaceState::Dropping => 1,
        });
        bytes.extend_from_slice(&self.token);
        self.storage.encode_into(&mut bytes);
        bytes.extend_from_slice(self.name.as_bytes());
        bytes
    }

    /// The record `bytes` hold; `None` for bytes no [`Self::encode`] wrote.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let (&state, rest) = bytes.split_first()?;
        let state = match state {
            0 => NamespaceState::Live,
            1 => NamespaceState::Dropping,
            _ => return None,
        };
        let (token, rest) = rest.split_first_chunk::<NAMESPACE_TOKEN_LEN>()?;
        let (storage, name) = rest.split_first_chunk::<{ NamespaceStorage::ENCODED_LEN }>()?;
        let storage = NamespaceStorage::decode(storage)?;
        let name = core::str::from_utf8(name).ok()?;
        validate_namespace_name(name).ok()?;
        Some(Self {
            name: name.into(),
            state,
            token: *token,
            storage,
        })
    }
}

/// A namespace incarnation of one deployment. Allocated by the deployment's
/// catalog and never reused within it: a namespace dropped and created again
/// under the same name is a new incarnation with a new id, so nothing bound
/// to the old one (a dictionary handle, a tree, a delayed worker) can reach
/// the new one.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct NamespaceId(u64);

impl NamespaceId {
    /// The id from its number.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The id's number.
    #[must_use]
    pub const fn as_raw(self) -> u64 {
        self.0
    }

    /// The name of the namespace's directory in a store: fixed width, so a
    /// listing sorts by id.
    #[must_use]
    pub fn dir_name(self) -> String {
        alloc::format!("{:016x}", self.0)
    }

    /// The id a namespace directory is named for, `None` for a name no
    /// namespace directory has.
    #[must_use]
    pub fn from_dir_name(name: &str) -> Option<Self> {
        (name.len() == 16 && name.bytes().all(|b| b.is_ascii_hexdigit()))
            .then(|| u64::from_str_radix(name, 16).ok().map(Self))
            .flatten()
    }
}

impl core::fmt::Display for NamespaceId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "namespace {}", self.0)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
