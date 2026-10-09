//! Which catalog records of a store this build cannot read.
//!
//! Every serialized record family in the Schema partition is decoded with
//! the type this build reads it as. A record that fails is shape the engine
//! will refuse when it reaches it, often only on the path that needs it (a
//! background task, one label's schema), so a check reports all of them at
//! once, with the record's content, for a migration to be written against.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use coordinode_storage::Guard as _;
use coordinode_storage::engine::partition::Partition;
use serde::de::DeserializeOwned;

use crate::{open_store, stores};

/// One family of catalog records: where it lives and the type it reads as.
struct Family {
    /// What the records are, as reports name them.
    what: &'static str,
    /// The key prefix of the family.
    prefix: &'static [u8],
    /// Whether a key under the prefix is a record of the family (some
    /// prefixes also hold markers or pointers).
    is_record: fn(&[u8]) -> bool,
    /// Decode a record, the error as text.
    decode: fn(&[u8]) -> Result<(), String>,
}

fn decodes<T: DeserializeOwned>(bytes: &[u8]) -> Result<(), String> {
    rmp_serde::from_slice::<T>(bytes)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// `<prefix><name>:<revision>`: a revisioned schema record, not a marker.
fn revisioned(prefix: &'static [u8]) -> impl Fn(&[u8]) -> bool {
    move |key: &[u8]| {
        key.strip_prefix(prefix)
            .and_then(|rest| core::str::from_utf8(rest).ok())
            .and_then(|rest| rest.rsplit_once(':'))
            .is_some_and(|(name, rev)| {
                !name.is_empty() && !rev.is_empty() && rev.bytes().all(|b| b.is_ascii_digit())
            })
    }
}

fn any_key(_: &[u8]) -> bool {
    true
}

const FAMILIES: &[Family] = &[
    Family {
        what: "label schema",
        prefix: b"schema:label:",
        is_record: |k| revisioned(b"schema:label:")(k),
        decode: decodes::<coordinode_core::schema::definition::LabelSchema>,
    },
    Family {
        what: "edge type schema",
        prefix: b"schema:edge_type:",
        is_record: |k| revisioned(b"schema:edge_type:")(k),
        decode: decodes::<coordinode_core::schema::definition::EdgeTypeSchema>,
    },
    Family {
        what: "index definition",
        prefix: b"schema:idx:",
        is_record: any_key,
        decode: decodes::<coordinode_modality::IndexDefinition>,
    },
    Family {
        what: "index build record",
        prefix: coordinode_modality::IndexBuildRecord::PREFIX,
        is_record: any_key,
        decode: decodes::<coordinode_modality::IndexBuildRecord>,
    },
    Family {
        what: "duplicate repair record",
        prefix: b"schema:idxrepair:",
        is_record: any_key,
        decode: decodes::<coordinode_modality::DuplicateRepairRecord>,
    },
    Family {
        what: "trigger",
        prefix: b"schema:trigger:",
        is_record: any_key,
        decode: decodes::<coordinode_core::schema::triggers::TriggerSchema>,
    },
    Family {
        what: "cardinality profile",
        prefix: coordinode_core::graph::cardinality::CARDINALITY_PROFILE_PREFIX,
        is_record: any_key,
        decode: decodes::<coordinode_core::graph::cardinality::CardinalityProfile>,
    },
];

/// A record this build cannot read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unreadable {
    /// The family it belongs to.
    pub what: &'static str,
    /// Its key, printable.
    pub key: String,
    /// Why it does not decode.
    pub error: String,
    /// Its content as msgpack, shortened.
    pub content: String,
}

/// What a check of one store found.
#[derive(Debug, Clone)]
pub struct StoreCheck {
    /// The store.
    pub path: PathBuf,
    /// Records decoded, per family.
    pub checked: Vec<(&'static str, usize)>,
    /// The ones that did not decode.
    pub unreadable: Vec<Unreadable>,
}

/// Longest content shown for one record.
const CONTENT_LIMIT: usize = 600;

fn render(bytes: &[u8]) -> String {
    let text = match rmpv::decode::read_value(&mut &bytes[..]) {
        Ok(value) => value.to_string(),
        Err(_) => format!("{} bytes, not msgpack", bytes.len()),
    };
    if text.len() <= CONTENT_LIMIT {
        return text;
    }
    let mut end = CONTENT_LIMIT;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &text[..end])
}

/// Check every store under `data` (the store and its checkpoints).
///
/// # Errors
///
/// A store cannot be opened or scanned.
pub fn check(data: &Path) -> anyhow::Result<Vec<StoreCheck>> {
    stores(data)?
        .into_iter()
        .map(|dir| check_store(&dir))
        .collect()
}

fn check_store(dir: &Path) -> anyhow::Result<StoreCheck> {
    let engine = open_store(dir)?;
    let mut checked = Vec::new();
    let mut unreadable = Vec::new();
    for family in FAMILIES {
        let mut n = 0;
        for guard in engine
            .prefix_scan(Partition::Schema, family.prefix)
            .with_context(|| format!("scan {}", dir.display()))?
        {
            let (key, value) = guard
                .into_inner()
                .with_context(|| format!("scan {}", dir.display()))?;
            if !(family.is_record)(&key) {
                continue;
            }
            n += 1;
            if let Err(error) = (family.decode)(&value) {
                unreadable.push(Unreadable {
                    what: family.what,
                    key: coordinode_storage::error::printable_key(&key),
                    error,
                    content: render(&value),
                });
            }
        }
        checked.push((family.what, n));
    }
    Ok(StoreCheck {
        path: dir.to_path_buf(),
        checked,
        unreadable,
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
