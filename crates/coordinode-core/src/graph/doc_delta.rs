//! Document merge operand types for path-targeted partial updates.
//!
//! `DocDelta` operands are written via `storage.merge()` on the `node:` partition,
//! enabling O(1) writes without reading the existing document. The LSM merge
//! function applies deltas during reads and compaction.
//!
//! Wire format: `[0x01, msgpack(DocDelta)]` — prefix byte 0x01 distinguishes
//! merge operands from full NodeRecords (prefix 0x00). See ADR-015.

use serde::{Deserialize, Serialize};

/// Prefix byte for a full NodeRecord (PUT value).
pub const PREFIX_NODE_RECORD: u8 = 0x00;

/// Prefix byte for a DocDelta merge operand.
pub const PREFIX_DOC_DELTA: u8 = 0x01;

/// Where a DocDelta path is rooted within a `NodeRecord`.
///
/// - `Extra`: targets `NodeRecord.extra` (string-keyed overflow map for
///   FLEXIBLE/VALIDATED schema mode). Path segments are string keys.
/// - `PropField(u32)`: targets `NodeRecord.props[field_id]` where the field_id
///   was resolved by the executor at write time via `FieldInterner`. The merge
///   function navigates `props[field_id] → Value::Document → sub_path`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum PathTarget {
    /// Target the `extra` overflow map (string keys).
    #[default]
    Extra,
    /// Target `props[field_id]` — the u32 field ID is baked in at write time.
    PropField(u32),
}

/// A document merge operand — one atomic operation on a nested document property.
///
/// Written as merge operands to the `node:` partition. The merge function
/// applies them in seqno order against the base `NodeRecord`.
///
/// ## Commutativity
///
/// - `SetPath` on **different paths**: commutative (conflict-free)
/// - `SetPath` on **same path**: last-writer-wins (seqno order)
/// - `DeletePath`: idempotent (deleting absent path = no-op)
/// - `ArrayPush`: **not** commutative (element order depends on seqno)
/// - `ArrayPull`: idempotent on distinct values
/// - `ArrayAddToSet`: commutative (dedup means order doesn't matter)
/// - `Increment`: commutative (addition is commutative)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DocDelta {
    /// Set a value at a dotted path. Creates intermediate objects if missing.
    SetPath {
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    },

    /// Delete a value at a dotted path. Idempotent (missing path = no-op).
    DeletePath {
        target: PathTarget,
        path: Vec<String>,
    },

    /// Append a value to an array at path. Creates array if missing.
    ArrayPush {
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    },

    /// Remove first occurrence of a value from an array at path. Idempotent.
    ArrayPull {
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    },

    /// Add value to array only if not already present. Commutative.
    ArrayAddToSet {
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    },

    /// Atomic numeric increment at path. Commutative.
    Increment {
        target: PathTarget,
        path: Vec<String>,
        amount: f64,
    },

    /// Remove an entire top-level property from the NodeRecord.
    ///
    /// - `PropField(field_id)`: removes `props[field_id]` entirely.
    /// - `Extra`: removes the key named in `key` from the `extra` overflow map.
    ///
    /// Idempotent (removing absent property = no-op). Used by TTL reaper
    /// for Field/Subtree scope deletion without read-modify-write.
    RemoveProperty {
        target: PathTarget,
        /// Key name for Extra target. Unused for PropField (field_id is in target).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        key: Option<String>,
    },
}

impl DocDelta {
    /// Encode as a merge operand: `[PREFIX_DOC_DELTA, msgpack(self)]`.
    pub fn encode(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        let msgpack = rmp_serde::to_vec(self)?;
        let mut buf = Vec::with_capacity(1 + msgpack.len());
        buf.push(PREFIX_DOC_DELTA);
        buf.extend_from_slice(&msgpack);
        Ok(buf)
    }

    /// Decode from a merge operand (without the prefix byte).
    /// Caller must strip the `PREFIX_DOC_DELTA` byte before calling.
    pub fn decode(data: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(data)
    }

    /// Returns the `PathTarget` for this delta.
    pub fn target(&self) -> &PathTarget {
        match self {
            Self::SetPath { target, .. }
            | Self::DeletePath { target, .. }
            | Self::ArrayPush { target, .. }
            | Self::ArrayPull { target, .. }
            | Self::ArrayAddToSet { target, .. }
            | Self::Increment { target, .. }
            | Self::RemoveProperty { target, .. } => target,
        }
    }

    /// Apply this delta to an `rmpv::Value` document in place.
    ///
    /// The document should be the node's DOCUMENT-typed property value.
    /// Returns `true` if the document was modified.
    pub fn apply(&self, doc: &mut rmpv::Value) -> bool {
        match self {
            DocDelta::SetPath { path, value, .. } => set_at_path(doc, path, value.clone()),
            DocDelta::DeletePath { path, .. } => delete_at_path(doc, path),
            DocDelta::ArrayPush { path, value, .. } => array_push(doc, path, value.clone()),
            DocDelta::ArrayPull { path, value, .. } => array_pull(doc, path, value),
            DocDelta::ArrayAddToSet { path, value, .. } => {
                array_add_to_set(doc, path, value.clone())
            }
            DocDelta::Increment { path, amount, .. } => increment(doc, path, *amount),
            // RemoveProperty is handled at the merge function level (NodeRecord),
            // not on the rmpv::Value document. No-op here.
            DocDelta::RemoveProperty { .. } => false,
        }
    }
}

// --- Path manipulation helpers ---

/// Set a value at a dotted path, creating intermediate maps if needed.
///
/// Empty path = replace the entire document at the root. Combined with
/// `PathTarget::PropField(fid)`, this lets callers replace `props[fid]`
/// wholesale with a new value (used by e.g. ATTACH DOCUMENT when promoting
/// a graph node back into a single-segment nested property).
fn set_at_path(doc: &mut rmpv::Value, path: &[String], value: rmpv::Value) -> bool {
    if path.is_empty() {
        *doc = value;
        return true;
    }

    if path.len() == 1 {
        return set_map_key(doc, &path[0], value);
    }

    // Navigate to the parent, creating intermediate maps as needed.
    let parent = ensure_path(doc, &path[..path.len() - 1]);
    set_map_key(parent, &path[path.len() - 1], value)
}

/// Delete a value at a dotted path. Returns true if something was removed.
fn delete_at_path(doc: &mut rmpv::Value, path: &[String]) -> bool {
    if path.is_empty() {
        return false;
    }

    if path.len() == 1 {
        return remove_map_key(doc, &path[0]);
    }

    // Navigate to the parent (don't create intermediates for delete).
    let parent = match navigate_to(doc, &path[..path.len() - 1]) {
        Some(v) => v,
        None => return false, // Path doesn't exist — idempotent no-op.
    };
    remove_map_key(parent, &path[path.len() - 1])
}

/// Push a value onto an array at path, creating the array if needed.
fn array_push(doc: &mut rmpv::Value, path: &[String], value: rmpv::Value) -> bool {
    let target = ensure_path(doc, path);
    match target {
        rmpv::Value::Array(arr) => {
            arr.push(value);
            true
        }
        rmpv::Value::Nil => {
            // Create array with the single value.
            *target = rmpv::Value::Array(vec![value]);
            true
        }
        _ => {
            // Target exists but is not an array — replace with array.
            *target = rmpv::Value::Array(vec![value]);
            true
        }
    }
}

/// Remove first occurrence of a value from an array at path.
fn array_pull(doc: &mut rmpv::Value, path: &[String], value: &rmpv::Value) -> bool {
    let target = match navigate_to(doc, path) {
        Some(v) => v,
        None => return false,
    };
    if let rmpv::Value::Array(arr) = target {
        if let Some(pos) = arr.iter().position(|v| v == value) {
            arr.remove(pos);
            return true;
        }
    }
    false
}

/// Add value to array only if not already present.
fn array_add_to_set(doc: &mut rmpv::Value, path: &[String], value: rmpv::Value) -> bool {
    let target = ensure_path(doc, path);
    match target {
        rmpv::Value::Array(arr) => {
            if arr.contains(&value) {
                return false; // Already present — no-op.
            }
            arr.push(value);
            true
        }
        rmpv::Value::Nil => {
            *target = rmpv::Value::Array(vec![value]);
            true
        }
        _ => {
            *target = rmpv::Value::Array(vec![value]);
            true
        }
    }
}

/// Atomic increment of a numeric value at path.
fn increment(doc: &mut rmpv::Value, path: &[String], amount: f64) -> bool {
    let target = ensure_path(doc, path);
    match target {
        rmpv::Value::Integer(n) => {
            let current = n.as_f64().unwrap_or(0.0);
            let result = current + amount;
            // Preserve integer type if both are integral.
            if amount.fract() == 0.0 && current.fract() == 0.0 {
                *target = rmpv::Value::Integer((result as i64).into());
            } else {
                *target = rmpv::Value::F64(result);
            }
            true
        }
        rmpv::Value::F64(f) => {
            *f += amount;
            true
        }
        rmpv::Value::F32(f) => {
            *target = rmpv::Value::F64(f64::from(*f) + amount);
            true
        }
        rmpv::Value::Nil => {
            // Initialize from zero.
            if amount.fract() == 0.0 {
                *target = rmpv::Value::Integer((amount as i64).into());
            } else {
                *target = rmpv::Value::F64(amount);
            }
            true
        }
        _ => false, // Non-numeric — can't increment.
    }
}

/// Navigate to a value at path, returning a mutable reference.
/// Does NOT create intermediate maps — returns None if path doesn't exist.
fn navigate_to<'a>(doc: &'a mut rmpv::Value, path: &[String]) -> Option<&'a mut rmpv::Value> {
    let mut current = doc;
    for segment in path {
        current = match current {
            rmpv::Value::Map(entries) => {
                let key = rmpv::Value::String(segment.as_str().into());
                entries
                    .iter_mut()
                    .find(|(k, _)| *k == key)
                    .map(|(_, v)| v)?
            }
            _ => return None,
        };
    }
    Some(current)
}

/// Navigate to a value at path, creating empty maps for missing segments.
fn ensure_path<'a>(doc: &'a mut rmpv::Value, path: &[String]) -> &'a mut rmpv::Value {
    let mut current = doc;
    for segment in path {
        // Ensure current is a map.
        if !matches!(current, rmpv::Value::Map(_)) {
            *current = rmpv::Value::Map(Vec::new());
        }

        let key = rmpv::Value::String(segment.as_str().into());

        // Find or insert the key.
        let entries = match current {
            rmpv::Value::Map(entries) => entries,
            _ => unreachable!("just ensured it's a Map"),
        };

        let idx = entries.iter().position(|(k, _)| *k == key);
        let idx = match idx {
            Some(i) => i,
            None => {
                entries.push((key, rmpv::Value::Nil));
                entries.len() - 1
            }
        };

        current = &mut entries[idx].1;
    }
    current
}

/// Set a key in a map value. Creates the map if the value isn't one.
fn set_map_key(doc: &mut rmpv::Value, key: &str, value: rmpv::Value) -> bool {
    if !matches!(doc, rmpv::Value::Map(_)) {
        *doc = rmpv::Value::Map(Vec::new());
    }

    let rmpv_key = rmpv::Value::String(key.into());
    let entries = match doc {
        rmpv::Value::Map(entries) => entries,
        _ => unreachable!(),
    };

    // Update existing or insert new.
    for (k, v) in entries.iter_mut() {
        if *k == rmpv_key {
            *v = value;
            return true;
        }
    }
    entries.push((rmpv_key, value));
    true
}

/// Remove a key from a map. Returns true if key was found and removed.
fn remove_map_key(doc: &mut rmpv::Value, key: &str) -> bool {
    let entries = match doc {
        rmpv::Value::Map(entries) => entries,
        _ => return false,
    };

    let rmpv_key = rmpv::Value::String(key.into());
    let len_before = entries.len();
    entries.retain(|(k, _)| *k != rmpv_key);
    entries.len() < len_before
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn make_map(entries: Vec<(&str, rmpv::Value)>) -> rmpv::Value {
        rmpv::Value::Map(
            entries
                .into_iter()
                .map(|(k, v)| (rmpv::Value::String(k.into()), v))
                .collect(),
        )
    }

    // --- SetPath tests ---

    #[test]
    fn set_path_single_level() {
        let mut doc = make_map(vec![("name", rmpv::Value::String("Alice".into()))]);
        let delta = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["name".into()],
            value: rmpv::Value::String("Bob".into()),
        };
        assert!(delta.apply(&mut doc));
        assert_eq!(
            doc,
            make_map(vec![("name", rmpv::Value::String("Bob".into()))])
        );
    }

    #[test]
    fn set_path_empty_path_replaces_root() {
        // SetPath with an empty path replaces the entire document at the root.
        // Used by e.g. ATTACH DOCUMENT when promoting a graph node back into a
        // single-segment nested property (`PropField(fid)` + empty subpath).
        let mut doc = make_map(vec![("old", rmpv::Value::Boolean(true))]);
        let replacement = make_map(vec![
            ("city", rmpv::Value::String("Prague".into())),
            ("zip", rmpv::Value::String("11000".into())),
        ]);
        let delta = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec![],
            value: replacement.clone(),
        };
        assert!(delta.apply(&mut doc));
        assert_eq!(doc, replacement);
    }

    #[test]
    fn set_path_creates_intermediates() {
        let mut doc = make_map(vec![]);
        let delta = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["config".into(), "network".into(), "ssid".into()],
            value: rmpv::Value::String("home".into()),
        };
        assert!(delta.apply(&mut doc));

        let result = super::super::document::extract_at_path(&doc, &["config", "network", "ssid"]);
        assert_eq!(result, rmpv::Value::String("home".into()));
    }

    #[test]
    fn set_path_overwrites_existing() {
        let mut doc = make_map(vec![(
            "a",
            make_map(vec![("b", rmpv::Value::Integer(1.into()))]),
        )]);
        let delta = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["a".into(), "b".into()],
            value: rmpv::Value::Integer(42.into()),
        };
        assert!(delta.apply(&mut doc));
        assert_eq!(
            super::super::document::extract_at_path(&doc, &["a", "b"]),
            rmpv::Value::Integer(42.into())
        );
    }

    #[test]
    fn set_path_empty_path_replaces_any_root_value() {
        // Empty path replaces the root regardless of its prior shape —
        // nil, scalar, map, or array. Used by ATTACH DOCUMENT to promote
        // a graph node's properties into a single-segment nested field.
        let mut doc = rmpv::Value::Nil;
        let delta = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec![],
            value: rmpv::Value::Integer(1.into()),
        };
        assert!(delta.apply(&mut doc));
        assert_eq!(doc, rmpv::Value::Integer(1.into()));

        // And it replaces scalar roots just as well.
        let mut doc = rmpv::Value::String("stale".into());
        let delta = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec![],
            value: rmpv::Value::Boolean(true),
        };
        assert!(delta.apply(&mut doc));
        assert_eq!(doc, rmpv::Value::Boolean(true));
    }

    // --- DeletePath tests ---

    #[test]
    fn delete_path_removes_key() {
        let mut doc = make_map(vec![
            ("a", rmpv::Value::Integer(1.into())),
            ("b", rmpv::Value::Integer(2.into())),
        ]);
        let delta = DocDelta::DeletePath {
            target: PathTarget::Extra,
            path: vec!["a".into()],
        };
        assert!(delta.apply(&mut doc));
        assert_eq!(doc, make_map(vec![("b", rmpv::Value::Integer(2.into()))]));
    }

    #[test]
    fn delete_path_nested() {
        let mut doc = make_map(vec![(
            "config",
            make_map(vec![
                ("a", rmpv::Value::Integer(1.into())),
                ("b", rmpv::Value::Integer(2.into())),
            ]),
        )]);
        let delta = DocDelta::DeletePath {
            target: PathTarget::Extra,
            path: vec!["config".into(), "a".into()],
        };
        assert!(delta.apply(&mut doc));
        assert_eq!(
            super::super::document::extract_at_path(&doc, &["config", "b"]),
            rmpv::Value::Integer(2.into())
        );
        assert_eq!(
            super::super::document::extract_at_path(&doc, &["config", "a"]),
            rmpv::Value::Nil
        );
    }

    #[test]
    fn delete_path_missing_is_noop() {
        let mut doc = make_map(vec![("a", rmpv::Value::Integer(1.into()))]);
        let delta = DocDelta::DeletePath {
            target: PathTarget::Extra,
            path: vec!["nonexistent".into()],
        };
        assert!(!delta.apply(&mut doc));
        assert_eq!(doc, make_map(vec![("a", rmpv::Value::Integer(1.into()))]));
    }

    // --- ArrayPush tests ---

    #[test]
    fn array_push_to_existing() {
        let mut doc = make_map(vec![(
            "tags",
            rmpv::Value::Array(vec![rmpv::Value::String("a".into())]),
        )]);
        let delta = DocDelta::ArrayPush {
            target: PathTarget::Extra,
            path: vec!["tags".into()],
            value: rmpv::Value::String("b".into()),
        };
        assert!(delta.apply(&mut doc));
        assert_eq!(
            doc,
            make_map(vec![(
                "tags",
                rmpv::Value::Array(vec![
                    rmpv::Value::String("a".into()),
                    rmpv::Value::String("b".into()),
                ])
            )])
        );
    }

    #[test]
    fn array_push_creates_array() {
        let mut doc = make_map(vec![]);
        let delta = DocDelta::ArrayPush {
            target: PathTarget::Extra,
            path: vec!["tags".into()],
            value: rmpv::Value::String("first".into()),
        };
        assert!(delta.apply(&mut doc));
        assert_eq!(
            doc,
            make_map(vec![(
                "tags",
                rmpv::Value::Array(vec![rmpv::Value::String("first".into())])
            )])
        );
    }

    // --- ArrayPull tests ---

    #[test]
    fn array_pull_removes_first_match() {
        let mut doc = make_map(vec![(
            "tags",
            rmpv::Value::Array(vec![
                rmpv::Value::String("a".into()),
                rmpv::Value::String("b".into()),
                rmpv::Value::String("a".into()),
            ]),
        )]);
        let delta = DocDelta::ArrayPull {
            target: PathTarget::Extra,
            path: vec!["tags".into()],
            value: rmpv::Value::String("a".into()),
        };
        assert!(delta.apply(&mut doc));
        // Only first "a" removed.
        assert_eq!(
            doc,
            make_map(vec![(
                "tags",
                rmpv::Value::Array(vec![
                    rmpv::Value::String("b".into()),
                    rmpv::Value::String("a".into()),
                ])
            )])
        );
    }

    #[test]
    fn array_pull_missing_value_noop() {
        let mut doc = make_map(vec![(
            "tags",
            rmpv::Value::Array(vec![rmpv::Value::String("a".into())]),
        )]);
        let delta = DocDelta::ArrayPull {
            target: PathTarget::Extra,
            path: vec!["tags".into()],
            value: rmpv::Value::String("z".into()),
        };
        assert!(!delta.apply(&mut doc));
    }

    // --- ArrayAddToSet tests ---

    #[test]
    fn add_to_set_adds_new() {
        let mut doc = make_map(vec![(
            "tags",
            rmpv::Value::Array(vec![rmpv::Value::String("a".into())]),
        )]);
        let delta = DocDelta::ArrayAddToSet {
            target: PathTarget::Extra,
            path: vec!["tags".into()],
            value: rmpv::Value::String("b".into()),
        };
        assert!(delta.apply(&mut doc));
        assert_eq!(
            doc,
            make_map(vec![(
                "tags",
                rmpv::Value::Array(vec![
                    rmpv::Value::String("a".into()),
                    rmpv::Value::String("b".into()),
                ])
            )])
        );
    }

    #[test]
    fn add_to_set_skips_duplicate() {
        let mut doc = make_map(vec![(
            "tags",
            rmpv::Value::Array(vec![rmpv::Value::String("a".into())]),
        )]);
        let delta = DocDelta::ArrayAddToSet {
            target: PathTarget::Extra,
            path: vec!["tags".into()],
            value: rmpv::Value::String("a".into()),
        };
        assert!(!delta.apply(&mut doc));
        // Array unchanged.
        assert_eq!(
            doc,
            make_map(vec![(
                "tags",
                rmpv::Value::Array(vec![rmpv::Value::String("a".into())])
            )])
        );
    }

    // --- Increment tests ---

    #[test]
    fn increment_integer() {
        let mut doc = make_map(vec![(
            "stats",
            make_map(vec![("views", rmpv::Value::Integer(10.into()))]),
        )]);
        let delta = DocDelta::Increment {
            target: PathTarget::Extra,
            path: vec!["stats".into(), "views".into()],
            amount: 5.0,
        };
        assert!(delta.apply(&mut doc));
        assert_eq!(
            super::super::document::extract_at_path(&doc, &["stats", "views"]),
            rmpv::Value::Integer(15.into())
        );
    }

    #[test]
    fn increment_float() {
        let mut doc = make_map(vec![("score", rmpv::Value::F64(1.5))]);
        let delta = DocDelta::Increment {
            target: PathTarget::Extra,
            path: vec!["score".into()],
            amount: 0.5,
        };
        assert!(delta.apply(&mut doc));
        assert_eq!(doc, make_map(vec![("score", rmpv::Value::F64(2.0))]));
    }

    #[test]
    fn increment_from_nil() {
        let mut doc = make_map(vec![]);
        let delta = DocDelta::Increment {
            target: PathTarget::Extra,
            path: vec!["counter".into()],
            amount: 1.0,
        };
        assert!(delta.apply(&mut doc));
        assert_eq!(
            doc,
            make_map(vec![("counter", rmpv::Value::Integer(1.into()))])
        );
    }

    #[test]
    fn increment_fractional_promotes_to_float() {
        let mut doc = make_map(vec![("val", rmpv::Value::Integer(10.into()))]);
        let delta = DocDelta::Increment {
            target: PathTarget::Extra,
            path: vec!["val".into()],
            amount: 0.5,
        };
        assert!(delta.apply(&mut doc));
        assert_eq!(doc, make_map(vec![("val", rmpv::Value::F64(10.5))]));
    }

    // --- Encode/decode roundtrip ---

    #[test]
    fn encode_decode_roundtrip() {
        let deltas = vec![
            DocDelta::SetPath {
                target: PathTarget::Extra,
                path: vec!["a".into(), "b".into()],
                value: rmpv::Value::Integer(42.into()),
            },
            DocDelta::DeletePath {
                target: PathTarget::Extra,
                path: vec!["x".into()],
            },
            DocDelta::ArrayPush {
                target: PathTarget::Extra,
                path: vec!["tags".into()],
                value: rmpv::Value::String("new".into()),
            },
            DocDelta::ArrayPull {
                target: PathTarget::Extra,
                path: vec!["tags".into()],
                value: rmpv::Value::String("old".into()),
            },
            DocDelta::ArrayAddToSet {
                target: PathTarget::Extra,
                path: vec!["tags".into()],
                value: rmpv::Value::String("unique".into()),
            },
            DocDelta::Increment {
                target: PathTarget::Extra,
                path: vec!["count".into()],
                amount: 1.0,
            },
        ];

        for delta in &deltas {
            let encoded = delta.encode().expect("encode failed");
            assert_eq!(encoded[0], PREFIX_DOC_DELTA);
            let decoded = DocDelta::decode(&encoded[1..]).expect("decode failed");
            assert_eq!(&decoded, delta, "roundtrip failed for {delta:?}");
        }
    }
}
