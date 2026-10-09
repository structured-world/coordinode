//! Label schema records written before uniqueness became a named constraint
//! only and before constraints carried a scope.
//!
//! A property definition was `[name, type, not_null, default, unique]`; the
//! current one has no uniqueness flag, uniqueness being a named constraint
//! of the label. A node constraint was `[name, properties, kind, state]`;
//! the current one adds the scope, the nodes it constrains when not all of
//! the label's. The current engine refuses such a schema ("array had
//! incorrect length, expected 4") wherever it loads the label.
//!
//! A flag of `false` carries nothing and is dropped. A flag of `true` is
//! dropped only when the label already holds a uniqueness constraint on
//! exactly that property; otherwise the uniqueness would be lost, and the
//! migration stops naming the record. A constraint gets no scope: it held
//! over every node of the label.

use std::path::Path;

use anyhow::{Context as _, bail};
use coordinode_core::schema::definition::LabelSchema;
use rmpv::Value;

use super::SchemaRecords;
use crate::{Backup, Finding, Migration};

pub(super) const MIGRATION: Migration = Migration {
    name: "label-schema-shape",
    summary: "label schemas lose the property uniqueness flag and gain constraint scopes",
    survey,
    apply,
};

const PREFIX: &[u8] = b"schema:label:";

const RECORDS: SchemaRecords = SchemaRecords {
    what: "label schema",
    prefix: PREFIX,
    is_record: is_revision_key,
    upgrade: upgrade_schema,
};

/// Field positions in a label schema record.
const PROPERTIES: usize = 1;
const CONSTRAINTS: usize = 10;

/// `schema:label:<name>:<revision>`: a revisioned schema, not a marker.
fn is_revision_key(key: &[u8]) -> bool {
    key.strip_prefix(PREFIX)
        .and_then(|rest| core::str::from_utf8(rest).ok())
        .and_then(|rest| rest.rsplit_once(':'))
        .is_some_and(|(name, rev)| {
            !name.is_empty() && !rev.is_empty() && rev.bytes().all(|b| b.is_ascii_digit())
        })
}

fn survey(data: &Path) -> anyhow::Result<Vec<Finding>> {
    RECORDS.survey(data)
}

fn apply(finding: &Finding, _backup: &Backup) -> anyhow::Result<()> {
    RECORDS.apply(finding)
}

/// Whether `constraint` (a record of either shape) is a uniqueness
/// constraint on exactly `property`.
fn is_unique_on(constraint: &Value, property: &str) -> bool {
    let Value::Array(fields) = constraint else {
        return false;
    };
    let on_property = matches!(
        fields.get(1),
        Some(Value::Array(props)) if props.len() == 1 && props[0].as_str() == Some(property)
    );
    let unique = matches!(fields.get(2), Some(v) if v.as_str() == Some("Unique"));
    on_property && unique
}

/// The schema `data` in the current shape, or `None` when it already reads
/// as one.
pub(crate) fn upgrade_schema(data: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
    if LabelSchema::from_msgpack(data).is_ok() {
        return Ok(None);
    }
    let mut value = rmpv::decode::read_value(&mut &data[..]).context("not msgpack")?;
    let Value::Array(fields) = &mut value else {
        bail!("not a label schema: {value}");
    };

    // Constraints first: the uniqueness flags below are checked against them.
    let mut constraints = Vec::new();
    if let Some(slot) = fields.get_mut(CONSTRAINTS) {
        let Value::Array(list) = slot else {
            bail!("constraints are not a list: {slot}");
        };
        for constraint in list.iter_mut() {
            let Value::Array(c) = constraint else {
                bail!("a constraint is not a record: {constraint}");
            };
            match c.len() {
                4 => c.push(Value::Nil),
                5 => {}
                n => {
                    bail!("a constraint with {n} fields, neither the old four nor the current five")
                }
            }
        }
        constraints = list.clone();
    }

    let Some(Value::Map(properties)) = fields.get_mut(PROPERTIES) else {
        bail!("no property map where a label schema has one");
    };
    for (name, def) in properties.iter_mut() {
        let Value::Array(d) = def else {
            bail!("property {name} is not a definition: {def}");
        };
        match d.len() {
            5 => {
                if d[4].as_bool() == Some(true) {
                    let property = d[0].as_str().unwrap_or_default().to_string();
                    if !constraints.iter().any(|c| is_unique_on(c, &property)) {
                        bail!(
                            "property {property} is marked unique with no uniqueness \
                             constraint on it; dropping the flag would lose the uniqueness"
                        );
                    }
                }
                d.truncate(4);
            }
            4 => {}
            n => {
                bail!("property {name} with {n} fields, neither the old five nor the current four")
            }
        }
    }

    let mut out = Vec::with_capacity(data.len());
    rmpv::encode::write_value(&mut out, &value).context("encode the rewritten schema")?;
    LabelSchema::from_msgpack(&out).context("the rewritten schema does not read back")?;
    Ok(Some(out))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
