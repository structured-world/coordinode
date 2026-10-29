//! Label kinds: which label of a node is its type, and what a type label may
//! declare about the labels it implies.
//!
//! A node carries exactly one type label as its primary label, whatever the
//! order its labels are written in. The labels that type implies (transitively)
//! go with it, and any other label must be auxiliary. The engine stores the
//! primary label first, so a reader of a stored record takes the first label
//! as the primary one without consulting the catalog.

use smallvec::SmallVec;

use crate::schema::definition::{LabelKind, LabelSchema};

/// What the catalog says about one label, as label resolution needs it. A
/// label without a definition is a FLEXIBLE type label implying nothing.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LabelFacts {
    /// Type or auxiliary.
    pub kind: LabelKind,
    /// The labels it names in `IMPLIES`.
    pub implies: Vec<String>,
}

impl LabelFacts {
    /// The facts of a declared label.
    pub fn of(schema: &LabelSchema) -> Self {
        Self {
            kind: schema.kind(),
            implies: schema.implies().to_vec(),
        }
    }
}

/// Why a node's labels cannot stand together.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LabelKindError {
    /// No label of the node is a type label: auxiliary labels only, or none.
    #[error("node has no type label: {labels:?} are all auxiliary")]
    NoTypeLabel {
        /// The node's labels, as written.
        labels: Vec<String>,
    },
    /// Two type labels, neither implying the other.
    #[error("node has two type labels :{first} and :{second}, neither implying the other")]
    TwoTypeLabels {
        /// One of them.
        first: String,
        /// The other.
        second: String,
    },
    /// Type labels imply one another in a cycle, so none is primary.
    #[error("labels {labels:?} imply one another in a cycle")]
    ImplicationCycle {
        /// The labels on the cycle.
        labels: Vec<String>,
    },
}

/// A node's labels as the engine stores them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeLabels {
    /// The primary label first, then every implied and auxiliary label in
    /// byte order, each once.
    pub labels: Vec<String>,
}

impl NodeLabels {
    /// The node's primary label.
    pub fn primary(&self) -> &str {
        // `resolve_node_labels` returns a value only with a primary label.
        &self.labels[0]
    }
}

/// Resolve `labels`, written in any order, into the labels a node stores:
/// its single type label first, then the labels it implies (added when not
/// written) and the auxiliary ones. `facts` answers for each label the node
/// carries or its type implies.
///
/// # Errors
///
/// The outer error is `facts`' own. The inner one says why the labels cannot
/// type a node.
pub fn resolve_node_labels<E>(
    labels: &[String],
    mut facts: impl FnMut(&str) -> Result<LabelFacts, E>,
) -> Result<Result<NodeLabels, LabelKindError>, E> {
    // One undeclared or plain type label, the common case, is stored as is.
    if let [only] = labels {
        let only_facts = facts(only)?;
        if only_facts.kind == LabelKind::Type && only_facts.implies.is_empty() {
            return Ok(Ok(NodeLabels {
                labels: labels.to_vec(),
            }));
        }
    }

    let mut written: SmallVec<[(&str, LabelFacts); 4]> = SmallVec::new();
    for label in labels {
        if written.iter().any(|(l, _)| *l == label.as_str()) {
            continue;
        }
        written.push((label.as_str(), facts(label)?));
    }

    // Each written type label with the labels it implies, transitively.
    let mut types: SmallVec<[(&str, Vec<String>); 4]> = SmallVec::new();
    for (label, label_facts) in &written {
        if label_facts.kind == LabelKind::Type {
            let closure = match implied_closure(label, label_facts, &mut facts)? {
                Ok(closure) => closure,
                Err(cycle) => return Ok(Err(cycle)),
            };
            types.push((label, closure));
        }
    }
    if types.is_empty() {
        return Ok(Err(LabelKindError::NoTypeLabel {
            labels: labels.to_vec(),
        }));
    }

    // The primary label is the type no other written type implies.
    let mut primary: Option<usize> = None;
    for (at, (label, _)) in types.iter().enumerate() {
        let implied = types
            .iter()
            .any(|(other, closure)| other != label && closure.iter().any(|c| c == label));
        if implied {
            continue;
        }
        if let Some(first) = primary {
            return Ok(Err(LabelKindError::TwoTypeLabels {
                first: types[first].0.to_string(),
                second: (*label).to_string(),
            }));
        }
        primary = Some(at);
    }
    let Some(primary) = primary else {
        // Every written type implies another: they form a cycle.
        return Ok(Err(LabelKindError::ImplicationCycle {
            labels: types.iter().map(|(l, _)| (*l).to_string()).collect(),
        }));
    };
    let (primary, closure) = &types[primary];

    // Every other written type is implied by the primary one: with a single
    // type no other implies, each is reached from it.
    let mut rest: Vec<String> = closure.clone();
    for (label, label_facts) in &written {
        if *label != *primary && label_facts.kind == LabelKind::Auxiliary {
            rest.push((*label).to_string());
        }
    }
    rest.sort_unstable();
    rest.dedup();
    let mut stored = Vec::with_capacity(rest.len() + 1);
    stored.push((*primary).to_string());
    stored.extend(rest);
    Ok(Ok(NodeLabels { labels: stored }))
}

/// The labels `label` implies, transitively, without itself.
fn implied_closure<E>(
    label: &str,
    label_facts: &LabelFacts,
    facts: &mut impl FnMut(&str) -> Result<LabelFacts, E>,
) -> Result<Result<Vec<String>, LabelKindError>, E> {
    let mut closure: Vec<String> = Vec::new();
    let mut pending: Vec<String> = label_facts.implies.clone();
    while let Some(next) = pending.pop() {
        if next == label {
            let mut labels = closure;
            labels.push(label.to_string());
            return Ok(Err(LabelKindError::ImplicationCycle { labels }));
        }
        if closure.contains(&next) {
            continue;
        }
        pending.extend(facts(&next)?.implies);
        closure.push(next);
    }
    Ok(Ok(closure))
}

/// Why a label declaration cannot be accepted.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LabelDeclarationError {
    /// An auxiliary label declares something only a type label may.
    #[error("auxiliary label :{label} cannot declare {what}")]
    AuxiliaryDeclares {
        /// The label.
        label: String,
        /// What it declares: properties, TEMPORAL, a table key or IMPLIES.
        what: &'static str,
    },
    /// A type label names an auxiliary label in `IMPLIES`.
    #[error(":{label} cannot imply :{implied}, an auxiliary label")]
    ImpliesAuxiliary {
        /// The implying label.
        label: String,
        /// The auxiliary label it names.
        implied: String,
    },
    /// `IMPLIES` leads back to the label itself.
    #[error(":{label} implies itself through {path:?}")]
    ImpliesItself {
        /// The label.
        label: String,
        /// The labels on the way back to it.
        path: Vec<String>,
    },
    /// An implied label declares a property the implying label does not
    /// declare with the same type.
    #[error(
        ":{label} implies :{implied}, whose property `{property}` ({declared}) \
         :{label} does not declare with that type"
    )]
    ImpliedPropertyMissing {
        /// The implying label.
        label: String,
        /// The implied label declaring the property.
        implied: String,
        /// The property.
        property: String,
        /// Its type as the implied label declares it.
        declared: String,
    },
}

/// Check `schema` as a declaration: what an auxiliary label may declare,
/// and for a type label that its `IMPLIES` reaches neither itself nor an
/// auxiliary label and that it declares every property an implied label
/// declares, with the same type. `schema_of` answers for the other labels,
/// `None` for one without a definition.
///
/// # Errors
///
/// The outer error is `schema_of`'s own.
pub fn check_label_declaration<E>(
    schema: &LabelSchema,
    mut schema_of: impl FnMut(&str) -> Result<Option<LabelSchema>, E>,
) -> Result<Result<(), LabelDeclarationError>, E> {
    if schema.is_auxiliary() {
        let what = if !schema.properties.is_empty() {
            Some("properties")
        } else if schema.temporal {
            Some("TEMPORAL")
        } else if schema.is_table() {
            Some("a table key")
        } else if !schema.implies().is_empty() {
            Some("IMPLIES")
        } else {
            // Explicit constraints are allowed: one declared on the label
            // binds every node carrying it.
            None
        };
        return Ok(match what {
            Some(what) => Err(LabelDeclarationError::AuxiliaryDeclares {
                label: schema.name.clone(),
                what,
            }),
            None => Ok(()),
        });
    }

    // Walk the closure breadth-first, remembering how each label was reached
    // so a cycle names its path.
    let mut seen: Vec<(String, Vec<String>)> = Vec::new();
    let mut pending: std::collections::VecDeque<(String, Vec<String>)> = schema
        .implies()
        .iter()
        .map(|l| (l.clone(), Vec::new()))
        .collect();
    while let Some((label, mut path)) = pending.pop_front() {
        if label == schema.name {
            return Ok(Err(LabelDeclarationError::ImpliesItself {
                label: schema.name.clone(),
                path,
            }));
        }
        if seen.iter().any(|(l, _)| *l == label) {
            continue;
        }
        let implied = schema_of(&label)?;
        if let Some(implied) = &implied {
            if implied.is_auxiliary() {
                return Ok(Err(LabelDeclarationError::ImpliesAuxiliary {
                    label: schema.name.clone(),
                    implied: label,
                }));
            }
            if let Some(conflict) = implied_property_conflict(schema, implied) {
                return Ok(Err(conflict));
            }
        }
        path.push(label.clone());
        if let Some(implied) = implied {
            for next in implied.implies() {
                pending.push_back((next.clone(), path.clone()));
            }
        }
        seen.push((label, path));
    }
    Ok(Ok(()))
}

/// The first property `implied` declares that `implying` does not declare
/// with the same type, as the error refusing the pair; `None` when every one
/// is there.
pub fn implied_property_conflict(
    implying: &LabelSchema,
    implied: &LabelSchema,
) -> Option<LabelDeclarationError> {
    implied.properties.values().find_map(|property| {
        let same = implying
            .properties
            .get(&property.name)
            .is_some_and(|p| p.property_type == property.property_type);
        (!same).then(|| LabelDeclarationError::ImpliedPropertyMissing {
            label: implying.name.clone(),
            implied: implied.name.clone(),
            property: property.name.clone(),
            declared: property.property_type.to_string(),
        })
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
