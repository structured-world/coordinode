//! Label kinds: which label of a node is its type, and what a type label may
//! declare about the labels it implies.
//!
//! A node carries exactly one type label as its primary label, whatever the
//! order its labels are written in. The labels that type implies (transitively)
//! go with it, and any other label must be auxiliary. A record stores the
//! primary label first, then the node's own auxiliary labels; the implied ones
//! belong to the type and are derived from its compiled [`ImpliedLabels`], so
//! a change of what a type implies rewrites no record. [`Membership`] is the
//! node's complete label set.

use alloc::borrow::Cow;

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
pub struct NodeLabels<'a> {
    /// The primary label first, then the node's auxiliary labels in byte
    /// order, each once; the labels the type implies are not among them.
    /// Borrowed when the labels as written are already the stored ones.
    pub labels: Cow<'a, [String]>,
}

impl NodeLabels<'_> {
    /// The node's primary label.
    pub fn primary(&self) -> &str {
        // `resolve_node_labels` returns a value only with a primary label.
        &self.labels[0]
    }

    /// The same labels, owned.
    pub fn into_owned(self) -> NodeLabels<'static> {
        NodeLabels {
            labels: Cow::Owned(self.labels.into_owned()),
        }
    }
}

/// Resolve `labels`, written in any order, into the labels a node stores:
/// its single type label first, then its auxiliary labels. A written label
/// the type implies is the type's, not stored. `facts` answers for each
/// label the node carries or its type implies.
///
/// # Errors
///
/// The outer error is `facts`' own. The inner one says why the labels cannot
/// type a node.
pub fn resolve_node_labels<'a, E>(
    labels: &'a [String],
    mut facts: impl FnMut(&str) -> Result<LabelFacts, E>,
) -> Result<Result<NodeLabels<'a>, LabelKindError>, E> {
    // One type label, the common case, is stored as is.
    if let [only] = labels {
        if facts(only)?.kind == LabelKind::Type {
            return Ok(Ok(NodeLabels {
                labels: Cow::Borrowed(labels),
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
    let (primary, _) = &types[primary];

    // Every other written type is implied by the primary one: with a single
    // type no other implies, each is reached from it. Those belong to the
    // type, not to the node, and are not stored; the auxiliary labels are
    // the node's own.
    let mut auxiliary: SmallVec<[&str; 4]> = written
        .iter()
        .filter(|(label, label_facts)| {
            *label != *primary && label_facts.kind == LabelKind::Auxiliary
        })
        .map(|(label, _)| *label)
        .collect();
    auxiliary.sort_unstable();
    let canonical = labels.len() == auxiliary.len() + 1
        && labels[0] == *primary
        && labels[1..].iter().zip(&auxiliary).all(|(l, a)| l == a);
    if canonical {
        return Ok(Ok(NodeLabels {
            labels: Cow::Borrowed(labels),
        }));
    }
    let mut stored = Vec::with_capacity(auxiliary.len() + 1);
    stored.push((*primary).to_string());
    stored.extend(auxiliary.iter().map(|l| (*l).to_string()));
    Ok(Ok(NodeLabels {
        labels: Cow::Owned(stored),
    }))
}

/// The labels type label `label` implies, transitively, without itself, in
/// byte order: what every node of the type carries besides its stored
/// labels. `facts` answers for `label` and each label it reaches.
///
/// # Errors
///
/// The outer error is `facts`' own; the inner one names an implication
/// cycle.
pub fn compile_implied<E>(
    label: &str,
    mut facts: impl FnMut(&str) -> Result<LabelFacts, E>,
) -> Result<Result<ImpliedLabels, LabelKindError>, E> {
    let own = facts(label)?;
    if own.implies.is_empty() {
        return Ok(Ok(ImpliedLabels::default()));
    }
    Ok(
        implied_closure(label, &own, &mut facts)?.map(|mut closure| {
            closure.sort_unstable();
            ImpliedLabels(closure.into_boxed_slice())
        }),
    )
}

/// The labels a type implies, transitively, in byte order: compiled once
/// from a catalog view and shared by every node of the type read under it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ImpliedLabels(Box<[String]>);

impl ImpliedLabels {
    /// The labels, in byte order.
    pub fn as_slice(&self) -> &[String] {
        &self.0
    }

    /// Whether `label` is among them.
    pub fn contains(&self, label: &str) -> bool {
        self.0.binary_search_by(|l| l.as_str().cmp(label)).is_ok()
    }
}

/// The compiled [`ImpliedLabels`] of each type met under one catalog view,
/// compiled once per type and shared by every node of it read under that
/// view: a membership test after the first costs no catalog read and no
/// walk of the implication graph.
#[derive(Debug, Clone, Default)]
pub struct ImpliedCache {
    compiled: rustc_hash::FxHashMap<String, ImpliedLabels>,
}

impl ImpliedCache {
    /// Compile what `primary` implies under `facts`, unless it was.
    ///
    /// # Errors
    ///
    /// The outer error is `facts`' own; the inner one names an implication
    /// cycle.
    pub fn compile<E>(
        &mut self,
        primary: &str,
        facts: impl FnMut(&str) -> Result<LabelFacts, E>,
    ) -> Result<Result<(), LabelKindError>, E> {
        if self.compiled.contains_key(primary) {
            return Ok(Ok(()));
        }
        Ok(compile_implied(primary, facts)?.map(|implied| {
            self.compiled.insert(primary.to_string(), implied);
        }))
    }

    /// What `primary` implies, when compiled.
    pub fn implied(&self, primary: &str) -> Option<&ImpliedLabels> {
        self.compiled.get(primary)
    }

    /// The membership of a node storing `stored`, when its type was
    /// compiled; a record without labels has the empty membership.
    pub fn membership<'a>(&'a self, stored: &'a [String]) -> Option<Membership<'a>> {
        match stored.first() {
            None => Some(Membership::stored_only(stored)),
            Some(primary) => self
                .compiled
                .get(primary.as_str())
                .map(|implied| Membership::new(stored, implied)),
        }
    }

    /// Forget every compiled type: the catalog view they were compiled
    /// under no longer holds.
    pub fn clear(&mut self) {
        self.compiled.clear();
    }
}

/// The labels a node carries: the ones its record stores (its type first,
/// then its own auxiliary labels in byte order) and the ones its type
/// implies, borrowed from both without building a list.
#[derive(Debug, Clone, Copy)]
pub struct Membership<'a> {
    stored: &'a [String],
    implied: &'a [String],
}

impl<'a> Membership<'a> {
    /// The membership of a node storing `stored` whose type implies
    /// `implied`.
    pub fn new(stored: &'a [String], implied: &'a ImpliedLabels) -> Self {
        Self {
            stored,
            implied: implied.as_slice(),
        }
    }

    /// The membership of a node whose type implies nothing.
    pub fn stored_only(stored: &'a [String]) -> Self {
        Self {
            stored,
            implied: &[],
        }
    }

    /// The node's type label; `None` for a record without labels.
    pub fn primary(&self) -> Option<&'a str> {
        self.stored.first().map(String::as_str)
    }

    /// Whether the node carries `label`.
    pub fn contains(&self, label: &str) -> bool {
        self.stored.iter().any(|l| l == label)
            || self
                .implied
                .binary_search_by(|l| l.as_str().cmp(label))
                .is_ok()
    }

    /// Every label the node carries, each once: its type first, then the
    /// implied and auxiliary ones merged in byte order.
    pub fn iter(&self) -> impl Iterator<Item = &'a str> + 'a {
        let (primary, auxiliary) = self
            .stored
            .split_first()
            .map_or((None, &[][..]), |(p, a)| (Some(p.as_str()), a));
        primary.into_iter().chain(MergedLabels {
            left: self.implied,
            right: auxiliary,
        })
    }

    /// How many labels the node carries.
    pub fn len(&self) -> usize {
        self.iter().count()
    }

    /// Whether the node carries no label.
    pub fn is_empty(&self) -> bool {
        self.stored.is_empty()
    }
}

/// Two byte-ordered label lists merged in byte order, a label in both
/// yielded once.
struct MergedLabels<'a> {
    left: &'a [String],
    right: &'a [String],
}

impl<'a> Iterator for MergedLabels<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        match (self.left.split_first(), self.right.split_first()) {
            (Some((l, left)), Some((r, right))) => match l.cmp(r) {
                core::cmp::Ordering::Less => {
                    self.left = left;
                    Some(l)
                }
                core::cmp::Ordering::Greater => {
                    self.right = right;
                    Some(r)
                }
                core::cmp::Ordering::Equal => {
                    self.left = left;
                    self.right = right;
                    Some(l)
                }
            },
            (Some((l, left)), None) => {
                self.left = left;
                Some(l)
            }
            (None, Some((r, right))) => {
                self.right = right;
                Some(r)
            }
            (None, None) => None,
        }
    }
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
