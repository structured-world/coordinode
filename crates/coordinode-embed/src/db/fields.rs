//! The database's field dictionary: the verified view queries read from, and
//! the registrar that publishes new bindings through the write pipeline.
//!
//! The authoritative bindings live in the Schema partition and change only
//! through applied [`MetadataCommand`]s. This handle keeps an immutable view
//! of them and refreshes it whenever the engine's dictionary generation has
//! moved, which covers every way a binding arrives: this handle's own
//! registration, a replica applying the leader's, a replay after a restart
//! and a snapshot installed over the store. A reader compares one counter per
//! query; no lock is held while a query runs or while a registration waits
//! for the pipeline.

use std::sync::Arc;

// no-std: spin::RwLock
use parking_lot::RwLock;

use coordinode_core::graph::intern::{
    DictionaryError, FieldInterner, FieldRegistrar, MAX_REGISTRATION_BATCH, validate_registration,
};
use coordinode_core::txn::proposal::{
    MetadataCommand, Mutation, ProposalError, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::metadata::{field_bindings_above, load_field_dictionary};
use coordinode_storage::error::StorageError;

/// Schema key the dictionary was kept under by releases that wrote it as one
/// whole-table value after the data.
const LEGACY_DICTIONARY_KEY: &[u8] = b"meta:field_interner";

/// Why the log did not publish a registration, as the statement that needed
/// it is refused. One this member stopped leading under may still be taken
/// by another leader, which only binds names early; the statement waiting
/// for it ran nothing, so it is refused as any write on a member that does
/// not lead is, not with an unknown outcome.
fn registration_refusal(e: ProposalError) -> DictionaryError {
    match e {
        ProposalError::NotLeader { leader_id } => DictionaryError::NotLeader { leader_id },
        ProposalError::LeadershipLost { .. } | ProposalError::ShuttingDown => {
            DictionaryError::NotLeader { leader_id: None }
        }
        // Refused as any write past the limit is, and reported the same way.
        ProposalError::CapacityExhausted {
            endpoint_id,
            used_bytes,
            hard_limit_bytes,
        } => DictionaryError::CapacityExhausted {
            endpoint_id,
            used_bytes,
            hard_limit_bytes,
        },
        other => DictionaryError::Registration(other.to_string()),
    }
}

/// A published view and the engine generations it covers.
struct Published {
    view: FieldInterner,
    generation: u64,
    epoch: u64,
}

/// The field dictionary of one database.
pub struct FieldDictionary {
    engine: Arc<StorageEngine>,
    pipeline: Arc<dyn ProposalPipeline>,
    proposal_ids: Arc<ProposalIdGenerator>,
    oracle: Arc<TimestampOracle>,
    published: RwLock<Published>,
    /// Held while a registration of field names is proposed: a concurrent
    /// first use of the same names waits for it and finds them bound,
    /// instead of proposing them again.
    registering: parking_lot::Mutex<()>,
    /// The same for label identities.
    binding: parking_lot::Mutex<()>,
}

impl FieldDictionary {
    /// Load and verify the stored dictionary.
    ///
    /// # Errors
    ///
    /// The stored records are inconsistent; the store still holds the
    /// dictionary in the format that could lose bindings, which this release
    /// cannot prove complete; or the store holds properties while the
    /// dictionary is empty. Each would serve stored data with no meaning, so
    /// the database refuses to open instead.
    pub(crate) fn open(
        engine: Arc<StorageEngine>,
        pipeline: Arc<dyn ProposalPipeline>,
        proposal_ids: Arc<ProposalIdGenerator>,
        oracle: Arc<TimestampOracle>,
    ) -> Result<Self, StorageError> {
        let generation = engine.field_dictionary_generation();
        let epoch = engine.field_dictionary_epoch();
        if engine
            .get(
                coordinode_storage::engine::partition::Partition::Schema,
                LEGACY_DICTIONARY_KEY,
            )?
            .is_some()
        {
            return Err(DictionaryError::Malformed(
                "the store keeps its field dictionary in the whole-table format, whose \
                 bindings cannot be proven complete; it was written by an earlier release, \
                 which this one does not open"
                    .into(),
            )
            .into());
        }
        let view = load_field_dictionary(&engine)?;
        if view.is_empty() {
            refuse_unbound_properties(&engine)?;
        }
        Ok(Self {
            engine,
            pipeline,
            proposal_ids,
            oracle,
            published: RwLock::new(Published {
                view,
                generation,
                epoch,
            }),
            registering: parking_lot::Mutex::new(()),
            binding: parking_lot::Mutex::new(()),
        })
    }

    /// The current verified view: every binding applied here so far.
    ///
    /// # Errors
    ///
    /// The stored records the refresh read are inconsistent.
    #[inline]
    pub fn current(&self) -> Result<FieldInterner, StorageError> {
        let generation = self.engine.field_dictionary_generation();
        {
            let published = self.published.read();
            if published.generation == generation {
                return Ok(published.view.clone());
            }
        }
        self.refresh()
    }

    #[cold]
    fn refresh(&self) -> Result<FieldInterner, StorageError> {
        let mut published = self.published.write();
        // Read before the records: a binding applied after this is picked
        // up by the next refresh, never skipped.
        let generation = self.engine.field_dictionary_generation();
        let epoch = self.engine.field_dictionary_epoch();
        if published.generation == generation {
            return Ok(published.view.clone());
        }
        published.view = if published.epoch == epoch {
            // Registrations only ever add ids above the frontier.
            let added = field_bindings_above(&self.engine, published.view.frontier())?;
            published.view.extended(added)?
        } else {
            // Adopted or installed bindings can land anywhere: reread all.
            load_field_dictionary(&self.engine)?
        };
        published.generation = generation;
        published.epoch = epoch;
        Ok(published.view.clone())
    }

    fn propose(&self, command: MetadataCommand) -> Result<(), DictionaryError> {
        // Held as not yet logged until the entry is handed over, so no
        // closed bound passes over it meanwhile. The dictionary is Schema
        // records, which consumers are charged for.
        let (commit_ts, held) = self
            .engine
            .pending_commits()
            .obligate_changing(|| self.oracle.next().as_raw())
            .map_err(|_| DictionaryError::Held)?;
        let proposal = RaftProposal {
            id: self.proposal_ids.next(),
            mutations: vec![Mutation::Command(command)],
            commit_ts: Timestamp::from_raw(commit_ts),
            start_ts: Timestamp::from_raw(0),
            bypass_rate_limiter: false,
        };
        // The proposal carries the whole registration.
        held.seal(proposal.id);
        // A registration promises nothing on its own: it is logged ahead of
        // every entry that uses its ids, so the sync those entries wait for
        // covers it, and a crash before that loses it together with them.
        self.pipeline
            .propose_with_concern(&proposal, &self.engine.bookkeeping_concern())
            .map(|_| ())
            .map_err(registration_refusal)
    }

    fn view_or_err(&self) -> Result<FieldInterner, DictionaryError> {
        self.current().map_err(|e| match e {
            StorageError::FieldDictionary(e) => e,
            other => DictionaryError::Registration(other.to_string()),
        })
    }
}

impl FieldRegistrar for FieldDictionary {
    fn register(&self, names: &[&str]) -> Result<Vec<u32>, DictionaryError> {
        let mut view = self.view_or_err()?;
        let missing_from = |view: &FieldInterner| {
            let mut missing: Vec<&str> = Vec::new();
            for &name in names {
                if view.lookup(name).is_none() && !missing.contains(&name) {
                    missing.push(name);
                }
            }
            missing
        };
        if !missing_from(&view).is_empty() {
            // One registration at a time: the names one waited for are
            // bound by the time it runs, and are not proposed again.
            let _registering = self.registering.lock();
            view = self.view_or_err()?;
            for batch in missing_from(&view).chunks(MAX_REGISTRATION_BATCH) {
                validate_registration(batch)?;
                self.propose(MetadataCommand::RegisterFields {
                    names: batch.iter().map(|n| (*n).to_owned()).collect(),
                })?;
            }
            view = self.view_or_err()?;
        }
        names
            .iter()
            .map(|name| {
                view.lookup(name).ok_or_else(|| {
                    // The application refused the batch: nothing else stops a
                    // valid batch from binding a name.
                    if view.frontier() == u32::MAX {
                        DictionaryError::Exhausted
                    } else {
                        DictionaryError::Registration(format!(
                            "{name:?} was not bound; the registration was refused"
                        ))
                    }
                })
            })
            .collect()
    }

    fn adopt(&self, bindings: &FieldInterner) -> Result<(), DictionaryError> {
        let mut pairs: Vec<(String, u32)> = bindings
            .iter()
            .map(|(name, id)| (name.to_owned(), id))
            .collect();
        pairs.sort_unstable_by_key(|&(_, id)| id);
        for batch in pairs.chunks(MAX_REGISTRATION_BATCH) {
            self.propose(MetadataCommand::AdoptFields {
                bindings: batch.to_vec(),
            })?;
        }
        let view = self.view_or_err()?;
        match pairs
            .iter()
            .find(|(name, id)| view.lookup(name) != Some(*id))
        {
            None => Ok(()),
            Some((name, id)) => Err(DictionaryError::Registration(format!(
                "{name:?} could not be bound to {id}: it contradicts a published binding"
            ))),
        }
    }

    fn view(&self) -> Result<FieldInterner, DictionaryError> {
        self.view_or_err()
    }
}

/// The label identities a write uses for the first time are published
/// through the same ordered path as field bindings: each is a metadata
/// command whose application decides the identity, so concurrent first uses
/// converge on one.
impl coordinode_core::schema::label_id::LabelRegistrar for FieldDictionary {
    fn bind_labels(
        &self,
        names: &[&str],
    ) -> Result<Vec<coordinode_core::schema::label_id::LabelId>, DictionaryError> {
        use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
        let store = LocalSchemaStore::new(&self.engine);
        let bound = |store: &LocalSchemaStore<'_>, name: &str| {
            store
                .label_id(name)
                .map_err(|e| DictionaryError::Registration(e.to_string()))
        };
        let missing_from = |store: &LocalSchemaStore<'_>| {
            let mut missing: Vec<&str> = Vec::new();
            for &name in names {
                if bound(store, name)?.is_none() && !missing.contains(&name) {
                    missing.push(name);
                }
            }
            Ok::<_, DictionaryError>(missing)
        };
        if missing_from(&store)?.is_empty() {
            return names
                .iter()
                .map(|name| {
                    bound(&store, name)?.ok_or_else(|| {
                        DictionaryError::Registration(format!("label {name:?} is not bound"))
                    })
                })
                .collect();
        }
        // One binding at a time: the labels one waited for are bound by the
        // time it runs, and are not proposed again.
        let _binding = self.binding.lock();
        let missing = missing_from(&store)?;
        for batch in missing.chunks(MAX_REGISTRATION_BATCH) {
            validate_registration(batch)?;
            // A binding refused because a retirement of the name carries a
            // later timestamp than the command is proposed again, at a fresh
            // one; anything else that leaves a name unbound is reported below.
            for _ in 0..MAX_BINDING_ATTEMPTS {
                self.propose(MetadataCommand::BindLabels {
                    names: batch.iter().map(|n| (*n).to_owned()).collect(),
                })?;
                if batch
                    .iter()
                    .try_fold(true, |all, name| Ok(all && bound(&store, name)?.is_some()))?
                {
                    break;
                }
            }
        }
        names
            .iter()
            .map(|name| {
                bound(&store, name)?.ok_or_else(|| {
                    DictionaryError::Registration(format!(
                        "label {name:?} was not bound; the binding was refused"
                    ))
                })
            })
            .collect()
    }
}

/// Proposals of one label binding batch before its refusal is reported: a
/// batch is refused only while a concurrent retirement of one of its names
/// carries a later timestamp, which a fresh proposal moves past.
const MAX_BINDING_ATTEMPTS: usize = 4;

/// Refuse an empty dictionary over stored properties: the bindings their ids
/// need are gone. Nodes without properties need none, so the check reads node
/// records until the first one that has a property.
fn refuse_unbound_properties(engine: &StorageEngine) -> Result<(), StorageError> {
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_storage::engine::partition::Partition;

    for guard in engine.prefix_scan(Partition::Node, b"node:")? {
        let (_, value) = guard.into_inner()?;
        let Ok(record) = NodeRecord::from_msgpack(&value) else {
            continue;
        };
        if let Some(&id) = record.props.keys().next() {
            return Err(DictionaryError::Unbound(id).into());
        }
    }
    if engine
        .prefix_scan(Partition::EdgeProp, b"edgeprop:")?
        .next()
        .is_some()
    {
        return Err(DictionaryError::Malformed(
            "edge properties are stored but the field dictionary is empty".into(),
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
