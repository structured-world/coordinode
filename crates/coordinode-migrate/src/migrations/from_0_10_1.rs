//! From release 0.10.1 to this build.
//!
//! 0.10.1 kept every tree of a store at the top of the endpoint's
//! directory and wrote units without naming a namespace. This build keeps
//! the deployment's own trees in the system set and a namespace's data in
//! that namespace's set, and refuses a unit that writes a namespace's data
//! before naming the namespace, and reads the deployment's own records from
//! a Deployment tree where 0.10.1 kept them in the Schema tree. A 0.10.1
//! store holds one namespace's data: its trees move into that namespace's
//! set, each unit of its journal names the namespace first, and the
//! deployment's records move to the Deployment tree.

mod deployment_records;
mod layout;
mod revision_stamps;
mod unit_namespace;

use coordinode_storage::engine::scope::NamespaceId;

use super::ReleaseStep;

/// In order: the journal first, read by this build's decoder from then on;
/// the trees, written with revisions while still at the top of the store,
/// which this build refuses to open; only then the move into the sets that
/// makes the store open, and the records moved by an open.
pub(super) const STEP: ReleaseStep = ReleaseStep {
    from: "0.10.1",
    migrations: &[
        unit_namespace::MIGRATION,
        revision_stamps::MIGRATION,
        layout::MIGRATION,
        deployment_records::MIGRATION,
    ],
};

/// The namespace a 0.10.1 store's data becomes: the first one a catalog
/// hands out, which this build creates for a store holding no catalog.
const NAMESPACE: NamespaceId = NamespaceId::from_raw(1);
