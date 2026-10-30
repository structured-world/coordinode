//! Label catalog fixtures for this crate's tests, written as the ordered
//! application of a label binding writes them.

use coordinode_core::schema::definition::encode_label_current_revision_key;
use coordinode_core::schema::label_id::{LabelBinding, LabelId};
use coordinode_core::txn::proposal::Mutation;

use crate::engine::batch::WriteBatch;
use crate::engine::core::StorageEngine;
use crate::engine::metadata::label_binding_effects;
use crate::engine::partition::Partition;

/// The identity label `name` is bound to, bound implicitly on its first use
/// with the decision a label binding command applies.
#[allow(clippy::expect_used)]
pub(crate) fn bind(engine: &StorageEngine, name: &str) -> LabelId {
    let key = encode_label_current_revision_key(name);
    let live = |engine: &StorageEngine| {
        engine
            .get(Partition::Schema, &key)
            .expect("binding")
            .and_then(|bytes| LabelBinding::decode(&bytes).expect("a binding").live_id())
    };
    if let Some(id) = live(engine) {
        return id;
    }
    let mut batch = WriteBatch::new(engine);
    for effect in label_binding_effects(engine, &[name.to_owned()], u64::MAX).expect("decide") {
        if let Mutation::Put { key, value, .. } = effect {
            batch.put(Partition::Schema, key, value);
        }
    }
    batch.commit().expect("bind");
    live(engine).expect("bound")
}
