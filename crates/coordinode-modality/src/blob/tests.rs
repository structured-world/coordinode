use std::sync::Arc;

use super::*;
use coordinode_core::graph::blob::{ContentId, ScopedContent, refs_counter_key};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::CommitContext;

struct Fixture {
    engine: Arc<StorageEngine>,
    oracle: Arc<TimestampOracle>,
    store: LocalBlobStore,
    _dir: tempfile::TempDir,
}

fn open() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine =
        Arc::new(StorageEngine::open_embedded(&config, Arc::clone(&oracle)).expect("open"));
    let store = LocalBlobStore::new(coordinode_storage::blob::BlobStore::direct(
        Arc::clone(&engine),
        Arc::clone(&oracle),
    ));
    Fixture {
        engine,
        oracle,
        store,
        _dir: dir,
    }
}

impl Fixture {
    /// Run `body` in one transaction and commit it.
    fn write(&self, body: impl FnOnce(&LocalBlobStore, &mut Transaction)) {
        let mut txn = Transaction::begin(&self.engine, Some(&self.oracle), self.oracle.next());
        body(&self.store, &mut txn);
        let wc = WriteConcern::majority();
        let ctx = CommitContext {
            write_concern: &wc,
            pipeline: None,
            id_gen: None,
        };
        txn.commit(&ctx).expect("commit");
    }

    /// Run `body` against the latest committed state.
    fn read<R>(&self, body: impl FnOnce(&LocalBlobStore, &Transaction) -> R) -> R {
        let txn = Transaction::begin(&self.engine, Some(&self.oracle), self.oracle.next());
        body(&self.store, &txn)
    }

    fn refs(&self, data: &[u8]) -> i64 {
        let key = refs_counter_key(&ScopedContent::new(
            SharingScope::STORE,
            ContentId::of(data),
        ));
        self.engine
            .get(Partition::Counter, &key)
            .expect("count")
            .map_or(0, |b| {
                coordinode_storage::engine::merge::decode_counter(&b).expect("decode")
            })
    }

    /// The bytes property `prop` of `node` holds, read whole.
    fn bytes(&self, node: NodeId, prop: u32) -> Option<Vec<u8>> {
        self.read(|s, txn| {
            let reader = s.open_blob(txn, node, prop).expect("open")?;
            Some(
                (0..reader.record().blob.chunk_count())
                    .flat_map(|i| reader.chunk(i).expect("chunk"))
                    .collect(),
            )
        })
    }
}

/// A property value published with the node write that holds it reads back
/// whole through the transaction, and counts one reference.
#[test]
fn a_published_property_reads_back() {
    let fx = open();
    let node = NodeId::from_raw(42);
    let payload = vec![0xaa_u8; INLINE_THRESHOLD + 64];
    let upload = fx.store.upload(&payload).expect("upload");
    fx.write(|s, txn| {
        s.put_blob(txn, node, 7, &upload).expect("publish");
    });

    let loaded = fx
        .read(|s, txn| s.get_blob_ref(txn, node, 7).expect("ok"))
        .expect("present");
    assert_eq!(&loaded, upload.blob());
    assert_eq!(fx.bytes(node, 7).expect("present"), payload);
    assert_eq!(fx.refs(&payload), 1);
}

/// Replacing a property releases the value it held in the same commit, and
/// deleting it releases the rest; bytes another property shares stay.
#[test]
fn replace_and_delete_release_in_the_property_commit() {
    let fx = open();
    let node = NodeId::from_raw(11);
    let first = vec![0x33_u8; INLINE_THRESHOLD + 8];
    let second = vec![0x44_u8; INLINE_THRESHOLD + 8];
    for (prop, data) in [(0, &first), (1, &first)] {
        let upload = fx.store.upload(data).expect("upload");
        fx.write(|s, txn| {
            s.put_blob(txn, node, prop, &upload).expect("publish");
        });
    }
    assert_eq!(fx.refs(&first), 2, "two properties hold the same bytes");

    let upload = fx.store.upload(&second).expect("upload");
    fx.write(|s, txn| {
        s.put_blob(txn, node, 0, &upload).expect("replace");
    });
    assert_eq!(fx.refs(&first), 1);
    assert_eq!(fx.refs(&second), 1);

    fx.write(|s, txn| {
        s.delete_blob_ref(txn, node, 0).expect("delete");
    });
    assert_eq!(fx.refs(&second), 0);
    assert!(fx.bytes(node, 0).is_none(), "the property is gone");
    assert_eq!(fx.bytes(node, 1).expect("the other property stays"), first);
}

/// An upload whose property write is abandoned holds nothing afterwards and
/// can no longer be published.
#[test]
fn an_abandoned_upload_cannot_be_published() {
    let fx = open();
    let node = NodeId::from_raw(5);
    let payload = vec![0x55_u8; INLINE_THRESHOLD + 1];
    let upload = fx.store.upload(&payload).expect("upload");
    fx.store.abandon(&upload).expect("abandon");
    let mut txn = Transaction::begin(&fx.engine, Some(&fx.oracle), fx.oracle.next());
    let refused = fx.store.put_blob(&mut txn, node, 0, &upload);
    assert!(
        matches!(
            refused,
            Err(StoreError::Blob(
                coordinode_storage::blob::BlobError::Fenced
            ))
        ),
        "{refused:?}"
    );
    assert_eq!(fx.refs(&payload), 0);
}

#[test]
fn multiple_blob_properties_per_node_stay_distinct() {
    let fx = open();
    let node = NodeId::from_raw(7);
    let payload_a = vec![0x01_u8; INLINE_THRESHOLD + 4];
    let payload_b = vec![0x02_u8; INLINE_THRESHOLD + 4];
    let upload_a = fx.store.upload(&payload_a).expect("a");
    let upload_b = fx.store.upload(&payload_b).expect("b");
    fx.write(|s, txn| {
        s.put_blob(txn, node, 1, &upload_a).expect("ref a");
        s.put_blob(txn, node, 2, &upload_b).expect("ref b");
    });
    assert_eq!(fx.bytes(node, 1).expect("a"), payload_a);
    assert_eq!(fx.bytes(node, 2).expect("b"), payload_b);
}

#[test]
fn corrupt_owner_record_surfaces_as_decode_error() {
    let fx = open();
    let node_id = NodeId::from_raw(99);
    fx.write(|_s, txn| {
        txn.put(
            Partition::BlobRef,
            &encode_blobref_key(node_id, 0),
            &[0xde, 0xad, 0xbe, 0xef],
        )
        .expect("inject");
    });
    let err = fx.read(|s, txn| s.get_blob_ref(txn, node_id, 0).expect_err("must error"));
    assert!(matches!(
        err,
        StoreError::Decode {
            kind: "blob owner record",
            ..
        }
    ));
}
