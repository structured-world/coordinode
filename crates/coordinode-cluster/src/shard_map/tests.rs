use super::*;

fn sid(n: u32) -> ShardId {
    ShardId(n)
}

fn chunk(start: u64, end: u64, shard: u32) -> ChunkAssignment {
    ChunkAssignment {
        range: ChunkRange { start, end },
        shard: sid(shard),
    }
}

fn table(chunks: Vec<ChunkAssignment>) -> Result<ChunkAssignmentTable, ChunkTableError> {
    ChunkAssignmentTable::from_chunks("Order", 1, chunks)
}

#[test]
fn single_shard_covers_whole_keyspace() {
    let t = ChunkAssignmentTable::single_shard("Order", 1, ShardId::FIRST);
    assert!(t.is_single_shard());
    assert_eq!(t.shard_for(0), ShardId::FIRST);
    assert_eq!(t.shard_for(42), ShardId::FIRST);
    assert_eq!(t.shard_for(u64::MAX), ShardId::FIRST, "max key is covered");
    assert_eq!(t.shards(), vec![ShardId::FIRST]);
    assert_eq!(t.label(), "Order");
    assert_eq!(t.revision(), 1);
}

#[test]
fn multi_chunk_routes_by_range() {
    // [0,100)->s1, [100,1000)->s2, [1000,MAX]->s3.
    let t = table(vec![
        chunk(0, 100, 1),
        chunk(100, 1000, 2),
        chunk(1000, u64::MAX, 3),
    ])
    .expect("valid tiling");

    assert_eq!(t.shard_for(0), sid(1));
    assert_eq!(t.shard_for(99), sid(1));
    assert_eq!(t.shard_for(100), sid(2));
    assert_eq!(t.shard_for(999), sid(2));
    assert_eq!(t.shard_for(1000), sid(3));
    assert_eq!(t.shard_for(u64::MAX), sid(3));
    assert_eq!(t.shards(), vec![sid(1), sid(2), sid(3)]);
    assert!(!t.is_single_shard());
}

#[test]
fn rejects_gap() {
    assert_eq!(
        table(vec![chunk(0, 100, 1), chunk(200, u64::MAX, 2)]),
        Err(ChunkTableError::NotContiguous {
            previous_end: 100,
            start: 200
        })
    );
}

#[test]
fn rejects_overlap() {
    assert_eq!(
        table(vec![chunk(0, 150, 1), chunk(100, u64::MAX, 2)]),
        Err(ChunkTableError::NotContiguous {
            previous_end: 150,
            start: 100
        })
    );
}

#[test]
fn rejects_not_starting_at_zero() {
    assert_eq!(
        table(vec![chunk(1, u64::MAX, 1)]),
        Err(ChunkTableError::FirstNotAtZero(1))
    );
}

#[test]
fn rejects_not_ending_at_max() {
    assert_eq!(
        table(vec![chunk(0, 1000, 1)]),
        Err(ChunkTableError::LastNotAtMax(1000))
    );
}

#[test]
fn rejects_empty() {
    assert_eq!(table(vec![]), Err(ChunkTableError::Empty));
}

#[test]
fn rejects_an_empty_chunk() {
    assert_eq!(
        table(vec![chunk(0, 0, 1), chunk(0, u64::MAX, 2)]),
        Err(ChunkTableError::EmptyRange { start: 0, end: 0 })
    );
}

/// Shard 0 is the NodeId hint sentinel; a chunk naming it would route keys
/// to no shard.
#[test]
fn rejects_the_sentinel_shard() {
    assert_eq!(
        table(vec![chunk(0, u64::MAX, 0)]),
        Err(ChunkTableError::SentinelShard)
    );
}

#[test]
fn shards_are_deduped_and_sorted() {
    let t = table(vec![
        chunk(0, 100, 5),
        chunk(100, 200, 2),
        chunk(200, u64::MAX, 5),
    ])
    .expect("valid");
    assert_eq!(t.shards(), vec![sid(2), sid(5)]);
}

#[test]
fn roundtrips_through_messagepack() {
    let t = ChunkAssignmentTable::from_chunks(
        "Event",
        12,
        vec![chunk(0, 100, 1), chunk(100, u64::MAX, 2)],
    )
    .expect("valid");
    let decoded =
        ChunkAssignmentTable::from_msgpack(&t.to_msgpack().expect("encode")).expect("decode");
    assert_eq!(decoded, t);
    assert_eq!(decoded.label(), "Event");
    assert_eq!(decoded.revision(), 12);
}

/// Stored bytes that do not tile the keyspace are refused on decode, so a
/// table read back from storage routes every key.
#[test]
fn decoding_refuses_chunks_that_do_not_tile_the_keyspace() {
    let stored = ChunkTableRecord {
        label: "Order".to_string(),
        revision: 3,
        chunks: vec![chunk(0, 100, 1)],
    };
    let bytes = rmp_serde::to_vec(&stored).expect("encode");
    let err = ChunkAssignmentTable::from_msgpack(&bytes).expect_err("a gap at the end");
    assert!(err.to_string().contains("last chunk ends at 100"), "{err}");
}
