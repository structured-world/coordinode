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

#[test]
fn single_shard_covers_whole_keyspace() {
    let t = ChunkAssignmentTable::single_shard(ShardId::ZERO);
    assert!(t.is_single_shard());
    assert_eq!(t.shard_for(0), ShardId::ZERO);
    assert_eq!(t.shard_for(42), ShardId::ZERO);
    assert_eq!(t.shard_for(u64::MAX), ShardId::ZERO, "max key is covered");
    assert_eq!(t.shards(), vec![ShardId::ZERO]);
}

#[test]
fn multi_chunk_routes_by_range() {
    // [0,100)->s1, [100,1000)->s2, [1000,MAX]->s3.
    let t = ChunkAssignmentTable::from_chunks(vec![
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
    // [0,100) then [200,MAX] — gap [100,200) -> invalid.
    assert!(
        ChunkAssignmentTable::from_chunks(vec![chunk(0, 100, 1), chunk(200, u64::MAX, 2)])
            .is_none()
    );
}

#[test]
fn rejects_overlap() {
    // [0,150) and [100,MAX] overlap.
    assert!(
        ChunkAssignmentTable::from_chunks(vec![chunk(0, 150, 1), chunk(100, u64::MAX, 2)])
            .is_none()
    );
}

#[test]
fn rejects_not_starting_at_zero() {
    assert!(ChunkAssignmentTable::from_chunks(vec![chunk(1, u64::MAX, 1)]).is_none());
}

#[test]
fn rejects_not_ending_at_max() {
    assert!(ChunkAssignmentTable::from_chunks(vec![chunk(0, 1000, 1)]).is_none());
}

#[test]
fn rejects_empty() {
    assert!(ChunkAssignmentTable::from_chunks(vec![]).is_none());
}

#[test]
fn shards_are_deduped_and_sorted() {
    // Two chunks owned by the same shard, plus another — dedup + sort.
    let t = ChunkAssignmentTable::from_chunks(vec![
        chunk(0, 100, 5),
        chunk(100, 200, 2),
        chunk(200, u64::MAX, 5),
    ])
    .expect("valid");
    assert_eq!(t.shards(), vec![sid(2), sid(5)]);
}

#[test]
fn roundtrips_through_messagepack() {
    let t = ChunkAssignmentTable::from_chunks(vec![chunk(0, 100, 1), chunk(100, u64::MAX, 2)])
        .expect("valid");
    let bytes = rmp_serde::to_vec(&t).expect("encode");
    let decoded: ChunkAssignmentTable = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(t, decoded);
}
