use super::*;

#[test]
fn insert_get_and_replace() {
    let map = IdMap::with_capacity(16);
    assert_eq!(map.get(7), None);
    assert_eq!(map.insert(7, 1), None);
    assert_eq!(map.get(7), Some(1));
    assert!(map.contains(7));
    // Re-pointing an id returns the index it pointed at.
    assert_eq!(map.insert(7, 5), Some(1));
    assert_eq!(map.get(7), Some(5));
    assert_eq!(map.len(), 1);
}

#[test]
fn ids_spread_over_shards_and_len_sums_them() {
    let map = IdMap::with_capacity(1000);
    for id in 0..1000u64 {
        map.insert(id, id as usize);
    }
    assert_eq!(map.len(), 1000);
    let used = map.shards.iter().filter(|s| !s.read().is_empty()).count();
    assert!(used > SHARDS / 2, "sequential ids crowd {used} shards");
    for id in 0..1000u64 {
        assert_eq!(map.get(id), Some(id as usize));
    }
}

/// Concurrent inserts of distinct ids all land; the last write per id wins.
#[test]
fn concurrent_inserts_land() {
    let map = IdMap::with_capacity(0);
    std::thread::scope(|s| {
        for t in 0..8u64 {
            let map = &map;
            s.spawn(move || {
                for i in 0..500u64 {
                    map.insert(t * 1000 + i, (t * 1000 + i) as usize);
                }
            });
        }
    });
    assert_eq!(map.len(), 4000);
    assert_eq!(map.get(7 * 1000 + 499), Some(7499));
}

#[test]
fn clear_empties_every_shard() {
    let mut map = IdMap::with_capacity(8);
    for id in 0..100u64 {
        map.insert(id, 0);
    }
    map.clear();
    assert_eq!(map.len(), 0);
    assert!(!map.contains(5));
}
