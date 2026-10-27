use super::*;

/// The identity carries the length as well as the digest, so content of a
/// different length is different content even before the digest is compared.
#[test]
fn content_id_records_digest_and_length() {
    let id = ContentId::of(b"hello world");
    assert_eq!(id.len(), 11);
    assert_eq!(id.algorithm(), DigestAlgorithm::Sha256);
    assert_eq!(id, ContentId::of(b"hello world"));
    assert_ne!(id, ContentId::of(b"hello world!"));
}

/// `matches` is the read path's integrity check: a payload whose bytes are
/// not the content, by length or by digest, must not pass it.
#[test]
fn matches_refuses_other_bytes() {
    let id = ContentId::of(b"payload");
    assert!(id.matches(b"payload"));
    assert!(!id.matches(b"payloaD"));
    assert!(!id.matches(b"payload "));
}

/// The key form decodes back to the same identity, and a truncated or
/// unknown-algorithm form is refused rather than read as some other content.
#[test]
fn content_id_key_form_round_trips_and_refuses_damage() {
    let id = ContentId::of(b"chunk");
    let mut raw = Vec::new();
    id.encode_into(&mut raw);
    assert_eq!(raw.len(), ContentId::ENCODED_LEN);
    assert_eq!(ContentId::decode(&raw), Some(id));
    assert_eq!(ContentId::decode(&raw[..raw.len() - 1]), None);
    raw[0] = 0xEE;
    assert_eq!(ContentId::decode(&raw), None);
}

#[test]
fn scoped_content_key_form_round_trips() {
    let scoped = ScopedContent::new(SharingScope::from_raw(7), ContentId::of(b"x"));
    let key = catalog_key(&scoped);
    let decoded = ScopedContent::decode(&key[CATALOG_PREFIX.len()..]).expect("decode");
    assert_eq!(decoded, scoped);
}

/// Two generations of one content have different payload keys, which is what
/// keeps a cleanup of a retired copy away from a later copy of the same bytes.
#[test]
fn payload_keys_differ_by_generation() {
    let scoped = ScopedContent::new(SharingScope::STORE, ContentId::of(b"same"));
    assert_ne!(payload_key(&scoped, 1), payload_key(&scoped, 2));
    assert!(payload_key(&scoped, 1).starts_with(PAYLOAD_PREFIX));
}

/// The same bytes in two scopes are two catalog entries: a digest is never
/// authority to share across scopes.
#[test]
fn scopes_separate_catalog_entries() {
    let content = ContentId::of(b"shared bytes");
    let a = ScopedContent::new(SharingScope::from_raw(1), content);
    let b = ScopedContent::new(SharingScope::from_raw(2), content);
    assert_ne!(catalog_key(&a), catalog_key(&b));
    assert_ne!(refs_counter_key(&a), refs_counter_key(&b));
    assert_ne!(refs_counter_key(&a), pins_counter_key(&a));
}

/// Releases by different owners, or by one owner at different incarnations,
/// write different candidate keys, so concurrent releases never conflict on
/// one record.
#[test]
fn candidate_keys_are_per_release() {
    let scoped = ScopedContent::new(SharingScope::STORE, ContentId::of(b"c"));
    let first = candidate_key(10, &scoped, b"blobmeta:a", 1);
    assert_ne!(first, candidate_key(10, &scoped, b"blobmeta:b", 1));
    assert_ne!(first, candidate_key(10, &scoped, b"blobmeta:a", 2));
}

/// Candidates sort by release time whatever their content, which is what lets
/// a collector stop at the first one still in its grace; the key gives back
/// the time and content it was written with.
#[test]
fn candidate_keys_sort_by_release_time_and_decode() {
    let early = ScopedContent::new(SharingScope::STORE, ContentId::of(b"zzz"));
    let late = ScopedContent::new(SharingScope::STORE, ContentId::of(b"aaa"));
    let early_key = candidate_key(5, &early, b"blobmeta:a", 1);
    let late_key = candidate_key(6, &late, b"blobmeta:a", 1);
    assert!(early_key < late_key);
    assert_eq!(
        decode_candidate_key(&early_key),
        Some(Candidate {
            released_at_micros: 5,
            content: early,
        })
    );
    // The bound past time 5 sorts after every candidate released at 5 and
    // before every later one.
    let bound = candidates_released_after(5);
    assert!(early_key < bound && bound < late_key);
    assert_eq!(decode_candidate_key(b"blobcand:short"), None);
    assert_eq!(decode_candidate_key(&staging_key(1)), None);
}

/// A time read before the epoch sorts as the epoch rather than wrapping to
/// the far future, where a collector would never reach it.
#[test]
fn a_negative_release_time_sorts_as_the_epoch() {
    let scoped = ScopedContent::new(SharingScope::STORE, ContentId::of(b"c"));
    let key = candidate_key(-7, &scoped, b"blobmeta:a", 1);
    assert_eq!(
        decode_candidate_key(&key).map(|c| c.released_at_micros),
        Some(0)
    );
    assert!(key < candidate_key(1, &scoped, b"blobmeta:a", 1));
}

/// Retired and payload keys give back the exact content and generation, so a
/// cleanup reaches only the copy it names; a key of another length is refused.
#[test]
fn retired_and_payload_keys_decode_their_generation() {
    let scoped = ScopedContent::new(SharingScope::from_raw(3), ContentId::of(b"r"));
    assert_eq!(
        decode_retired_key(&retired_key(&scoped, 9)),
        Some((scoped, 9))
    );
    assert_eq!(
        decode_payload_key(&payload_key(&scoped, 9)),
        Some((scoped, 9))
    );
    let mut long = retired_key(&scoped, 9);
    long.push(0);
    assert_eq!(decode_retired_key(&long), None);
    assert_eq!(decode_payload_key(&retired_key(&scoped, 9)), None);
}

/// Counter keys give back the content they count; a key of the other counter,
/// with a malformed scope or with damaged hex is not read as some content.
#[test]
fn counter_keys_decode_their_content() {
    let scoped = ScopedContent::new(SharingScope::from_raw(12), ContentId::of(b"k"));
    let refs = refs_counter_key(&scoped);
    assert_eq!(decode_counter_key(REFS_COUNTER_PREFIX, &refs), Some(scoped));
    assert_eq!(decode_counter_key(PINS_COUNTER_PREFIX, &refs), None);
    assert_eq!(
        decode_counter_key(PINS_COUNTER_PREFIX, &pins_counter_key(&scoped)),
        Some(scoped)
    );
    assert_eq!(
        decode_counter_key(REFS_COUNTER_PREFIX, b"counter:blob_refs:x:00"),
        None
    );
    let mut damaged = refs.clone();
    let last = damaged.len() - 1;
    damaged[last] = b'g';
    assert_eq!(decode_counter_key(REFS_COUNTER_PREFIX, &damaged), None);
}

#[test]
fn staging_keys_decode_their_id() {
    assert_eq!(decode_staging_key(&staging_key(77)), Some(77));
    assert_eq!(decode_staging_key(b"blobstage:abc"), None);
}

#[test]
fn owner_keys_are_recognised() {
    assert!(is_owner_key(&direct_owner_key(&[1; 16])));
    assert!(is_owner_key(&s3_owner_key("/b/k")));
    assert!(is_owner_key(&encode_blobref_key(NodeId::from_raw(1), 2)));
    assert!(!is_owner_key(&staging_key(1)));
    let scoped = ScopedContent::new(SharingScope::STORE, ContentId::of(b"z"));
    assert!(!is_owner_key(&retired_key(&scoped, 1)));
    assert!(!is_owner_key(&candidate_key(1, &scoped, b"blobmeta:a", 1)));
    assert!(!is_owner_key(COMPACTION_PENDING_KEY));
    assert!(!is_owner_key(&catalog_key(&ScopedContent::new(
        SharingScope::STORE,
        ContentId::of(b"z")
    ))));
}

#[test]
fn chunk_data_small() {
    let data = b"small data";
    let chunks = chunk_data(data, DEFAULT_CHUNK_SIZE);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].1, data);
}

#[test]
fn chunk_data_exact_boundary() {
    let data = vec![0xABu8; DEFAULT_CHUNK_SIZE];
    let chunks = chunk_data(&data, DEFAULT_CHUNK_SIZE);
    assert_eq!(chunks.len(), 1);
}

#[test]
fn chunk_data_multiple_chunks() {
    let data = vec![0xABu8; DEFAULT_CHUNK_SIZE * 3 + 100];
    let chunks = chunk_data(&data, DEFAULT_CHUNK_SIZE);
    assert_eq!(chunks.len(), 4);
    assert_eq!(chunks[0].1.len(), DEFAULT_CHUNK_SIZE);
    assert_eq!(chunks[3].1.len(), 100);
}

#[test]
fn chunk_data_empty() {
    assert!(chunk_data(b"", DEFAULT_CHUNK_SIZE).is_empty());
}

/// Repeated chunks keep their order and multiplicity in the reference, while
/// ownership counts each distinct content once.
#[test]
fn blob_ref_keeps_multiplicity_and_counts_distinct_once() {
    let mut data = vec![0xFFu8; DEFAULT_CHUNK_SIZE * 2];
    data.extend_from_slice(b"tail");
    let (blob_ref, chunks) = create_blob(SharingScope::STORE, &data);
    assert_eq!(chunks.len(), 3);
    assert_eq!(blob_ref.chunks[0], blob_ref.chunks[1]);
    assert_eq!(blob_ref.chunk_count(), 3);
    let distinct = blob_ref.distinct();
    assert_eq!(distinct.len(), 2);
    assert!(
        distinct.windows(2).all(|w| w[0] < w[1]),
        "sorted, no repeats"
    );
    for content in [blob_ref.chunks[0], blob_ref.chunks[2]] {
        assert!(distinct.contains(&ScopedContent::new(SharingScope::STORE, content)));
    }
}

#[test]
fn should_inline_small() {
    assert!(should_inline(&[0u8; 100]));
    assert!(should_inline(&[0u8; INLINE_THRESHOLD - 1]));
}

#[test]
fn should_inline_large() {
    assert!(!should_inline(&[0u8; INLINE_THRESHOLD]));
    assert!(!should_inline(&[0u8; INLINE_THRESHOLD + 1]));
}

#[test]
fn create_blob_roundtrip() {
    let data = vec![0xABu8; DEFAULT_CHUNK_SIZE * 2 + 500];
    let (blob_ref, chunks) = create_blob(SharingScope::STORE, &data);

    assert_eq!(blob_ref.total_size, data.len() as u64);
    assert_eq!(blob_ref.chunk_count(), 3);
    let reassembled: Vec<u8> = chunks.iter().flat_map(|(_, c)| c.iter().copied()).collect();
    assert_eq!(reassembled, data);
}

#[test]
fn records_msgpack_roundtrip() {
    let (blob_ref, _) = create_blob(SharingScope::STORE, b"test data for blob");
    let owner = OwnerRecord {
        incarnation: 9,
        blob: blob_ref.clone(),
    };
    assert_eq!(
        OwnerRecord::from_msgpack(&owner.to_msgpack().expect("encode")).expect("decode"),
        owner
    );
    let entry = CatalogEntry {
        eligible: Representation {
            generation: 3,
            codec: Codec::Raw,
            encoded_len: 18,
        },
    };
    assert_eq!(
        CatalogEntry::from_msgpack(&entry.to_msgpack().expect("encode")).expect("decode"),
        entry
    );
    let staged = StagingRecord {
        admitted_at_micros: 5,
        contents: blob_ref
            .distinct()
            .into_iter()
            .map(|content| StagedContent {
                content,
                generation: 4,
                writes: true,
            })
            .collect(),
    };
    assert_eq!(
        StagingRecord::from_msgpack(&staged.to_msgpack().expect("encode")).expect("decode"),
        staged
    );
}

#[test]
fn blobref_keys_sort_by_node_id() {
    let k1 = encode_blobref_key(NodeId::from_raw(1), 0);
    let k2 = encode_blobref_key(NodeId::from_raw(2), 0);
    assert!(k1 < k2);
}

#[test]
fn custom_chunk_size() {
    let data = vec![0u8; 1000];
    let (blob_ref, chunks) = create_blob_with_chunk_size(SharingScope::STORE, &data, 300);
    assert_eq!(chunks.len(), 4); // 300+300+300+100
    assert_eq!(blob_ref.chunk_count(), 4);
    assert_eq!(blob_ref.total_size, 1000);
}
