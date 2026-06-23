use super::*;

#[test]
fn chunk_id_from_data() {
    let id = ChunkId::from_data(b"hello world");
    assert_eq!(id.as_bytes().len(), 32);
}

#[test]
fn chunk_id_deterministic() {
    let id1 = ChunkId::from_data(b"same data");
    let id2 = ChunkId::from_data(b"same data");
    assert_eq!(id1, id2);
}

#[test]
fn chunk_id_different_data() {
    let id1 = ChunkId::from_data(b"data a");
    let id2 = ChunkId::from_data(b"data b");
    assert_ne!(id1, id2);
}

#[test]
fn chunk_id_hex_roundtrip() {
    let id = ChunkId::from_data(b"test");
    let hex = id.to_hex();
    assert_eq!(hex.len(), 64);
    let restored = ChunkId::from_hex(&hex).expect("parse hex");
    assert_eq!(id, restored);
}

#[test]
fn chunk_id_from_hex_invalid() {
    assert!(ChunkId::from_hex("short").is_none());
    assert!(ChunkId::from_hex(&"zz".repeat(32)).is_none());
}

#[test]
fn chunk_id_display() {
    let id = ChunkId::from_data(b"test");
    let display = format!("{id}");
    assert_eq!(display.len(), 64);
    assert_eq!(display, id.to_hex());
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
    let chunks = chunk_data(b"", DEFAULT_CHUNK_SIZE);
    assert!(chunks.is_empty());
}

#[test]
fn chunk_data_dedup() {
    // Two identical chunks should have same ChunkId
    let data = vec![0xFFu8; DEFAULT_CHUNK_SIZE * 2];
    let chunks = chunk_data(&data, DEFAULT_CHUNK_SIZE);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].0, chunks[1].0); // same content = same hash
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
    let (blob_ref, chunks) = create_blob(&data);

    assert_eq!(blob_ref.total_size, data.len() as u64);
    assert_eq!(blob_ref.chunk_count(), 3);
    assert_eq!(chunks.len(), 3);

    // Reassemble
    let mut reassembled = Vec::new();
    for (_, chunk_data) in &chunks {
        reassembled.extend_from_slice(chunk_data);
    }
    assert_eq!(reassembled, data);
}

#[test]
fn blob_ref_msgpack_roundtrip() {
    let (blob_ref, _) = create_blob(b"test data for blob");
    let bytes = blob_ref.to_msgpack().expect("serialize");
    let restored = BlobRef::from_msgpack(&bytes).expect("deserialize");
    assert_eq!(blob_ref, restored);
}

#[test]
fn blob_key_encoding() {
    let id = ChunkId::from_data(b"test");
    let key = encode_blob_key(&id);
    assert!(key.starts_with(b"blob:"));
    // 5 + 64 = 69 bytes
    assert_eq!(key.len(), 69);
}

#[test]
fn blobref_key_encoding() {
    let key = encode_blobref_key(NodeId::from_raw(42), 3);
    assert!(key.starts_with(b"blobref:"));
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
    let (blob_ref, chunks) = create_blob_with_chunk_size(&data, 300);
    assert_eq!(chunks.len(), 4); // 300+300+300+100
    assert_eq!(blob_ref.chunk_count(), 4);
    assert_eq!(blob_ref.total_size, 1000);
}
