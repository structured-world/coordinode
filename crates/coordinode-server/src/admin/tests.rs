use super::decompressing_reader;
use std::io::{Read, Write};

#[test]
fn gzip_input_is_transparently_decompressed() {
    let plain = b"hello\nworld\n";
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(plain).unwrap();
    let gz = enc.finish().unwrap();
    let mut r = decompressing_reader(std::io::Cursor::new(gz)).unwrap();
    let mut out = Vec::new();
    r.read_to_end(&mut out).unwrap();
    assert_eq!(out, plain);
}

#[test]
fn uncompressed_input_passes_through() {
    let plain = b"{\"type\":\"node\"}\n";
    let mut r = decompressing_reader(std::io::Cursor::new(plain.to_vec())).unwrap();
    let mut out = Vec::new();
    r.read_to_end(&mut out).unwrap();
    assert_eq!(out, plain);
}

#[test]
fn zstd_magic_is_rejected_with_guidance() {
    let zstd = vec![0x28u8, 0xb5, 0x2f, 0xfd, 0, 0, 0, 0];
    match decompressing_reader(std::io::Cursor::new(zstd)) {
        Err(e) => assert!(e.to_string().contains("zstd"), "got: {e}"),
        Ok(_) => panic!("expected zstd rejection"),
    }
}
