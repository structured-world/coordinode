use super::*;

#[test]
fn parse_valid_path() {
    let (bucket, key) = parse_s3_path("/mybucket/path/to/file.txt").expect("parse");
    assert_eq!(bucket, "mybucket");
    assert_eq!(key, "path/to/file.txt");
}

#[test]
fn parse_simple_path() {
    let (bucket, key) = parse_s3_path("/blobs/mykey").expect("parse");
    assert_eq!(bucket, "blobs");
    assert_eq!(key, "mykey");
}

#[test]
fn parse_root_path() {
    assert!(parse_s3_path("/").is_none());
}

#[test]
fn parse_bucket_only() {
    assert!(parse_s3_path("/bucket/").is_none());
}

#[test]
fn parse_no_leading_slash() {
    assert!(parse_s3_path("bucket/key").is_none());
}
