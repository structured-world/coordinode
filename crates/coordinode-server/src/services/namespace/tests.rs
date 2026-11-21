use super::*;

/// A call names its namespace in one metadata entry; one naming none selects
/// none; an empty or unreadable name is refused rather than taken as none.
#[test]
fn a_call_names_its_namespace_in_its_metadata() {
    let mut metadata = MetadataMap::new();
    assert_eq!(from_metadata(&metadata).expect("none"), None);
    metadata.insert(NAMESPACE_HEADER, "tenant-a".parse().expect("value"));
    assert_eq!(
        from_metadata(&metadata).expect("named"),
        Some("tenant-a".to_owned())
    );
    metadata.insert(NAMESPACE_HEADER, "".parse().expect("value"));
    assert_eq!(
        from_metadata(&metadata).expect_err("empty").code(),
        tonic::Code::InvalidArgument
    );
}
