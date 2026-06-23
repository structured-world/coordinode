use super::*;

#[test]
fn record_and_serialise_round_trips() {
    // Use a synthetic report (don't go through BenchReport::new
    // because it touches git which may not be available in
    // sandbox tests).
    let report = BenchReport {
        schema_version: SCHEMA_VERSION,
        timestamp: Utc::now(),
        git: GitMetadata {
            sha: "abc123def4567890".into(),
            sha_short: "abc123d".into(),
            branch: "main".into(),
            dirty: false,
            commit_date: Utc::now(),
        },
        hardware: HardwareFingerprint {
            cpu_brand: "Test CPU".into(),
            cpu_cores: 8,
            cpu_threads: 16,
            ram_gb: 32,
            os_name: "test-os".into(),
            os_version: "1.0".into(),
            arch: "x86_64".into(),
        },
        modality: "vector".into(),
        benchmark: "test".into(),
        dataset: "synth".into(),
        subject: "coordinode".into(),
        codec: "none".into(),
        version: "0.0.0".into(),
        metrics: serde_json::Map::new(),
        notes: None,
    };
    let json = serde_json::to_string(&report).expect("serialise");
    let back: BenchReport = serde_json::from_str(&json).expect("deserialise");
    assert_eq!(back.schema_version, SCHEMA_VERSION);
    assert_eq!(back.modality, "vector");
}

#[test]
fn record_metric_inserts_into_map() {
    let mut report = BenchReport {
        schema_version: SCHEMA_VERSION,
        timestamp: Utc::now(),
        git: GitMetadata {
            sha: "abc".into(),
            sha_short: "abc".into(),
            branch: "main".into(),
            dirty: false,
            commit_date: Utc::now(),
        },
        hardware: HardwareFingerprint {
            cpu_brand: String::new(),
            cpu_cores: 0,
            cpu_threads: 0,
            ram_gb: 0,
            os_name: String::new(),
            os_version: String::new(),
            arch: String::new(),
        },
        modality: "vector".into(),
        benchmark: "test".into(),
        dataset: "synth".into(),
        subject: "coordinode".into(),
        codec: "none".into(),
        version: "0.0.0".into(),
        metrics: serde_json::Map::new(),
        notes: None,
    };
    report.record("recall_at_10", 0.95_f64).expect("record");
    report.record("qps", 1234.5_f64).expect("record");
    assert_eq!(report.metrics.len(), 2);
    assert_eq!(report.metrics["recall_at_10"].as_f64(), Some(0.95));
    assert_eq!(report.metrics["qps"].as_f64(), Some(1234.5));
}

#[test]
fn write_json_creates_directories_and_file() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let mut report = BenchReport::new(
        "vector",
        "test-bench",
        "synth-ds",
        "coordinode",
        "none",
        "0.0.0",
    )
    .expect("build");
    report.record("metric_a", 1.0_f64).expect("record");
    let path = report.write_json(tmp.path(), None).expect("write");
    assert!(path.exists(), "JSON file must exist");
    // Round-trip verify file
    let bytes = std::fs::read(&path).expect("read");
    let back: BenchReport = serde_json::from_slice(&bytes).expect("decode");
    assert_eq!(back.metrics["metric_a"].as_f64(), Some(1.0));
}

#[test]
fn write_json_tag_appears_in_filename() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let report = BenchReport::new(
        "vector",
        "ann-benchmarks",
        "sift-128-euclidean",
        "coordinode",
        "none",
        "0.0.0",
    )
    .expect("build");
    let with_tag = report.write_json(tmp.path(), Some("M24")).expect("write");
    let bare = report.write_json(tmp.path(), None).expect("write");
    let with_name = with_tag.file_name().unwrap().to_string_lossy().to_string();
    let bare_name = bare.file_name().unwrap().to_string_lossy().to_string();
    assert!(
        with_name.contains("-coordinode-M24-"),
        "tag must appear between subject and timestamp: {with_name}"
    );
    assert!(
        !bare_name.contains("-M24-"),
        "untagged file must not carry M segment: {bare_name}"
    );
}
