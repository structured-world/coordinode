//! Round-trip test for the raft-snapshot backup format.
//!
//! A full Raft snapshot deliberately excludes the `meta:` Schema keys, which
//! include the field interner. A standalone backup therefore frames the
//! interner alongside the snapshot and restores it separately. These tests run
//! the `coordinode` binary, and check that property names resolve in a
//! freshly restored database, i.e. the interner survived the round trip.
#![allow(clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

/// Run `coordinode <args>` and require it to succeed.
fn coordinode(args: &[&std::ffi::OsStr]) {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_coordinode"))
        .args(args)
        .output()
        .expect("run coordinode");
    assert!(
        output.status.success(),
        "coordinode {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Run `coordinode restore` of the raft-snapshot `input` into `data`.
fn restore(data: &std::path::Path, input: &std::path::Path) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_coordinode"))
        .arg("restore")
        .arg("--data")
        .arg(data)
        .arg("--input")
        .arg(input)
        .arg("--format")
        .arg("raft-snapshot")
        .output()
        .expect("run coordinode restore")
}

/// A raft-snapshot backup of `src` (two users, a bound domain) at `out`.
fn backup_of_users(src: &std::path::Path, out: &std::path::Path) {
    {
        let mut db = Database::open(src).expect("open src");
        db.execute_cypher("CREATE (:User {name: 'Alice'}), (:User {name: 'Bob'})")
            .expect("create users");
    }
    coordinode(&[
        "backup".as_ref(),
        "--data".as_ref(),
        src.as_os_str(),
        "--output".as_ref(),
        out.as_os_str(),
        "--format".as_ref(),
        "raft-snapshot".as_ref(),
    ]);
}

/// Every file under `dir` with its bytes: what "left as it was" compares.
fn tree(dir: &std::path::Path) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
    let mut files = std::collections::BTreeMap::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in std::fs::read_dir(&next).expect("list") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.insert(path.clone(), std::fs::read(&path).expect("read"));
            }
        }
    }
    files
}

/// A restore into a directory that already holds a store is refused before
/// anything is opened or written: every byte of the target stays as it was,
/// and its data is still served.
#[test]
fn a_restore_into_a_store_leaves_it_byte_for_byte() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let (src_dir, dst_dir) = (dir.path().join("src"), dir.path().join("dst"));
    let backup = dir.path().join("db.snap");
    backup_of_users(&src_dir, &backup);
    {
        let mut dst = Database::open(&dst_dir).expect("open dst");
        dst.execute_cypher("CREATE (:Mine {k: 1})")
            .expect("own data");
    }
    let before = tree(&dst_dir);

    let output = restore(&dst_dir, &backup);
    assert!(!output.status.success(), "a store was restored over");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not empty"),
        "the refusal says why: {stderr}"
    );
    assert_eq!(tree(&dst_dir), before, "the target is untouched");
    let mut dst = Database::open(&dst_dir).expect("reopen dst");
    assert_eq!(
        dst.execute_cypher("MATCH (n:Mine) RETURN n")
            .expect("own")
            .len(),
        1
    );
    assert!(
        dst.execute_cypher("MATCH (n:User) RETURN n")
            .expect("users")
            .is_empty()
    );
}

/// Input that cannot be installed whole is refused before the target is
/// created: a frame naming another deployment than its image, a truncated
/// file, a corrupted byte, a mode this release does not write.
#[test]
fn an_invalid_backup_is_refused_before_the_target_exists() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let src_dir = dir.path().join("src");
    let backup = dir.path().join("db.snap");
    backup_of_users(&src_dir, &backup);
    let bytes = std::fs::read(&backup).expect("read backup");
    let interner_len = u32::from_be_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
    // [mode][len][interner][has_deployment][deployment]...
    let deployment_at = 5 + interner_len + 1;

    // A frame naming a deployment the image does not belong to.
    let mut foreign = bytes.clone();
    foreign[deployment_at - 1] = 1;
    foreign[deployment_at] ^= 0xFF;
    let mut corrupt = bytes.clone();
    let middle = deployment_at + 16 + (bytes.len() - deployment_at - 16) / 2;
    corrupt[middle] ^= 0xFF;
    let mut unsupported = bytes.clone();
    unsupported[0] = 9;
    let truncated = bytes[..bytes.len() - 64].to_vec();

    for (name, input, reason) in [
        ("foreign", foreign, "inconsistent"),
        ("corrupt", corrupt, "not valid"),
        ("truncated", truncated, "not valid"),
        ("unsupported", unsupported, "unknown snapshot mode"),
    ] {
        let path = dir.path().join(format!("{name}.snap"));
        std::fs::write(&path, input).expect("write input");
        let dst_dir = dir.path().join(format!("dst-{name}"));
        let output = restore(&dst_dir, &path);
        assert!(!output.status.success(), "{name}: restored");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(reason), "{name}: {stderr}");
        assert!(!dst_dir.exists(), "{name}: the target was created");
    }
}

/// A restore cut short at any step leaves a store nothing serves: an
/// ordinary open is refused, and so is the restore of another image into
/// it, which leaves it as it was. Running the same restore again finishes
/// it, and the store then holds exactly the image.
#[test]
fn an_interrupted_restore_is_finished_only_by_the_same_image() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let backup = dir.path().join("users.snap");
    backup_of_users(&dir.path().join("src"), &backup);
    let other = dir.path().join("other.snap");
    {
        let other_src = dir.path().join("other-src");
        let mut db = Database::open(&other_src).expect("open other");
        db.execute_cypher("CREATE (:Other)").expect("other data");
        drop(db);
        coordinode(&[
            "backup".as_ref(),
            "--data".as_ref(),
            other_src.as_os_str(),
            "--output".as_ref(),
            other.as_os_str(),
            "--format".as_ref(),
            "raft-snapshot".as_ref(),
        ]);
    }

    for step in ["claimed", "installed", "persisted"] {
        let dst_dir = dir.path().join(format!("dst-{step}"));
        let cut = std::process::Command::new(env!("CARGO_BIN_EXE_coordinode"))
            .env("COORDINODE_TEST_RESTORE_ABORT", step)
            .arg("restore")
            .arg("--data")
            .arg(&dst_dir)
            .arg("--input")
            .arg(&backup)
            .arg("--format")
            .arg("raft-snapshot")
            .output()
            .expect("run coordinode restore");
        assert!(
            !cut.status.success(),
            "{step}: the restore was not cut short"
        );

        let refused = Database::open(&dst_dir)
            .err()
            .unwrap_or_else(|| panic!("{step}: a partial restore opened"));
        assert!(
            refused.to_string().contains("unfinished restore"),
            "{step}: {refused}"
        );

        let before = tree(&dst_dir);
        let output = restore(&dst_dir, &other);
        assert!(!output.status.success(), "{step}: another image restored");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("unfinished restore of image"),
            "{step}: {stderr}"
        );
        assert_eq!(tree(&dst_dir), before, "{step}: the target was changed");

        let output = restore(&dst_dir, &backup);
        assert!(
            output.status.success(),
            "{step}: the same image did not finish: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let mut dst = Database::open(&dst_dir).expect("open the restored store");
        assert_eq!(
            dst.execute_cypher("MATCH (n:User) RETURN n")
                .expect("users")
                .len(),
            2,
            "{step}: exactly the image's users"
        );
        assert!(
            dst.execute_cypher("MATCH (n:Other) RETURN n")
                .expect("other")
                .is_empty(),
            "{step}"
        );
    }
}

/// A config restoring into `data` under a `limit_mb` memory limit.
fn limited_config(
    dir: &std::path::Path,
    data: &std::path::Path,
    limit_mb: u64,
) -> std::path::PathBuf {
    let path = dir.join(format!("restore-{limit_mb}.yaml"));
    std::fs::write(
        &path,
        format!(
            "data_dir: \"{}\"\nquery_memory_limit_mb: {limit_mb}\n",
            data.display()
        ),
    )
    .expect("write config");
    path
}

/// Run `coordinode restore` of the raft-snapshot `input` with `config`.
fn restore_with(config: &std::path::Path, input: &std::path::Path) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_coordinode"))
        .arg("restore")
        .arg("--config")
        .arg(config)
        .arg("--input")
        .arg(input)
        .arg("--format")
        .arg("raft-snapshot")
        .output()
        .expect("run coordinode restore")
}

/// The peak memory a restore reports holding at once, from its log.
fn reported_peak(output: &std::process::Output) -> Option<u64> {
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let at = text.find("memory_peak_bytes")? + "memory_peak_bytes".len();
    let digits: String = text[at..]
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// An image many times the memory limit restores whole, holding no more
/// than the limit at once: it streams into the target rather than being
/// read into memory first.
#[test]
fn a_large_image_restores_within_the_memory_limit() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let (src_dir, dst_dir) = (dir.path().join("src"), dir.path().join("dst"));
    let backup = dir.path().join("big.snap");
    {
        let mut src = Database::open(&src_dir).expect("open src");
        let pad = "x".repeat(1000);
        for batch in 0..20 {
            src.execute_cypher(&format!(
                "UNWIND range(1, 1000) AS i CREATE (:Big {{i: {batch} * 1000 + i, pad: '{pad}'}})"
            ))
            .expect("create");
        }
    }
    coordinode(&[
        "backup".as_ref(),
        "--data".as_ref(),
        src_dir.as_os_str(),
        "--output".as_ref(),
        backup.as_os_str(),
        "--format".as_ref(),
        "raft-snapshot".as_ref(),
    ]);
    let image = std::fs::metadata(&backup).expect("backup size").len();
    let limit = 2u64 << 20;
    assert!(
        image > 4 * limit,
        "the image ({image} bytes) dwarfs the limit"
    );

    let output = restore_with(&limited_config(dir.path(), &dst_dir, 2), &backup);
    assert!(
        output.status.success(),
        "restore failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let peak = reported_peak(&output).expect("the restore reports its memory peak");
    assert!(
        peak <= limit,
        "held {peak} bytes at once under a {limit}-byte limit"
    );
    let mut dst = Database::open(&dst_dir).expect("open dst");
    assert_eq!(
        dst.execute_cypher("MATCH (n:Big) RETURN count(n) AS c")
            .expect("count")[0]
            .get("c"),
        Some(&Value::Int(20_000))
    );
}

/// A record larger than the memory limit, or a length claiming one, is
/// refused before it is read into memory, and the target is not created.
#[test]
fn a_record_past_the_memory_limit_is_refused_before_it_is_read() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let (src_dir, backup) = (dir.path().join("src"), dir.path().join("wide.snap"));
    {
        let mut src = Database::open(&src_dir).expect("open src");
        let wide = "w".repeat(3 << 20);
        src.execute_cypher(&format!("CREATE (:Wide {{v: '{wide}'}})"))
            .expect("create");
    }
    coordinode(&[
        "backup".as_ref(),
        "--data".as_ref(),
        src_dir.as_os_str(),
        "--output".as_ref(),
        backup.as_os_str(),
        "--format".as_ref(),
        "raft-snapshot".as_ref(),
    ]);
    let wide_dst = dir.path().join("dst-wide");
    let output = restore_with(&limited_config(dir.path(), &wide_dst, 1), &backup);
    assert!(!output.status.success(), "a record past the limit restored");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("memory limit"), "{stderr}");
    assert!(!wide_dst.exists(), "the target was created");

    // A frame claiming a dictionary of 4 GiB.
    let mut claimed = std::fs::read(&backup).expect("read backup");
    claimed[1..5].copy_from_slice(&0xFFFF_FFF0u32.to_be_bytes());
    let claimed_path = dir.path().join("claimed.snap");
    std::fs::write(&claimed_path, claimed).expect("write");
    let claimed_dst = dir.path().join("dst-claimed");
    let output = restore_with(&limited_config(dir.path(), &claimed_dst, 1), &claimed_path);
    assert!(
        !output.status.success(),
        "a 4 GiB dictionary length restored"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("memory limit"), "{stderr}");
    assert!(!claimed_dst.exists(), "the target was created");
}

/// The commit timestamp of a write of an unrelated node.
fn mark(db: &mut Database) -> u64 {
    let tx = db.begin_transaction().expect("begin");
    db.execute_in_transaction(tx, "CREATE (:Mark)", None)
        .expect("mark");
    db.commit_transaction(tx)
        .expect("commit")
        .commit_ts
        .as_raw()
}

/// Run `alter`, wait for its parts to be folded, and return a system time
/// at which it is in force.
fn settled(db: &mut Database, alter: &str) -> u64 {
    use coordinode_modality::IndexStore as _;
    db.execute_cypher(alter).expect("alter");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !coordinode_modality::LocalIndexStore::new(db.engine())
        .list_definitions()
        .expect("definitions")
        .iter()
        .all(|def| def.selection.is_empty())
    {
        assert!(std::time::Instant::now() < deadline, "{alter}: unfolded");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    mark(db)
}

/// The names of the things at `as_of`, in order.
fn things(db: &mut Database, as_of: u64) -> Vec<String> {
    db.execute_cypher(&format!(
        "MATCH (n:Thing) RETURN n.name AS name ORDER BY name AS OF TIMESTAMP {as_of}"
    ))
    .unwrap_or_else(|e| panic!("as of {as_of}: {e}"))
    .iter()
    .map(|row| match row.get("name") {
        Some(Value::String(name)) => name.clone(),
        other => panic!("name {other:?}"),
    })
    .collect()
}

/// The same history restored into a fresh store keeps its domain and every
/// interpretation it retained: three interpretations of an implied label
/// over unchanged nodes, backed up, restored and read after the restored
/// store's row versions of those times are collected, answer exactly as the
/// source did, under the source's deployment.
#[test]
fn a_restored_history_keeps_its_domain_and_interpretations() {
    use coordinode_storage::engine::partition::Partition;
    let dir = tempfile::tempdir().expect("tmpdir");
    let (src_dir, dst_dir) = (dir.path().join("src"), dir.path().join("dst"));
    let backup = dir.path().join("db.snap");
    let (times, domain) = {
        let mut src = Database::open(&src_dir).expect("open src");
        for statement in [
            "CREATE NODE TYPE Thing TEMPORAL WITH (name: STRING, valid_from: INT)",
            "CREATE NODE TYPE X TEMPORAL WITH (name: STRING, valid_from: INT)",
            "CREATE NODE TYPE Y TEMPORAL WITH (name: STRING, valid_from: INT)",
            "CREATE (:X {name: 'x', valid_from: 0}), (:Y {name: 'y', valid_from: 0})",
        ] {
            src.execute_cypher(statement).expect(statement);
        }
        let before = mark(&mut src);
        let x_only = settled(&mut src, "ALTER NODE TYPE X SET IMPLIES (Thing)");
        let both = settled(&mut src, "ALTER NODE TYPE Y SET IMPLIES (Thing)");
        let y_only = settled(&mut src, "ALTER NODE TYPE X SET IMPLIES ()");
        let domain = coordinode_storage::engine::metadata::bound_deployment(src.engine())
            .expect("binding")
            .expect("the history founded a domain");
        ([before, x_only, both, y_only], domain)
    };
    let format = std::ffi::OsStr::new("raft-snapshot");
    coordinode(&[
        "backup".as_ref(),
        "--data".as_ref(),
        src_dir.as_os_str(),
        "--output".as_ref(),
        backup.as_os_str(),
        "--format".as_ref(),
        format,
    ]);
    coordinode(&[
        "restore".as_ref(),
        "--data".as_ref(),
        dst_dir.as_os_str(),
        "--input".as_ref(),
        backup.as_os_str(),
        "--format".as_ref(),
        format,
    ]);
    let mut dst = Database::open(&dst_dir).expect("open dst");
    assert_eq!(
        coordinode_storage::engine::metadata::bound_deployment(dst.engine()).expect("binding"),
        Some(domain),
        "the restored store continues the source's domain"
    );
    dst.set_retention_window(std::time::Duration::from_secs(2));
    let oracle = dst.engine().oracle().expect("oracle");
    oracle.advance_to(coordinode_core::txn::timestamp::Timestamp::from_raw(
        oracle.current().as_raw() + 10_000_000,
    ));
    mark(&mut dst);
    for part in [
        Partition::Node,
        Partition::Schema,
        Partition::Idx,
        Partition::NodeHistory,
    ] {
        dst.engine().force_compaction(part).expect("compact");
    }
    assert!(dst.oldest_readable_timestamp().as_raw() > times[3]);
    let expected: [&[&str]; 4] = [&[], &["x"], &["x", "y"], &["y"]];
    for (as_of, names) in times.into_iter().zip(expected) {
        assert_eq!(things(&mut dst, as_of), names, "as of {as_of}");
    }
}

#[test]
fn raft_snapshot_round_trips_data_and_interner() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let (src_dir, dst_dir) = (dir.path().join("src"), dir.path().join("dst"));
    let backup = dir.path().join("db.snap");
    {
        let mut src = Database::open(&src_dir).expect("open src");
        src.execute_cypher("CREATE (a:User {name: 'Alice', age: 30})")
            .expect("create alice");
        src.execute_cypher("CREATE (b:User {name: 'Bob', age: 25})")
            .expect("create bob");
    }

    let format = std::ffi::OsStr::new("raft-snapshot");
    coordinode(&[
        "backup".as_ref(),
        "--data".as_ref(),
        src_dir.as_os_str(),
        "--output".as_ref(),
        backup.as_os_str(),
        "--format".as_ref(),
        format,
    ]);
    coordinode(&[
        "restore".as_ref(),
        "--data".as_ref(),
        dst_dir.as_os_str(),
        "--input".as_ref(),
        backup.as_os_str(),
        "--format".as_ref(),
        format,
    ]);
    let mut dst = Database::open(&dst_dir).expect("open dst");

    // Property names must resolve: an inline `{name: 'Alice'}` filter only
    // matches if the interner mapped "name" -> field id (proves the interner
    // was carried, not lost with the snapshot's meta exclusion).
    let alice = dst
        .execute_cypher("MATCH (n:User {name: 'Alice'}) RETURN n.age")
        .expect("query alice");
    assert_eq!(alice.len(), 1, "Alice resolves by name after restore");
    assert_eq!(
        alice[0].get("n.age"),
        Some(&Value::Int(30)),
        "Alice's age property survived the round trip"
    );

    let all = dst
        .execute_cypher("MATCH (n:User) RETURN n.name")
        .expect("query all users");
    assert_eq!(all.len(), 2, "both users restored");
}

/// A server node's store keeps its group's history; the offline backup reads
/// it as the node itself opens it, writing nothing outside the group's log,
/// and the image restores into a fresh directory.
#[test]
fn a_member_store_is_backed_up_offline() {
    use coordinode_embed::replication::{MemberOptions, ReplicatedDatabase};
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };

    let dir = tempfile::tempdir().expect("tmpdir");
    let (src_dir, dst_dir) = (dir.path().join("src"), dir.path().join("dst"));
    let backup = dir.path().join("db.snap");
    let storage = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        &src_dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(async {
            let member = ReplicatedDatabase::open(&storage, MemberOptions::new(1), None)
                .await
                .expect("open the member");
            member
                .database()
                .write()
                .execute_cypher("CREATE (:User {name: 'Alice', age: 30})")
                .expect("create alice");
            // A capture holds what the group's log closed; the leader closes
            // an idle write shortly after it lands.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while member.database().read().engine().applied_past_closure() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "the write was never closed"
                );
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            member.shutdown().await.expect("shut down");
        });

    let format = std::ffi::OsStr::new("raft-snapshot");
    coordinode(&[
        "backup".as_ref(),
        "--data".as_ref(),
        src_dir.as_os_str(),
        "--output".as_ref(),
        backup.as_os_str(),
        "--format".as_ref(),
        format,
    ]);
    coordinode(&[
        "restore".as_ref(),
        "--data".as_ref(),
        dst_dir.as_os_str(),
        "--input".as_ref(),
        backup.as_os_str(),
        "--format".as_ref(),
        format,
    ]);
    let mut dst = Database::open(&dst_dir).expect("open dst");
    let alice = dst
        .execute_cypher("MATCH (n:User {name: 'Alice'}) RETURN n.age")
        .expect("query alice");
    assert_eq!(alice.len(), 1, "the member's data is in the image");
    assert_eq!(alice[0].get("n.age"), Some(&Value::Int(30)));
    // The member's own consensus state is its log's, not the data's: none
    // of it is installed as progress of the restored store.
    use coordinode_storage::engine::partition::Partition;
    assert!(
        dst.engine()
            .prefix_scan(Partition::Raft, &[])
            .expect("raft partition")
            .next()
            .is_none()
    );
    assert!(
        dst.engine()
            .prefix_scan(Partition::Schema, b"raft:")
            .expect("schema")
            .next()
            .is_none()
    );
}

/// An offline restore into a group member's store is refused: the directory
/// holds a store, and a restore installs only into empty ones. The store is
/// left as it was.
#[test]
fn an_offline_restore_into_a_member_store_is_refused() {
    use coordinode_embed::replication::{MemberOptions, ReplicatedDatabase};
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };

    let dir = tempfile::tempdir().expect("tmpdir");
    let (src_dir, member_dir) = (dir.path().join("src"), dir.path().join("member"));
    let backup = dir.path().join("db.snap");
    {
        let mut src = Database::open(&src_dir).expect("open src");
        src.execute_cypher("CREATE (:User {name: 'Bob'})")
            .expect("create bob");
    }
    let storage = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        &member_dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(async {
        let member = ReplicatedDatabase::open(&storage, MemberOptions::new(1), None)
            .await
            .expect("open the member");
        member
            .database()
            .write()
            .execute_cypher("CREATE (:User {name: 'Alice'})")
            .expect("create alice");
        member.shutdown().await.expect("shut down");
    });

    let format = std::ffi::OsStr::new("raft-snapshot");
    coordinode(&[
        "backup".as_ref(),
        "--data".as_ref(),
        src_dir.as_os_str(),
        "--output".as_ref(),
        backup.as_os_str(),
        "--format".as_ref(),
        format,
    ]);
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_coordinode"))
        .arg("restore")
        .arg("--data")
        .arg(&member_dir)
        .arg("--input")
        .arg(&backup)
        .arg("--format")
        .arg(format)
        .arg("--force")
        .output()
        .expect("run coordinode restore");
    assert!(
        !output.status.success(),
        "a member's store was restored into"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not empty"),
        "the refusal says why, got: {stderr}"
    );

    runtime.block_on(async {
        let member = ReplicatedDatabase::open(&storage, MemberOptions::new(1), None)
            .await
            .expect("reopen the member");
        let names = member
            .database()
            .write()
            .execute_cypher("MATCH (n:User) RETURN n.name")
            .expect("read");
        assert_eq!(names.len(), 1, "the member holds only its own history");
        assert_eq!(names[0].get("n.name"), Some(&Value::String("Alice".into())));
        member.shutdown().await.expect("shut down");
    });
}

/// A restored image carries its operation outcomes and admits nothing new
/// under a descriptor issued before the restore: a committed write resolves
/// as committed and its retry runs nothing, an unused number is refused, the
/// old incarnation cannot be resumed, and enrollment opens a fresh one above
/// every incarnation the image held.
#[test]
fn a_restored_snapshot_keeps_outcomes_and_fences_old_descriptors() {
    use coordinode_embed::db::StatementOptions;
    use coordinode_embed::operation::{IssuerId, OperationOutcome, Resolve};

    const ISSUER: IssuerId = IssuerId([0x31; 16]);
    let options = |op| StatementOptions {
        operation: Some(op),
        ..StatementOptions::default()
    };
    let dir = tempfile::tempdir().expect("tmpdir");
    let (src_dir, dst_dir) = (dir.path().join("src"), dir.path().join("dst"));
    let backup = dir.path().join("db.snap");
    let (done, unused) = {
        let src = Database::open(&src_dir).expect("open src");
        let issuer = src.enroll_issuer(ISSUER).expect("enroll");
        let done = issuer.next().expect("number");
        let unused = issuer.next().expect("number");
        src.execute_cypher_shared_with("CREATE (:Paid {k: 1})", None, None, &options(done))
            .expect("bound write");
        (done, unused)
    };
    let format = std::ffi::OsStr::new("raft-snapshot");
    coordinode(&[
        "backup".as_ref(),
        "--data".as_ref(),
        src_dir.as_os_str(),
        "--output".as_ref(),
        backup.as_os_str(),
        "--format".as_ref(),
        format,
    ]);
    coordinode(&[
        "restore".as_ref(),
        "--data".as_ref(),
        dst_dir.as_os_str(),
        "--input".as_ref(),
        backup.as_os_str(),
        "--format".as_ref(),
        format,
    ]);
    let mut dst = Database::open(&dst_dir).expect("open dst");

    assert!(matches!(
        dst.resolve_operation(&done, Resolve::Lookup)
            .expect("lookup"),
        OperationOutcome::Committed { .. }
    ));
    assert!(
        dst.execute_cypher_shared_with("CREATE (:Paid {k: 1})", None, None, &options(done))
            .is_err(),
        "a retry of the committed write runs nothing"
    );
    assert!(
        dst.execute_cypher_shared_with("CREATE (:Paid {k: 2})", None, None, &options(unused))
            .is_err(),
        "an unused number of the old domain is refused"
    );
    assert_eq!(
        dst.resolve_operation(&unused, Resolve::Lookup)
            .expect("lookup"),
        OperationOutcome::Aborted
    );
    assert_eq!(
        dst.execute_cypher("MATCH (n:Paid) RETURN n")
            .expect("count")
            .len(),
        1
    );
    assert!(
        dst.resume_issuer(ISSUER, done.op.incarnation, unused.op.sequence)
            .is_err(),
        "the old incarnation is closed"
    );
    let fresh = dst.enroll_issuer(ISSUER).expect("enroll after restore");
    assert_eq!(
        fresh.incarnation(),
        2,
        "above every incarnation the image held"
    );
    let next = fresh.next().expect("number");
    dst.execute_cypher_shared_with("CREATE (:Paid {k: 3})", None, None, &options(next))
        .expect("a fresh incarnation writes");
}

/// 0.6 could also write a delta of the changes after a seqno. A delta cannot
/// stand alone, and a seqno does not bound what changed after it (an entry
/// can apply later at a lower commit timestamp), so restore refuses one and
/// says why rather than installing part of a database.
#[test]
fn an_incremental_backup_is_refused_by_restore() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let input = dir.path().join("delta.snap");
    // [mode 1][interner length 0][no interner][no snapshot]
    std::fs::write(&input, [1u8, 0, 0, 0, 0]).expect("write delta");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_coordinode"))
        .arg("restore")
        .arg("--data")
        .arg(dir.path().join("store"))
        .arg("--input")
        .arg(&input)
        .arg("--format")
        .arg("raft-snapshot")
        .output()
        .expect("run coordinode restore");
    assert!(!output.status.success(), "a delta must not restore");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("incremental raft-snapshot backup"),
        "the refusal names the reason, got: {stderr}"
    );
}
