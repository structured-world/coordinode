use super::*;

/// A standalone executor over a fresh database.
pub(crate) fn standalone() -> (StatementExecutor, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let database = Database::open(dir.path()).expect("open database");
    (StatementExecutor::new(Arc::new(RwLock::new(database))), dir)
}

/// An executor fenced by a real single-node Raft group, wired the way the
/// server wires it: one engine and clock shared by the consensus node and the
/// database, writes proposed through the node.
pub(crate) async fn raft() -> (StatementExecutor, Arc<RaftNode>, tempfile::TempDir) {
    use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, Tier};

    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let config = coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        ),
    ]);
    let engine = Arc::new(
        coordinode_storage::engine::core::StorageEngine::open_with_oracle(
            &config,
            Arc::clone(&oracle),
        )
        .expect("open engine"),
    );
    let node = Arc::new(
        RaftNode::open_with_oracle(1, Arc::clone(&engine), Some(Arc::clone(&oracle)))
            .await
            .expect("raft node"),
    );
    // A single-node group needs a moment to elect itself.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> = Arc::new(
        coordinode_raft::proposal::RaftProposalPipeline::new(Arc::clone(node.raft())),
    );
    let database = Database::from_engine(dir.path(), engine, oracle, pipeline).expect("database");
    let executor =
        StatementExecutor::new(Arc::new(RwLock::new(database))).with_raft_node(Arc::clone(&node));
    (executor, node, dir)
}

/// Every executed statement is counted under what it did: a write that
/// committed, a read, or a failure, which also counts as an error.
#[test]
fn statements_are_counted_by_what_they_did() {
    let (executor, _dir) = standalone();
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(async {
            for query in [
                "CREATE (:T {v: 1})",
                "MATCH (t:T) RETURN t.v",
                "MATCH (t:T) RETURN no_such_function(t.v)",
            ] {
                let Ok(Admission::Run(admitted)) =
                    executor.admit(query, &Requested::default(), false).await
                else {
                    panic!("{query} was not admitted to run here");
                };
                let _ = executor.execute(query, None, None, &admitted);
            }
        });
    });
    let text = handle.render();
    for kind in ["write", "read", "failed"] {
        assert!(
            text.contains(&format!(r#"coordinode_query_total{{type="{kind}"}} 1"#)),
            "{kind}: {text}"
        );
    }
    assert!(text.contains("coordinode_query_errors_total 1"), "{text}");
    assert!(text.contains("coordinode_query_active 0"), "{text}");
}

/// What a statement leaves out comes from the defaults, what it names wins,
/// and the write concern it leaves out stays unset for the database to fill.
#[test]
fn a_setting_left_out_is_the_default_and_a_named_one_wins() {
    let (executor, _dir) = standalone();
    let executor = executor.with_statement_defaults(StatementDefaults {
        read_concern: ExecutorReadConcernLevel::Majority,
        read_preference: ReadPreference::Nearest,
        write_concern: WriteConcern::w1(),
    });

    let unnamed = executor
        .check("RETURN 1", &Requested::default())
        .expect("checked");
    assert_eq!(
        unnamed.read_concern.level,
        ExecutorReadConcernLevel::Majority
    );
    assert_eq!(unnamed.read_preference, ReadPreference::Nearest);
    assert_eq!(unnamed.write_concern, None);

    let named = executor
        .check(
            "RETURN 1",
            &Requested {
                read_concern: Some(ExecutorReadConcernLevel::Linearizable),
                read_preference: Some(ReadPreference::Primary),
                write_concern: Some(WriteConcern::majority()),
                ..Requested::default()
            },
        )
        .expect("checked");
    assert_eq!(
        named.read_concern.level,
        ExecutorReadConcernLevel::Linearizable
    );
    assert_eq!(named.read_preference, ReadPreference::Primary);
    assert_eq!(named.write_concern, Some(WriteConcern::majority()));
}

/// A combination that cannot be served is refused before anything waits or
/// runs: a causal read at LOCAL or LINEARIZABLE, a causal write under a
/// concern that can be lost, a pinned timestamp outside SNAPSHOT.
#[test]
fn an_unservable_combination_is_refused_before_it_runs() {
    let (executor, _dir) = standalone();
    let causal = |level| Requested {
        read_concern: Some(level),
        after_index: 1,
        ..Requested::default()
    };
    for level in [
        ExecutorReadConcernLevel::Local,
        ExecutorReadConcernLevel::Linearizable,
    ] {
        let status = executor
            .check("RETURN 1", &causal(level))
            .expect_err("a causal read needs MAJORITY or SNAPSHOT");
        assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    }

    let lossy_write = Requested {
        write_concern: Some(WriteConcern::w1()),
        ..causal(ExecutorReadConcernLevel::Majority)
    };
    let status = executor
        .check("CREATE (:Causal)", &lossy_write)
        .expect_err("a causal write must be majority-journaled");
    assert!(status.message().contains("writeConcern"), "{status:?}");
    // The same concern on a read is no concern of the causal check.
    executor
        .check("RETURN 1", &lossy_write)
        .expect("a read under a weak write concern");

    let pinned = Requested {
        read_concern: Some(ExecutorReadConcernLevel::Majority),
        at_timestamp: 7,
        ..Requested::default()
    };
    let status = executor
        .check("RETURN 1", &pinned)
        .expect_err("a pinned timestamp needs SNAPSHOT");
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
}

/// On the leader of a real group, a statement is fenced with its preference:
/// a SECONDARY read is refused there, a PRIMARY one is served and reports the
/// applied index and that the leader served it.
#[tokio::test(flavor = "multi_thread")]
async fn the_fence_applies_the_resolved_preference() {
    let (executor, node, _dir) = raft().await;

    let secondary = Requested {
        read_preference: Some(ReadPreference::Secondary),
        ..Requested::default()
    };
    let status = executor
        .admit("RETURN 1", &secondary, false)
        .await
        .expect_err("the leader is not a secondary");
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);

    match executor
        .admit("RETURN 1", &Requested::default(), false)
        .await
        .expect("a primary read on the leader")
    {
        Admission::Run(admitted) => {
            assert!(admitted.served_by_leader);
            assert!(admitted.applied_index > 0, "the group applied its election");
        }
        Admission::Forward(leader) => panic!("the leader forwarded to {leader}"),
    }

    node.shutdown().await.expect("shutdown");
}
