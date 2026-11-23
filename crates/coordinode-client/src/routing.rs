//! Which member a read goes to. A client permitted several endpoints of one
//! group measures how fast each answers it when it is built; a read that may
//! be served away from the leader goes to the fastest. After an answer that
//! took a hop and named the leader, a later read the leader may serve goes
//! straight to it when it is one of the endpoints this client reaches. A
//! member's own answer naming the leader moves nothing, and a leader the
//! client does not reach leaves it working through the member it uses.

use std::time::{Duration, Instant};

use tonic::metadata::MetadataMap;
use tonic::transport::Channel;

use crate::ReadPreference;
use crate::config::ClientConfig;
use crate::proto::query::{ExecuteCypherRequest, cypher_service_client::CypherServiceClient};

/// How long probing one endpoint may take before it counts as unreachable.
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// The member a response says served it, the hop it took and the leader it
/// names, as the server's response metadata carries them.
const NODE_HEADER: &str = "x-coordinode-node";
const HOPS_HEADER: &str = "x-coordinode-hops";
const LEADER_HEADER: &str = "x-coordinode-leader";

/// One endpoint this client reaches.
#[derive(Debug, Clone)]
pub(crate) struct Route {
    /// Where it is reached.
    pub(crate) endpoint: String,
    /// Statements sent there.
    pub(crate) cypher: CypherServiceClient<Channel>,
    /// How long it took to answer the probe.
    pub(crate) rtt: Duration,
    /// The member it is, as its answer named itself; `None` when it did not.
    pub(crate) node: Option<u64>,
}

/// The endpoints this client reaches, fastest first, and the leader to go to
/// directly once a forwarded answer named it.
#[derive(Debug, Default)]
pub(crate) struct Routes {
    members: Vec<Route>,
    direct: Option<usize>,
}

impl Routes {
    /// Probe the endpoint `config` names and every further one it permits,
    /// concurrently; keep those that answered, fastest first. `None` when no
    /// further endpoint is named: every call then goes to the one endpoint.
    pub(crate) async fn probe(config: &ClientConfig) -> Option<Self> {
        if config.endpoints.is_empty() {
            return None;
        }
        let mut probes = tokio::task::JoinSet::new();
        for endpoint in std::iter::once(config.endpoint.clone()).chain(config.endpoints.clone()) {
            let config = config.clone();
            probes.spawn(async move { probe(endpoint, &config).await });
        }
        let mut members = Vec::new();
        while let Some(probed) = probes.join_next().await {
            if let Ok(Some(route)) = probed {
                members.push(route);
            }
        }
        members.sort_by_key(|route| route.rtt);
        Some(Self {
            members,
            direct: None,
        })
    }

    /// Where a read with `preference` goes, and whether it goes there
    /// directly on a forwarded answer's word: the leader when one is known
    /// and the preference lets it serve; otherwise the fastest endpoint for a
    /// read a follower may serve. `None` for a read only the leader may serve
    /// with no leader known, or when nothing answered the probe: it then goes
    /// to the endpoint the client was built for.
    pub(crate) fn for_read(&self, preference: ReadPreference) -> Option<(Route, bool)> {
        let direct = self
            .direct
            .filter(|_| preference != ReadPreference::Secondary)
            .and_then(|i| self.members.get(i));
        if let Some(route) = direct {
            return Some((route.clone(), true));
        }
        match preference {
            ReadPreference::Primary => None,
            ReadPreference::PrimaryPreferred
            | ReadPreference::Secondary
            | ReadPreference::SecondaryPreferred
            | ReadPreference::Nearest => self.members.first().map(|route| (route.clone(), false)),
        }
    }

    /// Take a response's routing hints: an answer that took a hop and names
    /// a leader this client reaches sends later suitable reads straight to
    /// it. An answer served where it arrived moves nothing, whatever leader
    /// it names.
    pub(crate) fn observe(&mut self, metadata: &MetadataMap) {
        if header(metadata, HOPS_HEADER).as_deref() != Some("1") {
            return;
        }
        let leader = header(metadata, LEADER_HEADER).and_then(|v| v.parse::<u64>().ok());
        let at = leader.and_then(|leader| {
            self.members
                .iter()
                .position(|route| route.node == Some(leader))
        });
        if let Some(at) = at {
            self.direct = Some(at);
        }
    }

    /// The direct route failed to carry a read: later reads go as if no
    /// leader had been named.
    pub(crate) fn forget_direct(&mut self) {
        self.direct = None;
    }

    /// The endpoints that answered the probe, fastest first.
    pub(crate) fn members(&self) -> &[Route] {
        &self.members
    }
}

/// Connect to `endpoint` and time one trivial statement there; `None` when it
/// does not answer within [`PROBE_TIMEOUT`].
async fn probe(endpoint: String, config: &ClientConfig) -> Option<Route> {
    let probed = tokio::time::timeout(PROBE_TIMEOUT, async {
        let channel = Channel::from_shared(endpoint.clone())
            .ok()?
            .connect()
            .await
            .ok()?;
        let mut cypher = CypherServiceClient::new(channel);
        let mut request = tonic::Request::new(ExecuteCypherRequest {
            query: "RETURN 1".to_string(),
            ..Default::default()
        });
        config.name_namespace(request.metadata_mut());
        let started = Instant::now();
        let response = cypher.execute_cypher(request).await.ok()?;
        let rtt = started.elapsed();
        let node = header(response.metadata(), NODE_HEADER).and_then(|v| v.parse().ok());
        Some(Route {
            endpoint,
            cypher,
            rtt,
            node,
        })
    })
    .await;
    probed.ok().flatten()
}

#[cfg(test)]
impl Routes {
    /// Routes over `members` as measured, fastest first, for tests that
    /// control timing without a network.
    pub(crate) fn measured(members: Vec<Route>) -> Self {
        Self {
            members,
            direct: None,
        }
    }
}

fn header(metadata: &MetadataMap, key: &str) -> Option<String> {
    metadata
        .get(key)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
