//! Replacing a damaged copy of an index on this member alone from a peer's
//! copy, over the segment transfer.

use std::sync::Arc;

use coordinode_raft::cluster::{PeerTarget, RaftNode};
use coordinode_replicate::{PeerLinks, PieceEncoding, SegmentInstaller};

/// How this member reaches its peers' segment services, and the members it
/// has now, each with the identity it must prove; `None` for a member without
/// peers. Read at each use, so a membership change reaches the next pull.
pub fn segment_peers(node: &RaftNode) -> Option<(PeerLinks, Vec<PeerTarget>)> {
    let links = PeerLinks {
        connections: Arc::clone(node.connections()?),
        admission: Arc::clone(node.admission()?) as _,
    };
    let peers = node.peer_targets();
    (!peers.is_empty()).then_some((links, peers))
}

/// Fills a replacement of one of this member's index copies from its peers'
/// copies over the segment transfer.
pub(super) struct PeerIndexCopy {
    pub(super) installer: Arc<SegmentInstaller>,
    pub(super) node: Arc<RaftNode>,
    pub(super) runtime: tokio::runtime::Handle,
}

impl coordinode_query::index::IndexCopySource for PeerIndexCopy {
    fn prepare(
        &self,
        generation: coordinode_query::index::GenerationId,
    ) -> Result<Option<Vec<u8>>, String> {
        let (links, peers) = segment_peers(&self.node)
            .ok_or("no peer holds a copy to replace this member's from")?;
        // Called from an index-check thread, outside the runtime.
        self.runtime
            .block_on(self.installer.prepare_generation(
                &links,
                &peers,
                generation,
                1 << 20,
                PieceEncoding::None,
            ))
            .map(|(_, proof)| proof)
            .map_err(|e| e.to_string())
    }
}
