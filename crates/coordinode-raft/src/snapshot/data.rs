//! What a Raft snapshot carries between openraft, the network and the store.

use std::io::{self, Read};

use coordinode_storage::engine::store_transfer::{ReceivedStore, StoreTransfer};

use crate::storage::EngineWorkGuard;

/// A Raft snapshot's data.
///
/// The log is compacted against the applied store, so a build carries only
/// its metadata; the data is the store itself, pinned when a follower needs
/// it sent and rebuilt beside the follower's store as it arrives.
pub enum SnapshotData {
    /// Metadata only: what a build hands openraft, which keeps nothing of a
    /// build but its metadata.
    Empty,
    /// The store pinned at the snapshot's position, to send. Counted as work
    /// holding the engine until it drops, so a shutdown waits for the pins
    /// to go.
    Pinned {
        /// The pinned trees as a stream; dropped before `_work`.
        transfer: StoreTransfer,
        _work: EngineWorkGuard,
    },
    /// A store received from the leader, to install.
    Received(ReceivedStore),
}

impl std::fmt::Debug for SnapshotData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => f.write_str("Empty"),
            Self::Pinned { transfer, .. } => f.debug_tuple("Pinned").field(transfer).finish(),
            Self::Received(received) => f.debug_tuple("Received").field(received).finish(),
        }
    }
}

/// The bytes of a snapshot arriving as chunks on an async stream, read by a
/// blocking receiver: each read waits for the next chunk. The stream ends
/// when the sending side drops its end.
pub(crate) struct ChunkReader {
    chunks: tokio::sync::mpsc::Receiver<Vec<u8>>,
    current: Vec<u8>,
    at: usize,
}

impl ChunkReader {
    pub(crate) fn new(chunks: tokio::sync::mpsc::Receiver<Vec<u8>>) -> Self {
        Self {
            chunks,
            current: Vec::new(),
            at: 0,
        }
    }
}

impl Read for ChunkReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        while self.at == self.current.len() {
            match self.chunks.blocking_recv() {
                Some(chunk) => {
                    self.current = chunk;
                    self.at = 0;
                }
                None => return Ok(0),
            }
        }
        let n = buf.len().min(self.current.len() - self.at);
        buf[..n].copy_from_slice(&self.current[self.at..self.at + n]);
        self.at += n;
        Ok(n)
    }
}
