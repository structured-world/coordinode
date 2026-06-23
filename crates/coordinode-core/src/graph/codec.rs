//! StreamVByte UID compression for posting lists (V5 disk format).
//!
//! UIDs are stored in blocks of up to 256 entries. Each block has:
//! - `base`: the first UID (absolute u64)
//! - `deltas`: StreamVByte-encoded u32 deltas from previous UID
//! - `num_uids`: count of UIDs in the block
//!
//! **Wire format (V5):** deltas are encoded with `streamvbyte64::Coder1234`.
//! Each block stores `[tag_bytes | data_bytes]` in `deltas`, where
//! `tag_len = (num_uids - 1 + 3) / 4`. The decoder recomputes `tag_len`
//! from `num_uids`.
//!
//! A new block is forced when the 32 MSBs of consecutive UIDs differ
//! (because deltas must fit in u32).
//!
//! **Migration note:** V4 data used LEB128 encoding in `deltas`. Any V4
//! data must be re-encoded via a migration tool before reading with this
//! decoder. See ROADMAP R098 for the V5 migration plan.
//!
//! Inspired by Dgraph's `codec/codec.go` Encoder/Decoder pattern.

use serde::{Deserialize, Serialize};

/// Default number of UIDs per block.
pub const DEFAULT_BLOCK_SIZE: usize = 256;

/// Mask for the upper 32 bits of a u64.
const MSB_MASK: u64 = 0xFFFF_FFFF_0000_0000;

/// Check if two UIDs share the same upper 32 bits.
///
/// When the upper 32 bits differ, the delta between UIDs exceeds u32::MAX,
/// so a new block must be started.
fn match_32msb(a: u64, b: u64) -> bool {
    (a & MSB_MASK) == (b & MSB_MASK)
}

/// A compressed block of UIDs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UidBlock {
    /// First UID in the block (absolute).
    pub base: u64,
    /// Varint-encoded u32 deltas for subsequent UIDs.
    pub deltas: Vec<u8>,
    /// Total number of UIDs in this block (including base).
    pub num_uids: u32,
}

/// A pack of compressed UID blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UidPack {
    /// Block size used during encoding.
    pub block_size: u32,
    /// Compressed blocks.
    pub blocks: Vec<UidBlock>,
}

/// Default split threshold in bytes (512KB serialized).
pub const DEFAULT_SPLIT_THRESHOLD: usize = 512 * 1024;

impl UidPack {
    /// Total number of UIDs across all blocks.
    pub fn total_uids(&self) -> u64 {
        self.blocks.iter().map(|b| u64::from(b.num_uids)).sum()
    }

    /// Check if the pack is empty.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Approximate serialized size in bytes.
    pub fn serialized_size(&self) -> usize {
        // Rough estimate: 4 (block_size) + per block (8 base + deltas + 4 num_uids + overhead)
        let mut size = 4;
        for block in &self.blocks {
            size += 8 + block.deltas.len() + 4 + 8; // base + deltas + num_uids + msgpack overhead
        }
        size
    }

    /// Check if this pack should be split (exceeds size threshold and has >1 block).
    ///
    /// Follows Dgraph's `shouldSplit()` pattern:
    /// split only when serialized size >= threshold AND multiple blocks exist.
    pub fn should_split(&self) -> bool {
        self.should_split_at(DEFAULT_SPLIT_THRESHOLD)
    }

    /// Check if this pack should be split with a custom threshold.
    pub fn should_split_at(&self, threshold: usize) -> bool {
        self.blocks.len() > 1 && self.serialized_size() >= threshold
    }

    /// Binary split at the midpoint block.
    ///
    /// Returns `(low_pack, high_pack)` where:
    /// - `low_pack` contains blocks `[0, mid)`
    /// - `high_pack` contains blocks `[mid, len)`
    ///
    /// Returns `None` if the pack has fewer than 2 blocks.
    pub fn bin_split(&self) -> Option<(Self, Self)> {
        if self.blocks.len() < 2 {
            return None;
        }

        let mid = self.blocks.len() / 2;

        let low = UidPack {
            block_size: self.block_size,
            blocks: self.blocks[..mid].to_vec(),
        };
        let high = UidPack {
            block_size: self.block_size,
            blocks: self.blocks[mid..].to_vec(),
        };

        Some((low, high))
    }

    /// Recursively split until no part exceeds the threshold.
    ///
    /// Returns a vec of `(start_uid, pack)` pairs. The start_uid is the
    /// base UID of the first block in each part.
    pub fn recursive_split(&self) -> Vec<(u64, UidPack)> {
        self.recursive_split_at(DEFAULT_SPLIT_THRESHOLD)
    }

    /// Recursively split with a custom threshold.
    pub fn recursive_split_at(&self, threshold: usize) -> Vec<(u64, UidPack)> {
        let mut result = Vec::new();
        self.recursive_split_inner(threshold, &mut result);
        result
    }

    fn recursive_split_inner(&self, threshold: usize, result: &mut Vec<(u64, UidPack)>) {
        if !self.should_split_at(threshold) {
            let start_uid = self.blocks.first().map_or(0, |b| b.base);
            result.push((start_uid, self.clone()));
            return;
        }

        if let Some((low, high)) = self.bin_split() {
            low.recursive_split_inner(threshold, result);
            high.recursive_split_inner(threshold, result);
        } else {
            let start_uid = self.blocks.first().map_or(0, |b| b.base);
            result.push((start_uid, self.clone()));
        }
    }

    /// Get the start UID of this pack (base of first block).
    pub fn start_uid(&self) -> Option<u64> {
        self.blocks.first().map(|b| b.base)
    }
}

/// Encodes sorted UIDs into compressed `UidPack`.
pub struct UidEncoder {
    block_size: usize,
    uids: Vec<u64>,
    blocks: Vec<UidBlock>,
}

impl UidEncoder {
    /// Create a new encoder with the default block size (256).
    pub fn new() -> Self {
        Self::with_block_size(DEFAULT_BLOCK_SIZE)
    }

    /// Create a new encoder with a custom block size.
    pub fn with_block_size(block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        Self {
            block_size,
            uids: Vec::with_capacity(block_size),
            blocks: Vec::new(),
        }
    }

    /// Add a UID. UIDs must be added in ascending sorted order.
    pub fn add(&mut self, uid: u64) {
        // If 32-MSB boundary crossed, pack current block first
        if let Some(&last) = self.uids.last() {
            debug_assert!(uid > last, "UIDs must be strictly ascending");
            if !match_32msb(last, uid) {
                self.pack_block();
            }
        }

        self.uids.push(uid);

        if self.uids.len() >= self.block_size {
            self.pack_block();
        }
    }

    /// Finalize and return the compressed pack.
    pub fn done(mut self) -> UidPack {
        self.pack_block();
        UidPack {
            block_size: self.block_size as u32,
            blocks: self.blocks,
        }
    }

    fn pack_block(&mut self) {
        if self.uids.is_empty() {
            return;
        }

        let base = self.uids[0];
        let num_uids = self.uids.len() as u32;

        let deltas = if self.uids.len() > 1 {
            // Collect u32 deltas from previous UID.
            let n = self.uids.len() - 1; // number of actual deltas
            let mut deltas_u32 = Vec::with_capacity(n);
            let mut prev = base;
            for &uid in &self.uids[1..] {
                let delta = uid - prev;
                debug_assert!(delta <= u64::from(u32::MAX), "delta exceeds u32");
                deltas_u32.push(delta as u32);
                prev = uid;
            }

            // Coder1234 requires element count to be a multiple of 4 (SIMD alignment).
            // Pad with zeros to the next multiple of 4; decoder reads only `n` values.
            let n_padded = (n + 3) & !3;
            deltas_u32.resize(n_padded, 0);

            // Encode with StreamVByte Coder1234: 4 values per tag byte.
            // Stored layout: [tag_bytes (exact) | data_bytes (variable)].
            use streamvbyte64::{Coder, Coder1234};
            let coder = Coder1234::new();
            let (tag_len, data_max) = Coder1234::max_compressed_bytes(n_padded);
            let mut buf = vec![0u8; tag_len + data_max];
            let (tags, data) = buf.split_at_mut(tag_len);
            let data_used = coder.encode(&deltas_u32, tags, data);
            buf.truncate(tag_len + data_used);
            buf
        } else {
            Vec::new()
        };

        self.blocks.push(UidBlock {
            base,
            deltas,
            num_uids,
        });
        self.uids.clear();
    }
}

impl Default for UidEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Decodes a `UidPack` back into sorted UIDs.
pub struct UidDecoder<'a> {
    pack: &'a UidPack,
    block_idx: usize,
}

impl<'a> UidDecoder<'a> {
    /// Create a decoder for the given pack.
    pub fn new(pack: &'a UidPack) -> Self {
        Self { pack, block_idx: 0 }
    }

    /// Decode the next block of UIDs. Returns `None` when exhausted.
    ///
    /// Decodes V5 StreamVByte format: `deltas` stores `[tag_bytes | data_bytes]`.
    /// `tag_len = (num_uids - 1 + 3) / 4` (1 tag byte per 4 deltas, Coder1234).
    pub fn next_block(&mut self) -> Option<Vec<u64>> {
        if self.block_idx >= self.pack.blocks.len() {
            return None;
        }

        let block = &self.pack.blocks[self.block_idx];
        self.block_idx += 1;

        let mut uids = Vec::with_capacity(block.num_uids as usize);
        uids.push(block.base);

        let n = (block.num_uids as usize).saturating_sub(1);
        if n > 0 {
            // Mirror encoder: n was padded to n_padded = (n+3)&!3 before encoding.
            // tag_len = n_padded / 4 = (n+3)/4.
            let n_padded = (n + 3) & !3;
            let tag_len = n_padded / 4;
            let tags = &block.deltas[..tag_len];
            let data = &block.deltas[tag_len..];

            use streamvbyte64::{Coder, Coder1234};
            let coder = Coder1234::new();
            // Decode n_padded values (Coder1234 SIMD constraint); only use first n.
            let mut decoded_u32 = vec![0u32; n_padded];
            coder.decode(tags, data, &mut decoded_u32);

            let mut prev = block.base;
            for delta in &decoded_u32[..n] {
                let uid = prev + u64::from(*delta);
                uids.push(uid);
                prev = uid;
            }
        }

        Some(uids)
    }

    /// Decode all UIDs from all blocks into a single sorted Vec.
    pub fn decode_all(&mut self) -> Vec<u64> {
        let mut all = Vec::with_capacity(self.pack.total_uids() as usize);
        while let Some(block_uids) = self.next_block() {
            all.extend(block_uids);
        }
        all
    }
}

/// Encode a sorted slice of UIDs into a compressed `UidPack`.
pub fn encode_uids(uids: &[u64]) -> UidPack {
    let mut enc = UidEncoder::new();
    for &uid in uids {
        enc.add(uid);
    }
    enc.done()
}

/// Decode a `UidPack` back into a sorted Vec of UIDs.
pub fn decode_uids(pack: &UidPack) -> Vec<u64> {
    let mut dec = UidDecoder::new(pack);
    dec.decode_all()
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
