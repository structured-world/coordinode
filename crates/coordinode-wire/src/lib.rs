//! Inter-node wire transport codec: pure-Rust zstd compression for gRPC.
//!
//! [`ZstdCodec`] is a drop-in [`tonic::codec::Codec`] that compresses every gRPC
//! message body on the wire with pure-Rust zstd ([`structured_zstd`], no C FFI
//! per ADR-013), independent of any node's on-disk storage codec. Wire it into
//! an inter-node service through tonic-build `codec_path = "coordinode_wire::ZstdCodec"`;
//! both ends use it symmetrically, so a service's payloads travel compressed
//! between replicas — most valuable on a bandwidth-constrained geo link. Shared
//! by every inter-node service (RaftService, SegmentTransferService).
//!
//! The transport zstd level is process-global, set once at startup from config
//! ([`set_wire_zstd_level`]); the default is a fast level suited to hot paths and
//! raised on links where bandwidth costs more than CPU.
//!
//! The compress side uses the one-shot slice entry point so the
//! `frame_content_size` is written into the zstd header and the decoder knows the
//! decompressed size up front; no streaming pledged-size handshake is needed
//! because a gRPC message is delivered to the codec whole.

use core::marker::PhantomData;
use std::io::Read;
use std::sync::atomic::{AtomicI32, Ordering};

use bytes::{Buf, BufMut};
use prost::Message;
use structured_zstd::decoding::StreamingDecoder;
use structured_zstd::encoding::{compress_slice_to_vec, CompressionLevel};
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};
use tonic::Status;

/// Process-global transport zstd level (C-zstd numbering). Positive 1..=22 trade
/// speed for ratio; NEGATIVE values select zstd ultra-fast modes (fastest, lowest
/// ratio). Set once at startup from config before any RPC, then read-mostly.
///
/// Defaults to 1 — zstd's fastest positive level, suited to hot replication
/// (measured ~11% of raw on a Raft batch, ~9× smaller on the wire). The
/// ultra-fast `-22` mode panics in `structured-zstd` 0.0.44's huff0 encoder (an
/// internal-error bug, filed upstream), so it is not the default. A
/// bandwidth-constrained link (db4 geo) can raise the level via config.
static WIRE_ZSTD_LEVEL: AtomicI32 = AtomicI32::new(1);

/// Set the inter-node transport zstd level. Call once at startup from config,
/// before the gRPC services begin serving. Process-wide; each node configures
/// its own level (like the per-node storage codec).
pub fn set_wire_zstd_level(level: i32) {
    WIRE_ZSTD_LEVEL.store(level, Ordering::Relaxed);
}

/// The currently configured transport zstd level.
#[must_use]
pub fn wire_zstd_level() -> i32 {
    WIRE_ZSTD_LEVEL.load(Ordering::Relaxed)
}

fn current_level() -> CompressionLevel {
    CompressionLevel::Level(WIRE_ZSTD_LEVEL.load(Ordering::Relaxed))
}

/// Prost-encode `item`, then zstd-compress the bytes at `level`. The one-shot
/// slice path writes the frame content size into the header.
///
/// # Errors
/// [`Status::internal`] if prost encoding fails.
fn encode_compressed<T: Message>(item: &T, level: CompressionLevel) -> Result<Vec<u8>, Status> {
    let mut raw = Vec::with_capacity(item.encoded_len());
    item.encode(&mut raw)
        .map_err(|e| Status::internal(format!("prost encode: {e}")))?;
    Ok(compress_slice_to_vec(&raw, level))
}

/// zstd-decompress `bytes`, then prost-decode into `U`.
///
/// # Errors
/// [`Status::internal`] if the zstd frame is malformed or prost decoding fails.
fn decode_compressed<U: Message + Default>(bytes: &[u8]) -> Result<U, Status> {
    let mut decoder = StreamingDecoder::new(bytes)
        .map_err(|e| Status::internal(format!("zstd decoder init: {e}")))?;
    let mut raw = Vec::new();
    decoder
        .read_to_end(&mut raw)
        .map_err(|e| Status::internal(format!("zstd decode: {e}")))?;
    U::decode(raw.as_slice()).map_err(|e| Status::internal(format!("prost decode: {e}")))
}

/// A [`tonic`] codec that zstd-compresses each message body on the wire.
///
/// Symmetric: the same codec serves both directions. Wire it into a service via
/// tonic-build `codec_path = "coordinode_wire::ZstdCodec"`.
pub struct ZstdCodec<T, U>(PhantomData<(T, U)>);

impl<T, U> Default for ZstdCodec<T, U> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T, U> Codec for ZstdCodec<T, U>
where
    T: Message + Send + 'static,
    U: Message + Default + Send + 'static,
{
    type Encode = T;
    type Decode = U;
    type Encoder = ZstdEncoder<T>;
    type Decoder = ZstdDecoder<U>;

    fn encoder(&mut self) -> Self::Encoder {
        ZstdEncoder(PhantomData)
    }

    fn decoder(&mut self) -> Self::Decoder {
        ZstdDecoder(PhantomData)
    }
}

/// Encoder half of [`ZstdCodec`].
pub struct ZstdEncoder<T>(PhantomData<T>);

impl<T: Message> Encoder for ZstdEncoder<T> {
    type Item = T;
    type Error = Status;

    fn encode(&mut self, item: T, dst: &mut EncodeBuf<'_>) -> Result<(), Status> {
        let compressed = encode_compressed(&item, current_level())?;
        dst.put_slice(&compressed);
        Ok(())
    }
}

/// Decoder half of [`ZstdCodec`].
pub struct ZstdDecoder<U>(PhantomData<U>);

impl<U: Message + Default> Decoder for ZstdDecoder<U> {
    type Item = U;
    type Error = Status;

    fn decode(&mut self, src: &mut DecodeBuf<'_>) -> Result<Option<U>, Status> {
        if !src.has_remaining() {
            return Ok(None);
        }
        // tonic delivers one whole gRPC message per decode call (it owns the
        // length-prefixed framing); take all of it as the compressed frame.
        let compressed = src.copy_to_bytes(src.remaining());
        decode_compressed(compressed.as_ref()).map(Some)
    }
}

#[cfg(test)]
mod tests;
