//! Inter-node wire: the transport codec and the authenticated connections
//! every inter-node service runs over.
//!
//! [`ZstdCodec`] is a drop-in [`tonic::codec::Codec`] that compresses every gRPC
//! message body on the wire with pure-Rust zstd ([`structured_zstd`], no C
//! FFI), independent of any node's on-disk storage codec. Wire it into
//! an inter-node service through tonic-build `codec_path = "coordinode_wire::ZstdCodec"`;
//! both ends use it symmetrically, so a service's payloads travel compressed
//! between replicas — most valuable on a bandwidth-constrained geo link. Shared
//! by every inter-node service (RaftService, SegmentTransferService).
//!
//! [`peer`] authenticates the nodes at both ends of a connection: each
//! database has its own security context, dials a peer expecting the
//! identity it must prove, and accepts connections that prove one.
//!
//! The transport zstd level is process-global, set once at startup from config
//! ([`set_wire_zstd_level`]); the default is a fast level suited to hot paths and
//! raised on links where bandwidth costs more than CPU.
//!
//! The compress side writes one-shot frames, so the `frame_content_size` is in
//! the zstd header and the decoder sizes its output up front; no streaming
//! pledged-size handshake is needed because a gRPC message is delivered to the
//! codec whole.
//!
//! Each thread keeps one compressor, one decoder and their scratch buffers
//! across messages, as a C caller keeps a `ZSTD_CCtx`/`ZSTD_DCtx`: building
//! the matcher tables and decode window for every message cost more than
//! compressing a small one.

use core::cell::RefCell;
use core::marker::PhantomData;
use std::sync::atomic::{AtomicI32, Ordering};

use bytes::{Buf, BufMut};
use prost::Message;
use structured_zstd::decoding::FrameDecoder;
use structured_zstd::encoding::{CompressionLevel, FrameCompressor};
use tonic::Status;
use tonic::codec::{BufferSettings, Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};

pub mod peer;
pub mod tls;

/// Process-global transport zstd level (C-zstd numbering). Positive 1..=22 trade
/// speed for ratio; NEGATIVE values select zstd ultra-fast modes (fastest, lowest
/// ratio). Set once at startup from config before any RPC, then read-mostly.
///
/// Defaults to 3 — zstd's canonical speed/ratio default, and the lowest level
/// that is panic-safe with `structured-zstd` 0.0.44: levels 1-2 (and negatives)
/// resolve to the Fast strategy, whose huffman build skips the table-log search
/// for messages below one 128 KiB block (i.e. essentially every Raft message)
/// and takes an UNGUARDED path that can hit an internal-error panic on some
/// literal distributions. Levels >= 3 use non-Fast strategies that always run
/// the guarded search path, so they never panic. Level 3 measures the same ~11%
/// of raw on a Raft batch as level 1 (~9x smaller on the wire). A
/// bandwidth-constrained link (db4 geo) can raise the level via config.
static WIRE_ZSTD_LEVEL: AtomicI32 = AtomicI32::new(3);

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

/// Scratch a thread keeps above this many bytes after a large message is
/// given back, so one snapshot-sized message does not pin its size per
/// thread.
const SCRATCH_KEPT: usize = 4 << 20;

/// What one thread reuses from one message to the next.
struct Contexts {
    /// The compressor, with the level it was built for.
    compressor: Option<(i32, FrameCompressor)>,
    decoder: FrameDecoder,
    /// Prost bytes of the message being encoded, or the decompressed bytes
    /// of the one being decoded.
    raw: Vec<u8>,
    /// The compressed frame being written.
    frame: Vec<u8>,
}

impl Contexts {
    /// Give back what a large message grew the scratch to.
    fn trim(&mut self) {
        for buf in [&mut self.raw, &mut self.frame] {
            if buf.capacity() > SCRATCH_KEPT {
                buf.clear();
                buf.shrink_to(SCRATCH_KEPT);
            }
        }
    }
}

// no-std: a codec context the caller owns and passes in.
std::thread_local! {
    static CONTEXTS: RefCell<Contexts> = RefCell::new(Contexts {
        compressor: None,
        decoder: FrameDecoder::new(),
        raw: Vec::new(),
        frame: Vec::new(),
    });
}

/// Prost-encode `item`, zstd-compress the bytes at `level` and append the
/// frame to `dst`. The frame carries its content size.
///
/// # Errors
/// [`Status::internal`] if prost encoding fails.
fn encode_compressed<T: Message>(
    item: &T,
    level: i32,
    dst: &mut impl BufMut,
) -> Result<(), Status> {
    CONTEXTS.with_borrow_mut(|contexts| {
        let Contexts {
            compressor,
            raw,
            frame,
            ..
        } = contexts;
        raw.clear();
        item.encode(raw)
            .map_err(|e| Status::internal(format!("prost encode: {e}")))?;
        let compressor = match compressor {
            Some((built_for, compressor)) if *built_for == level => compressor,
            slot => {
                &mut slot
                    .insert((level, FrameCompressor::new(CompressionLevel::Level(level))))
                    .1
            }
        };
        compressor.compress_independent_frame_into(raw, frame);
        dst.put_slice(frame);
        contexts.trim();
        Ok(())
    })
}

/// zstd-decompress `bytes`, then prost-decode into `U`.
///
/// # Errors
/// [`Status::internal`] if the zstd frame is malformed or prost decoding fails.
fn decode_compressed<U: Message + Default>(bytes: &[u8]) -> Result<U, Status> {
    CONTEXTS.with_borrow_mut(|contexts| {
        let Contexts { decoder, raw, .. } = contexts;
        let corrupt = |e: structured_zstd::decoding::errors::FrameDecoderError| {
            Status::internal(format!("zstd decode: {e}"))
        };
        // A body that is not a zstd frame is refused by its header.
        decoder.reset(bytes).map_err(corrupt)?;
        let size = usize::try_from(decoder.content_size())
            .map_err(|_| Status::internal("zstd decode: content size past the address space"))?;
        raw.clear();
        raw.reserve(size);
        decoder.decode_all_to_vec(bytes, raw).map_err(corrupt)?;
        let message =
            U::decode(raw.as_slice()).map_err(|e| Status::internal(format!("prost decode: {e}")));
        contexts.trim();
        message
    })
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
        encode_compressed(&item, wire_zstd_level(), dst)
    }

    fn buffer_settings(&self) -> BufferSettings {
        call_buffers()
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
        // length-prefixed framing); all of it is the compressed frame, read
        // in place when it is one contiguous piece.
        let remaining = src.remaining();
        if src.chunk().len() == remaining {
            let message = decode_compressed(src.chunk());
            src.advance(remaining);
            return message.map(Some);
        }
        let compressed = src.copy_to_bytes(remaining);
        decode_compressed(compressed.as_ref()).map(Some)
    }

    fn buffer_settings(&self) -> BufferSettings {
        call_buffers()
    }
}

/// tonic's per-call buffers, sized for the small message most calls carry:
/// a larger message reserves its own length, so starting small costs it
/// nothing, while tonic's 8 KiB default cost every call four times that.
fn call_buffers() -> BufferSettings {
    BufferSettings::new(512, 32 << 10)
}

#[cfg(test)]
mod tests;
