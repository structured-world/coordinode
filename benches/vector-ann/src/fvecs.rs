//! Texmex INRIA `.fvecs` / `.ivecs` reader.
//!
//! ## Format (canonical for SIFT1M, GIST, etc.)
//!
//! Each vector is laid out back-to-back:
//!
//! ```text
//! ┌──── 4 bytes ────┐┌────── dim × 4 bytes ───────┐
//! │ dim (LE u32)    ││ values (LE f32 or i32)      │
//! └─────────────────┘└─────────────────────────────┘
//! ```
//!
//! `.fvecs` carries f32 vectors (data + queries); `.ivecs` carries
//! i32 vectors (ground-truth neighbour ids). Every vector in the
//! file has the SAME dim — the per-vector dim prefix is redundant
//! but we honour it for format spec compliance and to detect
//! corrupted files.
//!
//! Source: <http://corpus-texmex.irisa.fr/> — SIFT1M lives here as
//! `sift.tar.gz` containing `sift_base.fvecs` (1 000 000 × 128),
//! `sift_query.fvecs` (10 000 × 128), `sift_groundtruth.ivecs`
//! (10 000 × 100).

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use thiserror::Error;

/// (No `#[diagnostic::on_unimplemented]` — that attribute is for
/// traits, not enums. The canonical reader error is documented in
/// the module `//!`.)
#[derive(Debug, Error)]
pub enum FvecsError {
    /// Underlying IO failure (file not found, permission denied,
    /// short read).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// File contains an unexpected dim header (e.g. dim 0, or
    /// per-vector dim that differs from the first vector). Likely
    /// a corrupted download or wrong format.
    #[error("unexpected dim: first {first}, vector {idx} reports {actual}")]
    DimMismatch { first: u32, idx: usize, actual: u32 },

    /// File size is not a multiple of (4 + dim × 4) — a sure sign
    /// of truncation.
    #[error("file size {bytes} is not a multiple of vector stride {stride} — likely truncated")]
    Truncated { bytes: u64, stride: u64 },
}

/// Read every f32 vector in a `.fvecs` file. Returns `(dim,
/// vectors)` where `vectors[i * dim..(i + 1) * dim]` is the i-th
/// vector — flat row-major layout for ergonomic SIMD-friendly use
/// downstream.
pub fn read_fvecs(path: impl AsRef<Path>) -> Result<(usize, Vec<f32>), FvecsError> {
    let path = path.as_ref();
    let file_size = std::fs::metadata(path)?.len();
    let mut reader = BufReader::new(File::open(path)?);
    let mut first_dim: Option<u32> = None;
    let mut vectors: Vec<f32> = Vec::new();
    let mut idx = 0usize;
    loop {
        let mut dim_buf = [0u8; 4];
        match reader.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let dim = u32::from_le_bytes(dim_buf);
        match first_dim {
            None => {
                first_dim = Some(dim);
                let stride = 4u64 + u64::from(dim) * 4;
                if file_size % stride != 0 {
                    return Err(FvecsError::Truncated {
                        bytes: file_size,
                        stride,
                    });
                }
                vectors.reserve(((file_size / stride) as usize) * dim as usize);
            }
            Some(f) if f != dim => {
                return Err(FvecsError::DimMismatch {
                    first: f,
                    idx,
                    actual: dim,
                });
            }
            _ => {}
        }
        let mut row = vec![0u8; (dim as usize) * 4];
        reader.read_exact(&mut row)?;
        for chunk in row.chunks_exact(4) {
            let arr: [u8; 4] = chunk
                .try_into()
                .map_err(|_| FvecsError::Io(std::io::Error::other("chunk size mismatch")))?;
            vectors.push(f32::from_le_bytes(arr));
        }
        idx += 1;
    }
    Ok((first_dim.unwrap_or(0) as usize, vectors))
}

/// Read every i32 vector in an `.ivecs` file (ground-truth
/// nearest-neighbour ids). Same layout as [`read_fvecs`].
pub fn read_ivecs(path: impl AsRef<Path>) -> Result<(usize, Vec<i32>), FvecsError> {
    let path = path.as_ref();
    let file_size = std::fs::metadata(path)?.len();
    let mut reader = BufReader::new(File::open(path)?);
    let mut first_dim: Option<u32> = None;
    let mut vectors: Vec<i32> = Vec::new();
    let mut idx = 0usize;
    loop {
        let mut dim_buf = [0u8; 4];
        match reader.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let dim = u32::from_le_bytes(dim_buf);
        match first_dim {
            None => {
                first_dim = Some(dim);
                let stride = 4u64 + u64::from(dim) * 4;
                if file_size % stride != 0 {
                    return Err(FvecsError::Truncated {
                        bytes: file_size,
                        stride,
                    });
                }
                vectors.reserve(((file_size / stride) as usize) * dim as usize);
            }
            Some(f) if f != dim => {
                return Err(FvecsError::DimMismatch {
                    first: f,
                    idx,
                    actual: dim,
                });
            }
            _ => {}
        }
        let mut row = vec![0u8; (dim as usize) * 4];
        reader.read_exact(&mut row)?;
        for chunk in row.chunks_exact(4) {
            let arr: [u8; 4] = chunk
                .try_into()
                .map_err(|_| FvecsError::Io(std::io::Error::other("chunk size mismatch")))?;
            vectors.push(i32::from_le_bytes(arr));
        }
        idx += 1;
    }
    Ok((first_dim.unwrap_or(0) as usize, vectors))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_fvecs(path: &Path, dim: u32, vectors: &[Vec<f32>]) {
        let mut f = File::create(path).unwrap();
        for v in vectors {
            f.write_all(&dim.to_le_bytes()).unwrap();
            for x in v {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        }
    }

    fn write_ivecs(path: &Path, dim: u32, vectors: &[Vec<i32>]) {
        let mut f = File::create(path).unwrap();
        for v in vectors {
            f.write_all(&dim.to_le_bytes()).unwrap();
            for x in v {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        }
    }

    #[test]
    fn read_fvecs_round_trips_synthetic_input() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("synth.fvecs");
        let original = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        write_fvecs(&path, 3, &original);
        let (dim, flat) = read_fvecs(&path).unwrap();
        assert_eq!(dim, 3);
        assert_eq!(flat.len(), 9);
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn read_ivecs_round_trips_synthetic_input() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("synth.ivecs");
        let original = vec![vec![10, 20, 30, 40], vec![50, 60, 70, 80]];
        write_ivecs(&path, 4, &original);
        let (dim, flat) = read_ivecs(&path).unwrap();
        assert_eq!(dim, 4);
        assert_eq!(flat, vec![10, 20, 30, 40, 50, 60, 70, 80]);
    }

    #[test]
    fn truncated_file_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad.fvecs");
        // Write only the dim header, no payload — truncated mid-vec.
        let mut f = File::create(&path).unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&1.0_f32.to_le_bytes()).unwrap();
        // Stop here: claimed dim=3 but only 1 f32 follows.
        drop(f);
        match read_fvecs(&path) {
            Err(FvecsError::Truncated { .. }) => {}
            other => panic!("expected Truncated, got {other:?}"),
        }
    }
}
