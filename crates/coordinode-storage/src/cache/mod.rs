//! Tiered block cache: cascading DRAM → NVMe → SSD → persistent storage.
//!
//! Components:
//! - [`tiered::TieredCache`] — multi-layer volatile cache with drain-through eviction
//! - [`access::AccessTracker`] — per-key access counter for eviction and heat map
//! - [`config::TieredCacheConfig`] — cache layer configuration
//!
//! Each cache layer is a volatile buffer on a specific device. Eviction from
//! a faster layer drains entries to the next slower layer. All layers are
//! lossy — power loss means cold restart, zero data loss.

pub mod access;
pub mod config;
pub mod tiered;
pub mod write_buffer;
