//! Multiplexed session core.
//!
//! Transport- and dialect-agnostic machinery for a long-lived multiplexed
//! session: concurrent dispatch of requests, a single outbound event writer,
//! and request-id correlation (with cursors and an interactive-transaction
//! sub-thread layered on as they land). The core consumes neutral
//! [`SessionOp`]s and produces neutral [`SessionEvent`]s, so a gRPC binding and
//! a pgwire binding drive the same machinery. The core never sees a wire frame
//! or a query dialect: mapping a protocol's frames to and from the neutral
//! types is the binding's job.
//!
//! See `arch/api/session-protocol.md`.

mod engine;
mod session;
mod types;

pub use engine::{CursorEngine, EngineError, QueryCursor};
pub use session::{InOp, OutEvent, Session, SessionManager};
pub use types::{ErrorCode, Ordering, SessionEvent, SessionOp, SessionStats};
