//! Client error types.

use thiserror::Error;

/// Errors returned by [`super::CoordinodeClient`].
#[derive(Debug, Error)]
pub enum ClientError {
    /// The endpoint string is not a valid URI.
    #[error("invalid endpoint URI: {0}")]
    InvalidEndpoint(String),

    /// Failed to establish the gRPC transport connection.
    #[error("connection failed: {0}")]
    Connection(#[from] tonic::transport::Error),

    /// The server returned a gRPC error status.
    #[error("gRPC error {}: {}", .0.code(), .0.message())]
    Grpc(#[from] tonic::Status),

    /// The persistent session to the server ended; a new request opens a new
    /// one.
    #[error("the session to the server ended")]
    SessionClosed,

    /// The server answered a session request with something other than what
    /// the request expects.
    #[error("unexpected session answer: {0}")]
    UnexpectedAnswer(String),

    /// Source-tracking metadata value contained characters invalid for HTTP/2
    /// header values. This is always a bug in the driver — file paths and line
    /// numbers should always be ASCII-clean.
    #[error("invalid metadata value: {0}")]
    InvalidMetadata(#[from] tonic::metadata::errors::InvalidMetadataValue),
}
