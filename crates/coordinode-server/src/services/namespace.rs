//! Which namespace a client's call or session works in.
//!
//! A gRPC client names it in the `x-coordinode-namespace` metadata of the
//! call, or of the session stream when it opens; a session keeps the
//! namespace it opened with. Naming one selects it and nothing more: whether
//! the caller may work in it is decided by its grants.

use tonic::Status;
use tonic::metadata::MetadataMap;

/// The metadata key a client names its namespace under.
pub const NAMESPACE_HEADER: &str = "x-coordinode-namespace";

/// The namespace `metadata` names, `None` when it names none.
///
/// # Errors
///
/// `INVALID_ARGUMENT` for a value that is not a non-empty text: a name that
/// cannot be read selects nothing, rather than the deployment's default.
pub fn from_metadata(metadata: &MetadataMap) -> Result<Option<String>, Status> {
    let Some(value) = metadata.get(NAMESPACE_HEADER) else {
        return Ok(None);
    };
    match value.to_str() {
        Ok(name) if !name.is_empty() => Ok(Some(name.to_owned())),
        _ => Err(Status::invalid_argument(format!(
            "{NAMESPACE_HEADER} must name a namespace"
        ))),
    }
}

/// `message` as a call naming the namespace a fresh store opens with, as a
/// client of a one-namespace server names it.
#[cfg(test)]
pub(crate) fn named<T>(message: T) -> tonic::Request<T> {
    let mut request = tonic::Request::new(message);
    request.metadata_mut().insert(
        NAMESPACE_HEADER,
        tonic::metadata::MetadataValue::from_static(coordinode_core::namespace::DEFAULT_NAMESPACE),
    );
    request
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
