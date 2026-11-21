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

/// The databases a service answers from: the one of each namespace a call
/// names. Cheap to clone.
#[derive(Clone)]
pub struct Databases {
    served: coordinode_embed::replication::ServedDatabase,
}

impl Databases {
    /// The databases a member serves.
    pub fn serving(served: coordinode_embed::replication::ServedDatabase) -> Self {
        Self { served }
    }

    /// `database` alone: a call naming its namespace reaches it, a call
    /// naming another namespace of its deployment reaches a session of it.
    pub fn of(database: std::sync::Arc<parking_lot::RwLock<coordinode_embed::Database>>) -> Self {
        let served = coordinode_embed::replication::ServedDatabase::default();
        // A fresh registry serves nothing yet: the set cannot fail.
        let _ = served.set(database);
        Self { served }
    }

    /// The database of the namespace a call's `metadata` names.
    ///
    /// # Errors
    ///
    /// `INVALID_ARGUMENT` (`NAMESPACE_REQUIRED`) when the call names none,
    /// `NOT_FOUND` (`UNKNOWN_NAMESPACE`) for one the deployment does not
    /// hold, `UNAVAILABLE` before the member serves.
    pub fn for_call(
        &self,
        metadata: &MetadataMap,
    ) -> Result<std::sync::Arc<parking_lot::RwLock<coordinode_embed::Database>>, Status> {
        let database = match from_metadata(metadata)? {
            Some(namespace) => self.served.in_namespace(&namespace),
            None => Err(coordinode_embed::DatabaseError::NamespaceRequired),
        };
        database.map_err(crate::services::cypher::db_error_to_status)
    }

    /// The database of `namespace`, for a transport that names it its own
    /// way (a PostgreSQL connection's `database`).
    ///
    /// # Errors
    ///
    /// [`NamespaceRequired`](coordinode_embed::DatabaseError::NamespaceRequired)
    /// for `None`, `UnknownNamespace` for one the deployment does not hold,
    /// `NotServing` before the member serves.
    pub fn in_namespace(
        &self,
        namespace: Option<&str>,
    ) -> Result<
        std::sync::Arc<parking_lot::RwLock<coordinode_embed::Database>>,
        coordinode_embed::DatabaseError,
    > {
        match namespace {
            Some(namespace) if !namespace.is_empty() => self.served.in_namespace(namespace),
            _ => Err(coordinode_embed::DatabaseError::NamespaceRequired),
        }
    }

    /// The database the member opened with, for work of the deployment
    /// rather than of one namespace.
    ///
    /// # Errors
    ///
    /// `UNAVAILABLE` before the member serves.
    pub fn own(
        &self,
    ) -> Result<std::sync::Arc<parking_lot::RwLock<coordinode_embed::Database>>, Status> {
        self.served.get().cloned().ok_or_else(|| {
            crate::services::cypher::db_error_to_status(coordinode_embed::DatabaseError::NotServing)
        })
    }
}

#[cfg(test)]
impl Databases {
    /// The database a service test built its service over.
    #[allow(clippy::expect_used)]
    pub(crate) fn served_primary(
        &self,
    ) -> &std::sync::Arc<parking_lot::RwLock<coordinode_embed::Database>> {
        self.served
            .get()
            .expect("a test's service serves its database")
    }
}

/// The calls of a service test module, each naming the namespace a fresh
/// store opens with: a test module imports it as `Request`, so every call it
/// builds names its namespace as every client does.
#[cfg(test)]
pub(crate) struct NamedRequest;

#[cfg(test)]
impl NamedRequest {
    /// `message` as a call naming the namespace ([`named`]). Named `new` so
    /// a test module importing this as `Request` builds its calls unchanged.
    #[allow(clippy::new_ret_no_self)]
    pub(crate) fn new<T>(message: T) -> tonic::Request<T> {
        named(message)
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
