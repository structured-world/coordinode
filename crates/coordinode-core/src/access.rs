//! Who a write acts for, as the log carries it: the principal a session was
//! opened as, which every member checks the write's grants against when it
//! applies the entry.

use alloc::sync::Arc;

/// Most bytes a principal's subject or trust scope may hold: far above any
/// issuer's identifier, and small enough that a log entry naming one stays
/// small.
pub const MAX_PRINCIPAL_PART_BYTES: usize = 1024;

/// A verified subject and the issuer or trust scope that names it: the
/// identity a grant is held by. Two principals are the same exactly when
/// both parts are.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrincipalId {
    /// The stable subject, as its issuer names it.
    pub subject: Arc<str>,
    /// The issuer or trust scope the subject belongs to.
    pub trust_scope: Arc<str>,
}

/// A principal part that is empty or longer than
/// [`MAX_PRINCIPAL_PART_BYTES`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a principal's {part} holds {len} bytes; it must hold 1 to {MAX_PRINCIPAL_PART_BYTES}")]
pub struct BadPrincipal {
    /// Which part: `"subject"` or `"trust scope"`.
    pub part: &'static str,
    /// Its length in bytes.
    pub len: usize,
}

impl PrincipalId {
    /// The principal `subject` of `trust_scope`.
    ///
    /// # Errors
    ///
    /// [`BadPrincipal`] for an empty part or one past
    /// [`MAX_PRINCIPAL_PART_BYTES`].
    pub fn new(subject: &str, trust_scope: &str) -> Result<Self, BadPrincipal> {
        check_part("subject", subject)?;
        check_part("trust scope", trust_scope)?;
        Ok(Self {
            subject: Arc::from(subject),
            trust_scope: Arc::from(trust_scope),
        })
    }

    /// Bytes both parts hold.
    #[must_use]
    pub fn len(&self) -> usize {
        self.subject.len() + self.trust_scope.len()
    }

    /// Never true: both parts hold at least one byte.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// Serialized as `(subject, trust_scope)`.
impl serde::Serialize for PrincipalId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&(&*self.subject, &*self.trust_scope), serializer)
    }
}

/// Read back from `(subject, trust_scope)`, refusing what [`PrincipalId::new`]
/// refuses.
impl<'de> serde::Deserialize<'de> for PrincipalId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (subject, trust_scope): (alloc::string::String, alloc::string::String) =
            serde::Deserialize::deserialize(deserializer)?;
        Self::new(&subject, &trust_scope).map_err(serde::de::Error::custom)
    }
}

fn check_part(part: &'static str, value: &str) -> Result<(), BadPrincipal> {
    if value.is_empty() || value.len() > MAX_PRINCIPAL_PART_BYTES {
        return Err(BadPrincipal {
            part,
            len: value.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
