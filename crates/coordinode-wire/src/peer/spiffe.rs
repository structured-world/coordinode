//! SPIFFE IDs, parsed by the rules of the SPIFFE ID standard
//! (<https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE-ID.md>).
//!
//! The path is opaque: an ID is compared whole, never by a prefix or by a
//! layout of its path.

use core::fmt;

/// The longest SPIFFE ID the standard admits, in bytes (SPIFFE-ID §2.3).
const MAX_ID_LEN: usize = 2048;

/// The longest trust domain name the standard admits, in bytes
/// (SPIFFE-ID §2.1).
const MAX_TRUST_DOMAIN_LEN: usize = 255;

const SCHEME: &str = "spiffe://";

/// Why a string is not a SPIFFE ID.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SpiffeIdError {
    /// The ID does not start with the lower-case `spiffe://` scheme.
    #[error("a SPIFFE ID starts with spiffe://")]
    Scheme,
    /// The ID names no trust domain.
    #[error("the SPIFFE ID names no trust domain")]
    EmptyTrustDomain,
    /// The trust domain holds a character the standard does not allow
    /// (it takes lower-case letters, digits, `.`, `-` and `_` only, so a
    /// port, user info, a query or upper case are all refused here).
    #[error("character {0:?} is not allowed in a SPIFFE trust domain")]
    TrustDomainChar(char),
    /// The trust domain is longer than 255 bytes.
    #[error("the SPIFFE trust domain is longer than 255 bytes")]
    TrustDomainTooLong,
    /// The path ends with `/`.
    #[error("a SPIFFE ID path does not end with /")]
    TrailingSlash,
    /// A path segment is empty.
    #[error("a SPIFFE ID path has an empty segment")]
    EmptySegment,
    /// A path segment is `.` or `..`.
    #[error("a SPIFFE ID path has a . or .. segment")]
    DotSegment,
    /// The path holds a character the standard does not allow (letters,
    /// digits, `.`, `-` and `_` only; no query, fragment or percent
    /// encoding).
    #[error("character {0:?} is not allowed in a SPIFFE ID path")]
    PathChar(char),
    /// The ID is longer than 2048 bytes.
    #[error("the SPIFFE ID is longer than 2048 bytes")]
    TooLong,
}

/// A SPIFFE ID that satisfies the standard: a trust domain and an opaque
/// path, compared as a whole.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpiffeId {
    /// The whole ID, `spiffe://<trust domain><path>`.
    id: String,
    /// Where the path starts in `id`: right after the trust domain.
    path_at: usize,
}

impl SpiffeId {
    /// Parse `input` as a SPIFFE ID.
    ///
    /// # Errors
    ///
    /// The rule of the standard the input breaks.
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_wire::peer::SpiffeId;
    /// let id = SpiffeId::parse("spiffe://example.org/node/1").unwrap();
    /// assert_eq!(id.trust_domain(), "example.org");
    /// assert!(SpiffeId::parse("spiffe://example.org/node/").is_err());
    /// ```
    pub fn parse(input: &str) -> Result<Self, SpiffeIdError> {
        let rest = input.strip_prefix(SCHEME).ok_or(SpiffeIdError::Scheme)?;
        if input.len() > MAX_ID_LEN {
            return Err(SpiffeIdError::TooLong);
        }
        let domain_len = rest.find('/').unwrap_or(rest.len());
        let (domain, path) = rest.split_at(domain_len);
        if domain.is_empty() {
            return Err(SpiffeIdError::EmptyTrustDomain);
        }
        if let Some(c) = domain.chars().find(|c| !is_trust_domain_char(*c)) {
            return Err(SpiffeIdError::TrustDomainChar(c));
        }
        if domain.len() > MAX_TRUST_DOMAIN_LEN {
            return Err(SpiffeIdError::TrustDomainTooLong);
        }
        if !path.is_empty() {
            check_path(path)?;
        }
        Ok(Self {
            id: input.to_owned(),
            path_at: SCHEME.len() + domain_len,
        })
    }

    /// The trust domain, which names the trust bundle the ID is checked
    /// against.
    #[must_use]
    pub fn trust_domain(&self) -> &str {
        &self.id[SCHEME.len()..self.path_at]
    }

    /// The path, empty for the ID of the trust domain itself. Opaque: it
    /// is compared only as part of the whole ID.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.id[self.path_at..]
    }

    /// The whole ID.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.id
    }
}

impl fmt::Display for SpiffeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.id)
    }
}

/// SPIFFE-ID §2.1: lower-case letters, digits, dot, dash and underscore.
fn is_trust_domain_char(c: char) -> bool {
    matches!(c, 'a'..='z' | '0'..='9' | '.' | '-' | '_')
}

/// SPIFFE-ID §2.2: letters, digits, dot, dash and underscore.
fn is_path_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_')
}

/// SPIFFE-ID §2.2: `/`-separated segments, none empty, `.` or `..`, and no
/// trailing `/`. `path` starts with `/`.
fn check_path(path: &str) -> Result<(), SpiffeIdError> {
    if path.ends_with('/') {
        return Err(SpiffeIdError::TrailingSlash);
    }
    for segment in path[1..].split('/') {
        if segment.is_empty() {
            return Err(SpiffeIdError::EmptySegment);
        }
        if segment == "." || segment == ".." {
            return Err(SpiffeIdError::DotSegment);
        }
        if let Some(c) = segment.chars().find(|c| !is_path_char(*c)) {
            return Err(SpiffeIdError::PathChar(c));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
