//! A clustered node's inter-node identity: its certificate, key and trust
//! bundles, read from the files its configuration names, and read again on
//! a reload.

use std::sync::Arc;

use coordinode_raft::cluster::{DeploymentId, Enrollment};
use coordinode_wire::peer::{
    PeerIdentity, PeerProfile, PeerTarget, SecurityContext, SpiffeId, TrustAnchors,
};
use coordinode_wire::tls::{
    CertificateDer, PrivateKeyDer, load_certs, load_private_key, selected_provider,
};

use crate::config::server::NodeIdentitySettings;

/// A node's identity as configured, and the security context built from it.
pub(crate) struct NodeIdentity {
    settings: NodeIdentitySettings,
    context: Arc<SecurityContext>,
}

impl NodeIdentity {
    /// Read the files `settings` names and build the node's security context.
    ///
    /// # Errors
    ///
    /// A file cannot be read or parsed, or the certificate is not a valid
    /// node identity under the trust bundles.
    pub(crate) fn load(settings: NodeIdentitySettings) -> Result<Self, String> {
        let (chain, key) = credentials(&settings)?;
        let provider = selected_provider().map_err(|e| e.to_string())?;
        let context = SecurityContext::new(provider, chain, key, profile(&settings)?)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            settings,
            context: Arc::new(context),
        })
    }

    /// The security context every peer connection of the node is made under.
    pub(crate) fn context(&self) -> &Arc<SecurityContext> {
        &self.context
    }

    /// Read the files again and publish them as one replacement: a renewed
    /// certificate, changed trust bundles, or both. A replacement that does
    /// not validate as a whole changes nothing.
    ///
    /// # Errors
    ///
    /// A file cannot be read or parsed, the certificate is no valid identity
    /// under the bundles, or it names another identity than the node has.
    pub(crate) fn reload(&self) -> Result<(), String> {
        let (chain, key) = credentials(&self.settings)?;
        self.context
            .replace(chain, key, profile(&self.settings)?)
            .map_err(|e| e.to_string())
    }

    /// The deployment the configuration names, if any.
    ///
    /// # Errors
    ///
    /// It is not 32 hexadecimal digits.
    pub(crate) fn deployment(&self) -> Result<Option<DeploymentId>, String> {
        self.settings
            .deployment
            .as_deref()
            .map(|d| {
                d.parse::<DeploymentId>()
                    .map_err(|e| format!("node_identity.deployment '{d}': {e}"))
            })
            .transpose()
    }

    /// What a joining node enrolls with: the deployment it joins and the
    /// members it learns the membership from.
    ///
    /// # Errors
    ///
    /// The deployment or a member is missing or malformed.
    pub(crate) fn enrollment(&self) -> Result<Enrollment, String> {
        let deployment = self.deployment()?.ok_or(
            "node_identity.deployment is required to join: the deployment the cluster status \
             of a member reports",
        )?;
        if self.settings.enroll_from.is_empty() {
            return Err(
                "node_identity.enroll_from is required to join: a member to learn the \
                        membership from, with its identity"
                    .to_owned(),
            );
        }
        let authorities = self
            .settings
            .enroll_from
            .iter()
            .map(|peer| {
                Ok(PeerTarget {
                    addr: peer.addr.clone(),
                    identity: spiffe_identity(&peer.identity)
                        .map_err(|e| format!("node_identity.enroll_from: {e}"))?,
                })
            })
            .collect::<Result<_, String>>()?;
        Ok(Enrollment {
            deployment,
            authorities,
        })
    }
}

/// The SPIFFE ID `value` as a node identity.
///
/// # Errors
///
/// It is not a SPIFFE ID.
pub(crate) fn spiffe_identity(value: &str) -> Result<PeerIdentity, String> {
    SpiffeId::parse(value)
        .map(PeerIdentity::Spiffe)
        .map_err(|e| format!("'{value}' is not a SPIFFE ID: {e}"))
}

/// The certificate chain and key the settings name.
fn credentials(
    settings: &NodeIdentitySettings,
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>), String> {
    let chain = load_certs(&read(&settings.cert, "node_identity.cert")?)
        .map_err(|e| format!("node_identity.cert {}: {e}", settings.cert))?;
    let key = load_private_key(&read(&settings.key, "node_identity.key")?)
        .map_err(|e| format!("node_identity.key {}: {e}", settings.key))?;
    Ok((chain, key))
}

/// The SPIFFE profile trusting each configured domain under its own bundle.
fn profile(settings: &NodeIdentitySettings) -> Result<PeerProfile, String> {
    if settings.trust_bundles.is_empty() {
        return Err("node_identity.trust_bundles is empty: no peer could be verified".to_owned());
    }
    let bundles = settings
        .trust_bundles
        .iter()
        .map(|(domain, path)| {
            let key = format!("node_identity.trust_bundles.{domain}");
            let roots = load_certs(&read(path, &key)?).map_err(|e| format!("{key} {path}: {e}"))?;
            let anchors =
                TrustAnchors::from_der(&roots).map_err(|e| format!("{key} {path}: {e}"))?;
            Ok((domain.clone(), anchors))
        })
        .collect::<Result<_, String>>()?;
    Ok(PeerProfile::Spiffe { bundles })
}

fn read(path: &str, key: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{key} {path}: {e}"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
