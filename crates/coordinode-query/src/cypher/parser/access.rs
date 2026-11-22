//! The native ACL statements: `GRANT`, `REVOKE`, `TRANSFER` and
//! `CREATE / DROP / ALTER ROLE`, each one change to the grant catalog.

use coordinode_core::access::{NamespaceRole, PrincipalId};
use coordinode_core::graph::types::Value;
use coordinode_core::txn::proposal::{AccessChange, GranteeName};
use pest::iterators::Pair;

use super::{Expr, ParseError, Rule, build_string_literal, unquoted};

/// The catalog change an ACL statement names.
pub(super) fn change(pair: Pair<'_, Rule>) -> Result<AccessChange, ParseError> {
    let rule = pair.as_rule();
    let mut parts = Parts::of(pair)?;
    Ok(match rule {
        Rule::grant_clause => grant(&mut parts)?,
        Rule::revoke_clause => revoke(&mut parts)?,
        Rule::transfer_clause => transfer(&mut parts)?,
        Rule::create_role_clause => {
            let (home, name) = parts.role()?;
            let eligible = parts.list.take();
            let holder = parts.principals.pop_front();
            // An exclusive role has exactly one holder from its first entry
            // on, one of a set that is not empty.
            if let Some(eligible) = &eligible {
                match &holder {
                    Some(holder) if eligible.contains(holder) => {}
                    Some(_) => {
                        return Err(invalid(
                            "CREATE ROLE ... EXCLUSIVE AMONG: the holder must be one of the eligible principals",
                        ));
                    }
                    None => {
                        return Err(invalid(
                            "CREATE ROLE ... EXCLUSIVE AMONG requires HELD BY one of them",
                        ));
                    }
                }
            }
            AccessChange::CreateRole {
                home,
                name,
                eligible,
                holder,
            }
        }
        Rule::drop_role_clause => {
            let (home, name) = parts.role()?;
            AccessChange::DropRole { home, name }
        }
        Rule::alter_role_clause => {
            let (home, name) = parts.role()?;
            AccessChange::RoleEligible {
                home,
                name,
                eligible: parts.list.take().unwrap_or_default(),
            }
        }
        _ => return Err(invalid("not an access statement")),
    })
}

fn grant(parts: &mut Parts) -> Result<AccessChange, ParseError> {
    if parts.exclusive {
        return Err(invalid(
            "GRANT ROLE ... EXCLUSIVE AMONG: an exclusive role's eligible set is named by CREATE ROLE and changed by ALTER ROLE",
        ));
    }
    if parts.deployment {
        return Ok(AccessChange::DeploymentRights {
            principal: parts.principal()?,
            create_namespace: None,
            admin: Some(true),
        });
    }
    if parts.create_namespace {
        return Ok(AccessChange::DeploymentRights {
            principal: parts.principal()?,
            create_namespace: Some(true),
            admin: None,
        });
    }
    if let Some(role) = parts.level {
        return Ok(AccessChange::Grant {
            namespace: parts.namespace()?,
            grantee: parts.grantee()?,
            role,
        });
    }
    let (home, name) = parts.role()?;
    Ok(AccessChange::GrantRole {
        home,
        name,
        to: parts.principal()?,
    })
}

fn revoke(parts: &mut Parts) -> Result<AccessChange, ParseError> {
    if parts.deployment {
        return Ok(AccessChange::DeploymentRights {
            principal: parts.principal()?,
            create_namespace: None,
            admin: Some(false),
        });
    }
    if parts.create_namespace {
        return Ok(AccessChange::DeploymentRights {
            principal: parts.principal()?,
            create_namespace: Some(false),
            admin: None,
        });
    }
    if parts.all {
        return Ok(AccessChange::Revoke {
            namespaces: std::mem::take(&mut parts.namespaces),
            grantee: parts.grantee()?,
        });
    }
    if let Some(role) = parts.level {
        return Ok(AccessChange::RevokeGrant {
            namespace: parts.namespace()?,
            grantee: parts.grantee()?,
            role,
        });
    }
    let (home, name) = parts.role()?;
    Ok(AccessChange::RevokeRole {
        home,
        name,
        from: parts.principal()?,
    })
}

fn transfer(parts: &mut Parts) -> Result<AccessChange, ParseError> {
    if let Some(role) = parts.level {
        return Ok(AccessChange::Transfer {
            namespace: parts.namespace()?,
            role,
            from: parts.principal()?,
            to: parts.principal()?,
        });
    }
    let (home, name) = parts.role()?;
    Ok(AccessChange::TransferRole {
        home,
        name,
        from: parts.principal()?,
        to: parts.principal()?,
    })
}

/// The operands of an ACL statement, in the order it names them.
#[derive(Default)]
struct Parts {
    /// The namespace level named (`reader`, `writer`, `admin` on a namespace).
    level: Option<NamespaceRole>,
    /// Namespaces named by `ON NAMESPACE`, in order.
    namespaces: Vec<String>,
    /// Roles named, `(home, name)`, in order.
    roles: std::collections::VecDeque<(String, String)>,
    /// Principals named outside a list, in order.
    principals: std::collections::VecDeque<PrincipalId>,
    /// Whether the operand after `TO` / `FROM` is a role rather than a
    /// principal.
    grantee_is_role: bool,
    /// The parenthesised principal list, when one is named.
    list: Option<Vec<PrincipalId>>,
    /// `... ON DEPLOYMENT`.
    deployment: bool,
    /// `... CREATE NAMESPACE ...`.
    create_namespace: bool,
    /// `REVOKE ALL ...`.
    all: bool,
    /// `GRANT ROLE ... EXCLUSIVE AMONG (...)`.
    exclusive: bool,
}

impl Parts {
    fn of(pair: Pair<'_, Rule>) -> Result<Self, ParseError> {
        let mut parts = Self::default();
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::kw_deployment => parts.deployment = true,
                Rule::kw_namespace => {}
                Rule::kw_all => parts.all = true,
                Rule::kw_create => parts.create_namespace = true,
                Rule::namespace_role => parts.level = Some(level(inner.as_str())?),
                Rule::identifier => parts.namespaces.push(unquoted(inner.as_str())),
                Rule::grantee => {
                    let operand = inner
                        .into_inner()
                        .next()
                        .ok_or_else(|| invalid("a grantee names a principal or a role"))?;
                    if operand.as_rule() == Rule::role_ref {
                        parts.grantee_is_role = true;
                        parts.roles.push_back(role_ref(operand)?);
                    } else {
                        parts.principals.push_back(principal_ref(operand)?);
                    }
                }
                Rule::role_ref => parts.roles.push_back(role_ref(inner)?),
                Rule::principal_ref => parts.principals.push_back(principal_ref(inner)?),
                Rule::principal_list => parts.list = Some(principal_list(inner)?),
                Rule::grant_role_exclusive => parts.exclusive = true,
                _ => {}
            }
        }
        // `CREATE ROLE` starts with CREATE too: a role statement is no
        // namespace-creation right.
        if !parts.roles.is_empty() && parts.level.is_none() {
            parts.create_namespace = false;
        }
        Ok(parts)
    }

    fn namespace(&mut self) -> Result<String, ParseError> {
        if self.namespaces.is_empty() {
            return Err(invalid("the statement names no namespace"));
        }
        Ok(self.namespaces.remove(0))
    }

    fn principal(&mut self) -> Result<PrincipalId, ParseError> {
        self.principals
            .pop_front()
            .ok_or_else(|| invalid("the statement names no principal"))
    }

    fn role(&mut self) -> Result<(String, String), ParseError> {
        self.roles
            .pop_front()
            .ok_or_else(|| invalid("the statement names no role"))
    }

    fn grantee(&mut self) -> Result<GranteeName, ParseError> {
        if self.grantee_is_role {
            let (home, name) = self.role()?;
            return Ok(GranteeName::Role { home, name });
        }
        Ok(GranteeName::Principal(self.principal()?))
    }
}

fn level(text: &str) -> Result<NamespaceRole, ParseError> {
    match text.to_ascii_lowercase().as_str() {
        "reader" => Ok(NamespaceRole::Reader),
        "writer" => Ok(NamespaceRole::Writer),
        "admin" => Ok(NamespaceRole::Admin),
        other => Err(invalid(&format!("{other} is not a namespace role"))),
    }
}

/// `PRINCIPAL 'subject' IN 'trust scope'`: both literals as they decode,
/// checked as a principal is.
fn principal_ref(pair: Pair<'_, Rule>) -> Result<PrincipalId, ParseError> {
    let mut texts = pair
        .into_inner()
        .filter(|inner| inner.as_rule() == Rule::string_literal)
        .map(text);
    let subject = texts
        .next()
        .ok_or_else(|| invalid("PRINCIPAL names a subject"))??;
    let scope = texts
        .next()
        .ok_or_else(|| invalid("PRINCIPAL names a trust scope after IN"))??;
    PrincipalId::new(&subject, &scope).map_err(|e| invalid(&e.to_string()))
}

/// `ROLE name IN NAMESPACE home`, as `(home, name)`.
fn role_ref(pair: Pair<'_, Rule>) -> Result<(String, String), ParseError> {
    let mut names = pair
        .into_inner()
        .filter(|inner| inner.as_rule() == Rule::identifier)
        .map(|inner| unquoted(inner.as_str()));
    let name = names.next().ok_or_else(|| invalid("ROLE names a role"))?;
    let home = names
        .next()
        .ok_or_else(|| invalid("ROLE names its home after IN NAMESPACE"))?;
    Ok((home, name))
}

fn principal_list(pair: Pair<'_, Rule>) -> Result<Vec<PrincipalId>, ParseError> {
    pair.into_inner()
        .filter(|inner| inner.as_rule() == Rule::principal_ref)
        .map(principal_ref)
        .collect()
}

fn text(pair: Pair<'_, Rule>) -> Result<String, ParseError> {
    match build_string_literal(pair)? {
        Expr::Literal(Value::String(text)) => Ok(text),
        _ => Err(invalid("a string literal")),
    }
}

fn invalid(message: &str) -> ParseError {
    ParseError::Invalid(message.to_string())
}
