//! Server configuration: CLI args, config file, env vars.

/// Operational mode for the `coordinode` binary.
///
/// CE supports only [`ServeMode::Full`]. `--mode=compute` and
/// `--mode=storage` require coordinode-ee and are rejected with a clear
/// error by the CE binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ServeMode {
    /// Full monolith: gRPC (:7080) + REST (:7081) + Bolt (:7082, stub) +
    /// WebSocket (:7083, stub) + ops (:7084).
    ///
    /// This is the only mode available in the CE binary.
    #[default]
    Full,
}

impl ServeMode {
    /// Parse from the string value of `--mode=<value>`.
    ///
    /// Returns an error for EE-only modes so callers can print a clear
    /// "requires coordinode-ee" message.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "full" => Ok(Self::Full),
            "compute" | "storage" => Err(format!(
                "--mode={s} requires coordinode-ee. \
                 CE supports only --mode=full."
            )),
            other => Err(format!("unknown --mode='{other}'. CE supports: full")),
        }
    }
}

impl std::fmt::Display for ServeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "full"),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn full_mode_parses() {
        assert_eq!(ServeMode::parse("full").unwrap(), ServeMode::Full);
    }

    #[test]
    fn compute_mode_rejected_with_ee_message() {
        let err = ServeMode::parse("compute").unwrap_err();
        assert!(err.contains("coordinode-ee"), "expected EE mention: {err}");
    }

    #[test]
    fn storage_mode_rejected_with_ee_message() {
        let err = ServeMode::parse("storage").unwrap_err();
        assert!(err.contains("coordinode-ee"), "expected EE mention: {err}");
    }

    #[test]
    fn unknown_mode_rejected() {
        let err = ServeMode::parse("sharded").unwrap_err();
        assert!(err.contains("unknown"), "expected 'unknown' in: {err}");
    }

    #[test]
    fn default_is_full() {
        assert_eq!(ServeMode::default(), ServeMode::Full);
    }
}
