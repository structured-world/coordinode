//! An instant a statement names in text: RFC 3339 §5.6 `date-time`, read as
//! microseconds since the Unix epoch, the unit of the store's clock.

use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

/// The byte position of the separator between date and time in a
/// `full-date` of four year digits.
const SEPARATOR_AT: usize = 10;

/// The microseconds since the Unix epoch `text` names, an RFC 3339
/// `date-time` (§5.6): the date, `T`, `t` or a space between date and time
/// (§5.6 NOTE), seconds with an optional fraction of any length, and `Z`,
/// `z` or a `±hh:mm` offset. A leap second (§5.7) is the last instant of
/// its minute. Digits past the microsecond are dropped, rounding down.
///
/// # Errors
///
/// Why `text` is not such an instant, for a person reading the refusal.
pub fn parse_instant_us(text: &str) -> Result<i64, String> {
    // The separator is checked here, so that only these three are read
    // whatever the parser admits.
    let normalized;
    let text = match text.as_bytes().get(SEPARATOR_AT) {
        Some(b'T') => text,
        Some(b't' | b' ') => {
            let mut bytes = text.as_bytes().to_vec();
            bytes[SEPARATOR_AT] = b'T';
            // The byte replaced was ASCII, so the rest is still UTF-8.
            normalized = String::from_utf8(bytes).map_err(|e| e.to_string())?;
            normalized.as_str()
        }
        _ => {
            return Err(
                "the date and the time are separated by T, t or a space, as in \
                 2026-03-15T10:00:00Z"
                    .into(),
            );
        }
    };
    let instant = OffsetDateTime::parse(text, &Rfc3339).map_err(|e| e.to_string())?;
    // Nanoseconds of a four-digit year fit an i128 many times over, and
    // their microseconds an i64.
    let micros = instant.unix_timestamp_nanos().div_euclid(1_000);
    i64::try_from(micros).map_err(|_| "the instant is out of range".to_owned())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
