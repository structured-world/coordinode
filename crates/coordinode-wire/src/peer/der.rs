//! The few X.509 fields an identity check reads that the chain verifier
//! does not expose: every subjectAltName URI and the keyUsage bits of a
//! leaf. Strict DER (ITU-T X.690 §10): definite lengths in their shortest
//! form only, nothing left over; anything else is refused, never guessed.

/// A DER encoding the reader does not accept.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("malformed certificate encoding: {0}")]
pub struct DerError(pub &'static str);

type Result<T> = core::result::Result<T, DerError>;

const SEQUENCE: u8 = 0x30;
const OID: u8 = 0x06;
const BOOLEAN: u8 = 0x01;
const OCTET_STRING: u8 = 0x04;
const BIT_STRING: u8 = 0x03;
/// `[0] EXPLICIT Version` in a TBSCertificate (RFC 5280 §4.1).
const VERSION: u8 = 0xA0;
/// `[1] IMPLICIT issuerUniqueID` (RFC 5280 §4.1).
const ISSUER_UID: u8 = 0x81;
/// `[2] IMPLICIT subjectUniqueID` (RFC 5280 §4.1).
const SUBJECT_UID: u8 = 0x82;
/// `[3] EXPLICIT Extensions` (RFC 5280 §4.1).
const EXTENSIONS: u8 = 0xA3;
/// `uniformResourceIdentifier [6] IA5String` in GeneralName
/// (RFC 5280 §4.2.1.6).
const GENERAL_NAME_URI: u8 = 0x86;

/// id-ce-subjectAltName, 2.5.29.17.
const OID_SUBJECT_ALT_NAME: &[u8] = &[0x55, 0x1D, 0x11];
/// id-ce-keyUsage, 2.5.29.15.
const OID_KEY_USAGE: &[u8] = &[0x55, 0x1D, 0x0F];

/// The keyUsage bits an X.509-SVID check reads (RFC 5280 §4.2.1.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeyUsage {
    /// digitalSignature (bit 0).
    pub digital_signature: bool,
    /// keyCertSign (bit 5).
    pub key_cert_sign: bool,
    /// cRLSign (bit 6).
    pub crl_sign: bool,
}

/// What a leaf certificate says about itself for an identity check.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LeafFields {
    /// Every URI in subjectAltName, raw, in order; not decoded, so a URI
    /// that is not valid text is still counted.
    pub uris: Vec<Vec<u8>>,
    /// The keyUsage extension, `None` when the certificate has none.
    pub key_usage: Option<KeyUsage>,
    /// The end of the validity period, in seconds since the Unix epoch.
    pub not_after: u64,
}

/// Read the subjectAltName URIs and keyUsage of the certificate `der`.
///
/// # Errors
///
/// The encoding is not strict DER, or a field is not where RFC 5280 puts
/// it, or an extension appears twice (RFC 5280 §4.2).
pub fn leaf_fields(der: &[u8]) -> Result<LeafFields> {
    let mut cert = Reader::new(der);
    let mut cert_body = cert.expect(SEQUENCE)?;
    cert.finish()?;
    let mut tbs = cert_body.expect(SEQUENCE)?;
    if tbs.peek() == Some(VERSION) {
        tbs.next_tlv()?;
    }
    // serialNumber, signature, issuer.
    for _ in 0..3 {
        tbs.next_tlv()?;
    }
    let mut validity = tbs.expect(SEQUENCE)?;
    validity.next_tlv()?;
    let (time_tag, time) = validity.next_tlv()?;
    validity.finish()?;
    let not_after = parse_time(time_tag, time)?;
    // subject, subjectPublicKeyInfo.
    for _ in 0..2 {
        tbs.next_tlv()?;
    }
    for tag in [ISSUER_UID, SUBJECT_UID] {
        if tbs.peek() == Some(tag) {
            tbs.next_tlv()?;
        }
    }
    let mut fields = LeafFields {
        not_after,
        ..LeafFields::default()
    };
    let mut seen_san = false;
    if tbs.peek() == Some(EXTENSIONS) {
        let mut wrapper = tbs.expect(EXTENSIONS)?;
        let mut extensions = wrapper.expect(SEQUENCE)?;
        wrapper.finish()?;
        while !extensions.is_empty() {
            let mut extension = extensions.expect(SEQUENCE)?;
            let oid = extension.expect(OID)?.rest();
            if extension.peek() == Some(BOOLEAN) {
                extension.next_tlv()?;
            }
            let value = extension.expect(OCTET_STRING)?.rest();
            extension.finish()?;
            if oid == OID_SUBJECT_ALT_NAME {
                if seen_san {
                    return Err(DerError("subjectAltName appears twice"));
                }
                seen_san = true;
                fields.uris = san_uris(value)?;
            } else if oid == OID_KEY_USAGE {
                if fields.key_usage.is_some() {
                    return Err(DerError("keyUsage appears twice"));
                }
                fields.key_usage = Some(key_usage(value)?);
            }
        }
    }
    tbs.finish()?;
    Ok(fields)
}

/// The URIs of a GeneralNames value (RFC 5280 §4.2.1.6).
fn san_uris(value: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut outer = Reader::new(value);
    let mut names = outer.expect(SEQUENCE)?;
    outer.finish()?;
    let mut uris = Vec::new();
    while !names.is_empty() {
        let (tag, content) = names.next_tlv()?;
        if tag == GENERAL_NAME_URI {
            uris.push(content.to_vec());
        }
    }
    Ok(uris)
}

/// The keyUsage BIT STRING (RFC 5280 §4.2.1.3; X.690 §8.6).
fn key_usage(value: &[u8]) -> Result<KeyUsage> {
    let mut outer = Reader::new(value);
    let bits = outer.expect(BIT_STRING)?.rest();
    outer.finish()?;
    let (&unused, bytes) = bits
        .split_first()
        .ok_or(DerError("empty keyUsage bit string"))?;
    if unused > 7 || (bytes.is_empty() && unused != 0) {
        return Err(DerError("keyUsage bit string has a bad unused-bit count"));
    }
    let first = bytes.first().copied().unwrap_or(0);
    Ok(KeyUsage {
        digital_signature: first & 0x80 != 0,
        key_cert_sign: first & 0x04 != 0,
        crl_sign: first & 0x02 != 0,
    })
}

const UTC_TIME: u8 = 0x17;
const GENERALIZED_TIME: u8 = 0x18;

/// A certificate Time (RFC 5280 §4.1.2.5) as seconds since the Unix epoch.
/// UTCTime is `YYMMDDHHMMSSZ` with years 50-99 in the 1900s
/// (§4.1.2.5.1); GeneralizedTime is `YYYYMMDDHHMMSSZ` (§4.1.2.5.2).
fn parse_time(tag: u8, bytes: &[u8]) -> Result<u64> {
    let digits = |s: &[u8]| -> Result<u64> {
        s.iter().try_fold(0u64, |acc, &b| {
            if b.is_ascii_digit() {
                Ok(acc * 10 + u64::from(b - b'0'))
            } else {
                Err(DerError("non-digit in a certificate time"))
            }
        })
    };
    let (year, rest) = match (tag, bytes.len()) {
        (UTC_TIME, 13) => {
            let yy = digits(&bytes[..2])?;
            (if yy >= 50 { 1900 + yy } else { 2000 + yy }, &bytes[2..])
        }
        (GENERALIZED_TIME, 15) => (digits(&bytes[..4])?, &bytes[4..]),
        _ => return Err(DerError("certificate time in an unsupported form")),
    };
    if rest[10] != b'Z' {
        return Err(DerError("certificate time not in UTC"));
    }
    let month = digits(&rest[0..2])?;
    let day = digits(&rest[2..4])?;
    let hour = digits(&rest[4..6])?;
    let minute = digits(&rest[6..8])?;
    let second = digits(&rest[8..10])?;
    if !(1..=12).contains(&month)
        || !(1..=days_in_month(year, month)).contains(&day)
        || hour > 23
        || minute > 59
        || second > 59
        || year < 1970
    {
        return Err(DerError("certificate time out of range"));
    }
    let days = days_since_epoch(year, month, day);
    Ok(days * 86_400 + hour * 3600 + minute * 60 + second)
}

fn is_leap(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

fn days_in_month(year: u64, month: u64) -> u64 {
    match month {
        2 if is_leap(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Days from 1970-01-01 to the given date; the date is at or after it.
fn days_since_epoch(year: u64, month: u64, day: u64) -> u64 {
    let years: u64 = (1970..year)
        .map(|y| if is_leap(y) { 366 } else { 365 })
        .sum();
    let months: u64 = (1..month).map(|m| days_in_month(year, m)).sum();
    years + months + day - 1
}

/// A cursor over DER TLVs.
struct Reader<'a> {
    rest: &'a [u8],
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { rest: bytes }
    }

    fn is_empty(&self) -> bool {
        self.rest.is_empty()
    }

    fn peek(&self) -> Option<u8> {
        self.rest.first().copied()
    }

    /// What is left, consuming it.
    fn rest(self) -> &'a [u8] {
        self.rest
    }

    fn finish(&self) -> Result<()> {
        if self.rest.is_empty() {
            Ok(())
        } else {
            Err(DerError("trailing bytes"))
        }
    }

    /// The next TLV's tag and content.
    fn next_tlv(&mut self) -> Result<(u8, &'a [u8])> {
        let (&tag, after_tag) = self.rest.split_first().ok_or(DerError("truncated tag"))?;
        if tag & 0x1F == 0x1F {
            return Err(DerError("multi-byte tag"));
        }
        let (&first, after_len) = after_tag
            .split_first()
            .ok_or(DerError("truncated length"))?;
        let (len, after_len) = match first {
            short @ 0..=0x7F => (usize::from(short), after_len),
            0x80 => return Err(DerError("indefinite length")),
            long => {
                let count = usize::from(long & 0x7F);
                if count > core::mem::size_of::<usize>() || count > after_len.len() {
                    return Err(DerError("length does not fit"));
                }
                let (len_bytes, after) = after_len.split_at(count);
                if len_bytes[0] == 0 {
                    return Err(DerError("length with a leading zero byte"));
                }
                let len = len_bytes
                    .iter()
                    .fold(0usize, |acc, &b| (acc << 8) | usize::from(b));
                if len < 0x80 {
                    return Err(DerError("long form for a short length"));
                }
                (len, after)
            }
        };
        if len > after_len.len() {
            return Err(DerError("content runs past the end"));
        }
        let (content, rest) = after_len.split_at(len);
        self.rest = rest;
        Ok((tag, content))
    }

    /// The next TLV, which must carry `tag`, as a reader over its content.
    fn expect(&mut self, tag: u8) -> Result<Reader<'a>> {
        let (found, content) = self.next_tlv()?;
        if found == tag {
            Ok(Reader::new(content))
        } else {
            Err(DerError("unexpected tag"))
        }
    }
}

#[cfg(test)]
mod tests;
