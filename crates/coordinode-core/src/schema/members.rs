//! Unordered members of a document: the lists a property declares compared
//! by what they hold, never by where a member stands.
//!
//! A declaration names a list by the fields leading to it below the
//! property; a list on the way is passed through, so the name reaches the
//! list inside every element. Everywhere, declared or not, the fields of a
//! map compare by name, a list that is not declared by position, and scalars
//! by type and value.

use core::cmp::Ordering;

use rmpv::Value;

/// What a document breaks of the unordered members declared for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemberError {
    /// The list at `path` holds a member twice.
    Repeated {
        /// The fields leading to the list, joined by `.`; empty for the
        /// value itself.
        path: String,
    },
    /// The value at `path` is declared a list of unordered members and is
    /// neither a list nor absent.
    NotAList {
        /// As for [`Self::Repeated`].
        path: String,
    },
}

/// The declared paths still leading somewhere below the value in hand, each
/// cut down to what remains of it.
type Remaining<'p, S> = Vec<&'p [S]>;

/// Whether the value in hand is itself declared: a path ends at it.
fn ends_here<S>(paths: &[&[S]]) -> bool {
    paths.iter().any(|p| p.is_empty())
}

/// The paths that go on below the value in hand.
fn below<'p, S>(paths: &[&'p [S]]) -> Remaining<'p, S> {
    paths.iter().copied().filter(|p| !p.is_empty()).collect()
}

/// The paths that go on through field `key`, without it.
fn through<'p, S: AsRef<str>>(paths: &[&'p [S]], key: &Value) -> Remaining<'p, S> {
    let Some(key) = key.as_str() else {
        return Vec::new();
    };
    paths
        .iter()
        .filter_map(|p| match p.split_first() {
            Some((first, rest)) if first.as_ref() == key => Some(rest),
            _ => None,
        })
        .collect()
}

/// The kinds of value, in the order values of different kinds sort in. Every
/// number is one kind, whatever holds it: a number is the value it
/// represents.
fn rank(value: &Value) -> u8 {
    match value {
        Value::Nil => 0,
        Value::Boolean(_) => 1,
        Value::Integer(_) | Value::F32(_) | Value::F64(_) => 2,
        Value::String(_) => 5,
        Value::Binary(_) => 6,
        Value::Array(_) => 7,
        Value::Map(_) => 8,
        Value::Ext(..) => 9,
    }
}

/// A number as it is compared: an integer whole, a float as the binary
/// value it holds.
#[derive(Clone, Copy)]
enum Number {
    Integer(i128),
    Float(f64),
}

fn number(value: &Value) -> Option<Number> {
    match value {
        Value::Integer(i) => i
            .as_i64()
            .map(i128::from)
            .or_else(|| i.as_u64().map(i128::from))
            .map(Number::Integer),
        Value::F32(f) => Some(Number::Float(f64::from(*f))),
        Value::F64(f) => Some(Number::Float(*f)),
        _ => None,
    }
}

/// An integer against a float, by the values they represent and without
/// rounding either: the float's whole part is compared as an integer, and
/// its fraction decides between equal whole parts. An infinity lies beyond
/// every integer, and a NaN after everything.
fn integer_to_float(integer: i128, float: f64) -> Ordering {
    // Every integer a document holds lies within these, which a float
    // represents exactly.
    const BELOW: f64 = -9_223_372_036_854_775_808.0;
    const ABOVE: f64 = 18_446_744_073_709_551_616.0;
    if float.is_nan() || float >= ABOVE {
        return Ordering::Less;
    }
    if float < BELOW {
        return Ordering::Greater;
    }
    let whole = float.trunc();
    // Exact: `whole` is an integer within the range above.
    integer.cmp(&(whole as i128)).then_with(|| {
        let fraction = float - whole;
        0.0_f64.partial_cmp(&fraction).unwrap_or(Ordering::Equal)
    })
}

/// Numbers by the values they represent: `1` and `1.0` are one number, as
/// are the two zeros; a NaN is the value it is and sorts after the rest.
fn number_order(a: Number, b: Number) -> Ordering {
    match (a, b) {
        (Number::Integer(a), Number::Integer(b)) => a.cmp(&b),
        (Number::Integer(a), Number::Float(b)) => integer_to_float(a, b),
        (Number::Float(a), Number::Integer(b)) => integer_to_float(b, a).reverse(),
        (Number::Float(a), Number::Float(b)) => match a.partial_cmp(&b) {
            Some(order) => order,
            None => a.is_nan().cmp(&b.is_nan()).then_with(|| a.total_cmp(&b)),
        },
    }
}

/// A total order of values, by kind and then by content: lists element by
/// element, maps pair by pair as they stand. Two values are the same value
/// exactly when it calls them equal, once their maps and declared lists
/// stand in this order ([`canonical`]).
fn order(a: &Value, b: &Value) -> Ordering {
    if let (Some(a), Some(b)) = (number(a), number(b)) {
        return number_order(a, b);
    }
    match (a, b) {
        (Value::Boolean(a), Value::Boolean(b)) => a.cmp(b),
        (Value::String(a), Value::String(b)) => a.as_bytes().cmp(b.as_bytes()),
        (Value::Binary(a), Value::Binary(b)) => a.cmp(b),
        (Value::Array(a), Value::Array(b)) => a
            .iter()
            .zip(b)
            .map(|(a, b)| order(a, b))
            .find(|o| o.is_ne())
            .unwrap_or_else(|| a.len().cmp(&b.len())),
        (Value::Map(a), Value::Map(b)) => a
            .iter()
            .zip(b)
            .map(|((ak, av), (bk, bv))| order(ak, bk).then_with(|| order(av, bv)))
            .find(|o| o.is_ne())
            .unwrap_or_else(|| a.len().cmp(&b.len())),
        (Value::Ext(at, a), Value::Ext(bt, b)) => at.cmp(bt).then_with(|| a.cmp(b)),
        _ => rank(a).cmp(&rank(b)),
    }
}

/// `value` with every map's fields and every declared list's members in
/// [`order`]: the one form all values that are the same value share.
fn canonical<S: AsRef<str>>(value: &Value, paths: &[&[S]]) -> Value {
    match value {
        Value::Map(fields) => {
            let mut fields: Vec<(Value, Value)> = fields
                .iter()
                .map(|(key, value)| (key.clone(), canonical(value, &through(paths, key))))
                .collect();
            fields.sort_by(|(a, _), (b, _)| order(a, b));
            Value::Map(fields)
        }
        Value::Array(items) => {
            let inner = below(paths);
            let mut items: Vec<Value> = items.iter().map(|i| canonical(i, &inner)).collect();
            if ends_here(paths) {
                items.sort_by(order);
            }
            Value::Array(items)
        }
        other => other.clone(),
    }
}

/// Whether `a` and `b` are the same document under the unordered members
/// `sets` declares: the fields of a map by name, the members of a declared
/// list as a whole whatever their order, any other list by position.
pub fn same_document<S: AsRef<str>>(a: &Value, b: &Value, sets: &[&[S]]) -> bool {
    same(a, b, sets)
}

fn same<S: AsRef<str>>(a: &Value, b: &Value, paths: &[&[S]]) -> bool {
    match (a, b) {
        (Value::Map(left), Value::Map(right)) => {
            left.len() == right.len() && same_fields(left, right, paths)
        }
        (Value::Array(left), Value::Array(right)) => {
            if left.len() != right.len() {
                return false;
            }
            let inner = below(paths);
            let in_place = || left.iter().zip(right).all(|(l, r)| same(l, r, &inner));
            if !ends_here(paths) {
                return in_place();
            }
            // Members in the same order are the common case and need no copy.
            in_place() || {
                let sorted = |items: &[Value]| {
                    let mut items: Vec<Value> =
                        items.iter().map(|i| canonical(i, &inner)).collect();
                    items.sort_by(order);
                    items
                };
                let (left, right) = (sorted(left), sorted(right));
                left.iter().zip(&right).all(|(l, r)| order(l, r).is_eq())
            }
        }
        _ => order(a, b).is_eq(),
    }
}

/// Whether two maps of one length hold the same fields. Fields standing in
/// the same order are compared in place; from the first that does not, both
/// maps are read in the order of their names.
fn same_fields<S: AsRef<str>>(
    left: &[(Value, Value)],
    right: &[(Value, Value)],
    paths: &[&[S]],
) -> bool {
    let pair = |(lk, lv): &(Value, Value), (rk, rv): &(Value, Value)| {
        order(lk, rk).is_eq() && same(lv, rv, &through(paths, lk))
    };
    let aligned = left
        .iter()
        .zip(right)
        .take_while(|((lk, _), (rk, _))| order(lk, rk).is_eq())
        .count();
    if !left
        .iter()
        .zip(right)
        .take(aligned)
        .all(|(l, r)| pair(l, r))
    {
        return false;
    }
    if aligned == left.len() {
        return true;
    }
    let (left, right) = (by_name(&left[aligned..]), by_name(&right[aligned..]));
    left.iter().zip(&right).all(|(l, r)| pair(l, r))
}

/// The fields of a map, in the order of their names.
fn by_name(fields: &[(Value, Value)]) -> Vec<&(Value, Value)> {
    let mut fields: Vec<&(Value, Value)> = fields.iter().collect();
    fields.sort_by(|(a, _), (b, _)| order(a, b));
    fields
}

/// Check `value` against the unordered members `sets` declares: every
/// declared list holds each member once, and what stands at a declared path
/// is a list or absent.
///
/// # Errors
///
/// The first list found holding a member twice, or the first declared path
/// holding something else than a list, with the path to it.
pub fn check_members<S: AsRef<str>>(value: &Value, sets: &[&[S]]) -> Result<(), MemberError> {
    let mut at = Vec::new();
    check(value, sets, &mut at)
}

fn check<'v, S: AsRef<str>>(
    value: &'v Value,
    paths: &[&[S]],
    at: &mut Vec<&'v str>,
) -> Result<(), MemberError> {
    let inner = below(paths);
    if ends_here(paths) {
        match value {
            Value::Array(items) => {
                let mut members: Vec<Value> = items.iter().map(|i| canonical(i, &inner)).collect();
                members.sort_by(order);
                if members.windows(2).any(|w| order(&w[0], &w[1]).is_eq()) {
                    return Err(MemberError::Repeated { path: at.join(".") });
                }
            }
            Value::Nil => {}
            _ => return Err(MemberError::NotAList { path: at.join(".") }),
        }
    }
    if inner.is_empty() {
        return Ok(());
    }
    match value {
        Value::Map(fields) => {
            for (key, value) in fields {
                let next = through(&inner, key);
                if let (false, Some(key)) = (next.is_empty(), key.as_str()) {
                    at.push(key);
                    check(value, &next, at)?;
                    at.pop();
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                check(item, &inner, at)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests;
