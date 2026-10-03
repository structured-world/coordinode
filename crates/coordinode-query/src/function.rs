//! The built-in function catalog.
//!
//! One table defines every function the evaluator runs: its names, its
//! signature, its category and what it does. Name resolution, scalar and
//! aggregate dispatch, and the `dbms.functions()` listing all read that table,
//! so a listed function is always one the evaluator runs and the reverse: the
//! dispatch matches on [`ScalarFn`] / [`AggregateFn`] exhaustively, and those
//! enums exist only as rows of the table.
//!
//! Cypher function names are case-insensitive. A call spelled the way the
//! table spells it resolves with one string match; any other spelling falls
//! back to a case-insensitive scan.

/// One listed name of a built-in function, as `dbms.functions()` reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionEntry {
    /// The name, in the spelling the catalog lists.
    pub name: &'static str,
    /// Parameters and result, e.g. `(input :: STRING) :: STRING`.
    pub params: &'static str,
    /// Neo4j-style category (`String`, `Numeric`, `Aggregating`, ...).
    pub category: &'static str,
    /// What the function returns.
    pub description: &'static str,
    /// Whether it aggregates over rows rather than computing per row.
    pub aggregating: bool,
}

impl FunctionEntry {
    /// The full signature, `name(params) :: result`.
    pub fn signature(&self) -> String {
        format!("{}{}", self.name, self.params)
    }
}

/// Defines a function enum, its exact-spelling resolver, its case-insensitive
/// fallback and its catalog rows from one list, so the four cannot disagree.
macro_rules! function_table {
    (
        $(#[$meta:meta])*
        $vis:vis enum $enum_name:ident, catalog $catalog:ident, aggregating $aggregating:literal {
            $(
                $variant:ident {
                    names: [$($fname:literal),+ $(,)?],
                    params: $params:literal,
                    category: $category:literal,
                    description: $description:literal $(,)?
                }
            ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        $vis enum $enum_name {
            $(
                #[doc = $description]
                $variant,
            )*
        }

        impl $enum_name {
            /// Resolve a called name: the catalog's own spelling first, then
            /// any other casing of it.
            pub fn resolve(name: &str) -> Option<Self> {
                match name {
                    $($($fname)|+ => Some(Self::$variant),)*
                    _ => {
                        const NAMES: &[(&str, $enum_name)] = &[
                            $($(($fname, $enum_name::$variant),)+)*
                        ];
                        NAMES
                            .iter()
                            .find(|(listed, _)| listed.eq_ignore_ascii_case(name))
                            .map(|(_, f)| *f)
                    }
                }
            }
        }

        /// Catalog rows, one per listed name.
        $vis const $catalog: &[FunctionEntry] = &[
            $($(
                FunctionEntry {
                    name: $fname,
                    params: $params,
                    category: $category,
                    description: $description,
                    aggregating: $aggregating,
                },
            )+)*
        ];
    };
}

function_table! {
    /// A built-in function evaluated once per row.
    pub enum ScalarFn, catalog SCALAR_FUNCTIONS, aggregating false {
        Coalesce {
            names: ["coalesce"],
            params: "(input :: ANY, ...) :: ANY",
            category: "Scalar",
            description: "Returns the first non-null argument.",
        },
        ToString {
            names: ["toString"],
            params: "(input :: ANY) :: STRING",
            category: "String",
            description: "Converts an integer, float, boolean or string to a string.",
        },
        ToStringOrNull {
            names: ["toStringOrNull"],
            params: "(input :: ANY) :: STRING",
            category: "String",
            description: "Converts a value to a string, or returns null when it cannot be converted.",
        },
        ToStringList {
            names: ["toStringList"],
            params: "(input :: LIST<ANY>) :: LIST<STRING>",
            category: "List",
            description: "Converts each element of a list to a string; unconvertible elements become null.",
        },
        Size {
            names: ["size"],
            params: "(input :: STRING | LIST<ANY>) :: INTEGER",
            category: "Scalar",
            description: "Returns the number of elements in a list or of characters in a string.",
        },
        Length {
            names: ["length"],
            params: "(input :: PATH) :: INTEGER",
            category: "Scalar",
            description: "Returns the number of relationships in a path.",
        },
        Nodes {
            names: ["nodes"],
            params: "(input :: PATH) :: LIST<INTEGER>",
            category: "List",
            description: "Returns the ids of the nodes along a path, in order.",
        },
        Relationships {
            names: ["relationships"],
            params: "(input :: PATH) :: LIST<MAP>",
            category: "List",
            description: "Returns the relationships along a path, each as a map of type, source and target.",
        },
        Type {
            names: ["type"],
            params: "(input :: RELATIONSHIP) :: STRING",
            category: "Scalar",
            description: "Returns the type of a relationship.",
        },
        ElementId {
            names: ["elementId"],
            params: "(input :: NODE) :: STRING",
            category: "Scalar",
            description: "Returns the element id of a node.",
        },
        Id {
            names: ["id"],
            params: "(input :: NODE) :: INTEGER",
            category: "Scalar",
            description: "Returns the numeric id of a node.",
        },
        Labels {
            names: ["labels"],
            params: "(input :: NODE) :: LIST<STRING>",
            category: "List",
            description: "Returns the labels of a node.",
        },
        StartNode {
            names: ["startNode"],
            params: "(input :: RELATIONSHIP) :: INTEGER",
            category: "Scalar",
            description: "Returns the id of the node a relationship starts at.",
        },
        EndNode {
            names: ["endNode"],
            params: "(input :: RELATIONSHIP) :: INTEGER",
            category: "Scalar",
            description: "Returns the id of the node a relationship ends at.",
        },
        Properties {
            names: ["properties"],
            params: "(input :: NODE | RELATIONSHIP) :: MAP",
            category: "Scalar",
            description: "Returns the properties of a node or relationship as a map.",
        },
        Keys {
            names: ["keys"],
            params: "(input :: NODE | RELATIONSHIP | MAP) :: LIST<STRING>",
            category: "List",
            description: "Returns the property keys of a node or relationship, or the keys of a map.",
        },
        NullIf {
            names: ["nullIf"],
            params: "(v1 :: ANY, v2 :: ANY) :: ANY",
            category: "Scalar",
            description: "Returns null when both arguments are equal, otherwise the first.",
        },
        Timestamp {
            names: ["timestamp"],
            params: "() :: INTEGER",
            category: "Scalar",
            description: "Returns the current time in milliseconds since the Unix epoch.",
        },
        Now {
            names: ["now"],
            params: "() :: TIMESTAMP",
            category: "Temporal",
            description: "Returns the current time as a timestamp with microsecond precision.",
        },
        RandomUuid {
            names: ["randomUUID"],
            params: "() :: STRING",
            category: "Scalar",
            description: "Returns a random version 4 UUID.",
        },
        ValueType {
            names: ["valueType"],
            params: "(input :: ANY) :: STRING",
            category: "Scalar",
            description: "Returns the Cypher type name of a value.",
        },
        TemporalActiveAt {
            names: ["temporal_active_at"],
            params: "(relationship :: RELATIONSHIP, at :: INTEGER) :: BOOLEAN",
            category: "Temporal",
            description: "Returns whether a temporal relationship is valid at a time given in microseconds since the Unix epoch.",
        },
        TemporalOverlaps {
            names: ["temporal_overlaps"],
            params: "(relationship :: RELATIONSHIP, from :: INTEGER, to :: INTEGER) :: BOOLEAN",
            category: "Temporal",
            description: "Returns whether a temporal relationship's validity overlaps the half-open interval [from, to).",
        },
        VectorDistance {
            names: ["vector_distance"],
            params: "(a :: VECTOR, b :: VECTOR) :: FLOAT",
            category: "Vector",
            description: "Returns the Euclidean (L2) distance between two vectors.",
        },
        VectorSimilarity {
            names: ["vector_similarity"],
            params: "(a :: VECTOR, b :: VECTOR) :: FLOAT",
            category: "Vector",
            description: "Returns the cosine similarity of two vectors.",
        },
        VectorDot {
            names: ["vector_dot"],
            params: "(a :: VECTOR, b :: VECTOR) :: FLOAT",
            category: "Vector",
            description: "Returns the dot product of two vectors.",
        },
        VectorManhattan {
            names: ["vector_manhattan"],
            params: "(a :: VECTOR, b :: VECTOR) :: FLOAT",
            category: "Vector",
            description: "Returns the Manhattan (L1) distance between two vectors.",
        },
        MaxsimScore {
            names: ["maxsim_score"],
            params: "(document :: LIST<VECTOR>, query :: LIST<VECTOR>) :: FLOAT",
            category: "Vector",
            description: "Returns the late-interaction score: for each query token the best dot product against any document token, summed.",
        },
        TextScore {
            names: ["text_score"],
            params: "(field :: STRING, query :: STRING) :: FLOAT",
            category: "Search",
            description: "Returns the BM25 score of the row's full-text match.",
        },
        TextMatch {
            names: ["text_match"],
            params: "(field :: STRING, query :: STRING, language = null :: STRING) :: BOOLEAN",
            category: "Search",
            description: "Returns whether a field matches a full-text query.",
        },
        HybridScore {
            names: ["hybrid_score"],
            params: "(node :: NODE, query :: ANY, weights = null :: MAP) :: FLOAT",
            category: "Search",
            description: "Blends the row's vector and full-text scores; weights default to vector 0.65 and text 0.35.",
        },
        RrfScore {
            names: ["rrf_score"],
            params: "(methods :: LIST<ANY>, query :: ANY) :: FLOAT",
            category: "Search",
            description: "Returns the row's reciprocal rank fusion score over the listed retrieval methods.",
        },
        DocScore {
            names: ["doc_score"],
            params: "(document :: NODE, query :: ANY, alpha = null :: FLOAT, beta = null :: FLOAT, gamma = null :: FLOAT) :: FLOAT",
            category: "Search",
            description: "Returns the row's document-level aggregate retrieval score.",
        },
        EncryptedMatch {
            names: ["encrypted_match"],
            params: "(field :: ANY, token :: ANY) :: BOOLEAN",
            category: "Search",
            description: "Returns whether an encrypted field matches a search token.",
        },
        Point {
            names: ["point"],
            params: "(input :: MAP) :: POINT",
            category: "Spatial",
            description: "Creates a WGS-84 point from a map with latitude and longitude.",
        },
        PointDistance {
            names: ["point.distance"],
            params: "(from :: POINT, to :: POINT) :: FLOAT",
            category: "Spatial",
            description: "Returns the geodesic distance in metres between two WGS-84 points.",
        },
        Head {
            names: ["head"],
            params: "(list :: LIST<ANY>) :: ANY",
            category: "Scalar",
            description: "Returns the first element of a list.",
        },
        Last {
            names: ["last"],
            params: "(list :: LIST<ANY>) :: ANY",
            category: "Scalar",
            description: "Returns the last element of a list.",
        },
        Tail {
            names: ["tail"],
            params: "(list :: LIST<ANY>) :: LIST<ANY>",
            category: "List",
            description: "Returns a list without its first element.",
        },
        IsEmpty {
            names: ["isEmpty"],
            params: "(input :: LIST<ANY> | STRING | MAP) :: BOOLEAN",
            category: "Predicate",
            description: "Returns whether a list, string or map is empty.",
        },
        Range {
            names: ["range"],
            params: "(start :: INTEGER, end :: INTEGER, step = 1 :: INTEGER) :: LIST<INTEGER>",
            category: "List",
            description: "Returns the integers from start to end inclusive, in steps of step.",
        },
        ToLower {
            names: ["toLower", "lower"],
            params: "(input :: STRING) :: STRING",
            category: "String",
            description: "Returns a string in lowercase.",
        },
        ToUpper {
            names: ["toUpper", "upper"],
            params: "(input :: STRING) :: STRING",
            category: "String",
            description: "Returns a string in uppercase.",
        },
        Trim {
            names: ["trim", "btrim"],
            params: "(input :: STRING) :: STRING",
            category: "String",
            description: "Removes leading and trailing whitespace.",
        },
        LTrim {
            names: ["ltrim"],
            params: "(input :: STRING) :: STRING",
            category: "String",
            description: "Removes leading whitespace.",
        },
        RTrim {
            names: ["rtrim"],
            params: "(input :: STRING) :: STRING",
            category: "String",
            description: "Removes trailing whitespace.",
        },
        Left {
            names: ["left"],
            params: "(original :: STRING, length :: INTEGER) :: STRING",
            category: "String",
            description: "Returns the leftmost length characters of a string.",
        },
        Right {
            names: ["right"],
            params: "(original :: STRING, length :: INTEGER) :: STRING",
            category: "String",
            description: "Returns the rightmost length characters of a string.",
        },
        Substring {
            names: ["substring"],
            params: "(original :: STRING, start :: INTEGER, length = null :: INTEGER) :: STRING",
            category: "String",
            description: "Returns the characters from a zero-based start, to the end or for length characters.",
        },
        Replace {
            names: ["replace"],
            params: "(original :: STRING, search :: STRING, replace :: STRING) :: STRING",
            category: "String",
            description: "Replaces every occurrence of search with replace.",
        },
        Reverse {
            names: ["reverse"],
            params: "(input :: STRING | LIST<ANY>) :: STRING | LIST<ANY>",
            category: "String",
            description: "Reverses the characters of a string or the elements of a list.",
        },
        Split {
            names: ["split"],
            params: "(original :: STRING, splitDelimiters :: STRING | LIST<STRING>) :: LIST<STRING>",
            category: "String",
            description: "Splits a string on any of the given delimiters.",
        },
        CharLength {
            names: ["char_length", "character_length", "charLength"],
            params: "(input :: STRING) :: INTEGER",
            category: "Scalar",
            description: "Returns the number of characters in a string.",
        },
        Normalize {
            names: ["normalize"],
            params: "(input :: STRING, normalForm = 'NFC' :: STRING) :: STRING",
            category: "String",
            description: "Returns a string in a Unicode normal form: NFC, NFD, NFKC or NFKD.",
        },
        Pi {
            names: ["pi"],
            params: "() :: FLOAT",
            category: "Trigonometric",
            description: "Returns the mathematical constant pi.",
        },
        E {
            names: ["e"],
            params: "() :: FLOAT",
            category: "Logarithmic",
            description: "Returns the base of the natural logarithm, e.",
        },
        Rand {
            names: ["rand"],
            params: "() :: FLOAT",
            category: "Numeric",
            description: "Returns a random number in [0, 1).",
        },
        Abs {
            names: ["abs"],
            params: "(input :: INTEGER | FLOAT) :: INTEGER | FLOAT",
            category: "Numeric",
            description: "Returns the absolute value of a number.",
        },
        Ceil {
            names: ["ceil"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Numeric",
            description: "Returns the smallest integral float not less than a number.",
        },
        Floor {
            names: ["floor"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Numeric",
            description: "Returns the largest integral float not greater than a number.",
        },
        Round {
            names: ["round"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Numeric",
            description: "Rounds a number to the nearest integral float, halves away from zero.",
        },
        Sign {
            names: ["sign"],
            params: "(input :: INTEGER | FLOAT) :: INTEGER",
            category: "Numeric",
            description: "Returns -1, 0 or 1 by the sign of a number.",
        },
        IsNaN {
            names: ["isNaN"],
            params: "(input :: INTEGER | FLOAT) :: BOOLEAN",
            category: "Numeric",
            description: "Returns whether a number is NaN.",
        },
        Sqrt {
            names: ["sqrt"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Logarithmic",
            description: "Returns the square root of a number.",
        },
        Exp {
            names: ["exp"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Logarithmic",
            description: "Returns e raised to the power of a number.",
        },
        Log {
            names: ["log"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Logarithmic",
            description: "Returns the natural logarithm of a number.",
        },
        Log10 {
            names: ["log10"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Logarithmic",
            description: "Returns the base-10 logarithm of a number.",
        },
        ToInteger {
            names: ["toInteger", "toIntegerOrNull"],
            params: "(input :: ANY) :: INTEGER",
            category: "Scalar",
            description: "Converts a value to an integer, or returns null when it cannot be converted.",
        },
        ToFloat {
            names: ["toFloat", "toFloatOrNull"],
            params: "(input :: ANY) :: FLOAT",
            category: "Scalar",
            description: "Converts a value to a float, or returns null when it cannot be converted.",
        },
        ToBoolean {
            names: ["toBoolean", "toBooleanOrNull"],
            params: "(input :: ANY) :: BOOLEAN",
            category: "Scalar",
            description: "Converts a value to a boolean, or returns null when it cannot be converted.",
        },
        ToIntegerList {
            names: ["toIntegerList"],
            params: "(input :: LIST<ANY>) :: LIST<INTEGER>",
            category: "List",
            description: "Converts each element of a list to an integer; unconvertible elements become null.",
        },
        ToFloatList {
            names: ["toFloatList"],
            params: "(input :: LIST<ANY>) :: LIST<FLOAT>",
            category: "List",
            description: "Converts each element of a list to a float; unconvertible elements become null.",
        },
        ToBooleanList {
            names: ["toBooleanList"],
            params: "(input :: LIST<ANY>) :: LIST<BOOLEAN>",
            category: "List",
            description: "Converts each element of a list to a boolean; unconvertible elements become null.",
        },
        Sin {
            names: ["sin"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Trigonometric",
            description: "Returns the sine of an angle in radians.",
        },
        Cos {
            names: ["cos"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Trigonometric",
            description: "Returns the cosine of an angle in radians.",
        },
        Tan {
            names: ["tan"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Trigonometric",
            description: "Returns the tangent of an angle in radians.",
        },
        Cot {
            names: ["cot"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Trigonometric",
            description: "Returns the cotangent of an angle in radians.",
        },
        Asin {
            names: ["asin"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Trigonometric",
            description: "Returns the arcsine of a number, in radians.",
        },
        Acos {
            names: ["acos"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Trigonometric",
            description: "Returns the arccosine of a number, in radians.",
        },
        Atan {
            names: ["atan"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Trigonometric",
            description: "Returns the arctangent of a number, in radians.",
        },
        Atan2 {
            names: ["atan2"],
            params: "(y :: FLOAT, x :: FLOAT) :: FLOAT",
            category: "Trigonometric",
            description: "Returns the angle in radians of the point (x, y).",
        },
        Haversin {
            names: ["haversin"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Trigonometric",
            description: "Returns half the versine of an angle in radians.",
        },
        Degrees {
            names: ["degrees"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Trigonometric",
            description: "Converts radians to degrees.",
        },
        Radians {
            names: ["radians"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Trigonometric",
            description: "Converts degrees to radians.",
        },
    }
}

function_table! {
    /// A built-in function aggregating over the rows of a group.
    pub enum AggregateFn, catalog AGGREGATE_FUNCTIONS, aggregating true {
        Count {
            names: ["count"],
            params: "(input :: ANY) :: INTEGER",
            category: "Aggregating",
            description: "Returns the number of non-null values, or of rows for count(*).",
        },
        Sum {
            names: ["sum"],
            params: "(input :: INTEGER | FLOAT) :: INTEGER | FLOAT",
            category: "Aggregating",
            description: "Returns the sum of the numeric values.",
        },
        Avg {
            names: ["avg"],
            params: "(input :: INTEGER | FLOAT) :: FLOAT",
            category: "Aggregating",
            description: "Returns the average of the numeric values.",
        },
        Min {
            names: ["min"],
            params: "(input :: ANY) :: ANY",
            category: "Aggregating",
            description: "Returns the smallest value.",
        },
        Max {
            names: ["max"],
            params: "(input :: ANY) :: ANY",
            category: "Aggregating",
            description: "Returns the largest value.",
        },
        Collect {
            names: ["collect"],
            params: "(input :: ANY) :: LIST<ANY>",
            category: "Aggregating",
            description: "Returns the non-null values as a list.",
        },
        PercentileCont {
            names: ["percentileCont"],
            params: "(input :: FLOAT, percentile :: FLOAT) :: FLOAT",
            category: "Aggregating",
            description: "Returns the percentile of the values, interpolating between the two nearest.",
        },
        PercentileDisc {
            names: ["percentileDisc"],
            params: "(input :: FLOAT, percentile :: FLOAT) :: FLOAT",
            category: "Aggregating",
            description: "Returns the value nearest to the percentile, by the nearest-rank method.",
        },
        StDev {
            names: ["stDev"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Aggregating",
            description: "Returns the sample standard deviation of the values.",
        },
        StDevP {
            names: ["stDevP"],
            params: "(input :: FLOAT) :: FLOAT",
            category: "Aggregating",
            description: "Returns the population standard deviation of the values.",
        },
    }
}

/// Every listed function, scalar and aggregating, in name order.
pub fn catalog() -> Vec<FunctionEntry> {
    let mut all: Vec<FunctionEntry> = SCALAR_FUNCTIONS
        .iter()
        .chain(AGGREGATE_FUNCTIONS)
        .copied()
        .collect();
    all.sort_by(|a, b| a.name.cmp(b.name));
    all
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
