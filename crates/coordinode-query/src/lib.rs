// ADR-041: the query layer reaches storage only through Layer-4 modality
// stores. Naming the physical storage partition enum here leaks a Layer-3
// concern upward; the only allowances are the partition-parameterised
// transaction primitives (which take a partition by argument) and test
// fixtures that plant raw state. See clippy.toml for the disallowed type.
#![deny(clippy::disallowed_types)]

pub mod advisor;
pub mod cypher;
pub mod executor;
pub mod graphql;
pub mod index;
pub mod plan;
pub mod planner;
