//! GraphQL SDL (Schema Definition Language) generator.
//!
//! Produces a GraphQL schema string from CoordiNode's graph schema definitions.
//! Schema is regenerated whenever the graph schema changes (label added, etc.).
//!
//! # Cluster-ready notes
//! - SDL generation is a pure function (schema in → SDL out).
//! - Each CE node generates the same SDL from replicated schema data.

use coordinode_core::schema::definition::{EdgeTypeSchema, LabelSchema, PropertyType};

/// Generate a complete GraphQL SDL from graph schema definitions.
///
/// Produces: types, connections, filters, queries, mutations.
pub fn generate_graphql_sdl(labels: &[&LabelSchema], edge_types: &[&EdgeTypeSchema]) -> String {
    let mut sdl = String::new();

    // Scalar types
    sdl.push_str("scalar DateTime\nscalar JSON\n\n");

    // PageInfo (Relay spec)
    sdl.push_str(
        "type PageInfo {\n  hasNextPage: Boolean!\n  hasPreviousPage: Boolean!\n  startCursor: String\n  endCursor: String\n}\n\n",
    );

    // Vector metric enum
    sdl.push_str("enum VectorMetric {\n  COSINE\n  L2\n  DOT\n  L1\n}\n\n");

    // Change operation enum (for subscriptions)
    sdl.push_str("enum ChangeOperation {\n  CREATED\n  UPDATED\n  DELETED\n}\n\n");

    // Generate type + filter + connection for each label
    for label in labels {
        sdl.push_str(&generate_type(label, edge_types));
        sdl.push_str(&generate_filter_type(label));
        sdl.push_str(&generate_order_type(label));
        sdl.push_str(&generate_connection_type(label));
        sdl.push_str(&generate_input_types(label));
    }

    // Generate unique input types for labels with unique properties
    for label in labels {
        let unique_input = generate_unique_input(label);
        if !unique_input.is_empty() {
            sdl.push_str(&unique_input);
        }
    }

    // Generate edge connection types and edge input types
    for et in edge_types {
        if !et.properties.is_empty() {
            sdl.push_str(&generate_edge_type_sdl(et));
            sdl.push_str(&generate_edge_input_type(et));
        }
    }

    // Query type
    sdl.push_str(&generate_query_type(labels));

    // Mutation type
    sdl.push_str(&generate_mutation_type(labels, edge_types));

    // Subscription type (WebSocket subscriptions)
    sdl.push_str(&generate_subscription_type(labels, edge_types));

    // Change event types for subscriptions
    for label in labels {
        sdl.push_str(&generate_change_event_type(label));
    }

    sdl
}

/// Generate a GraphQL object type from a label schema.
fn generate_type(label: &LabelSchema, edge_types: &[&EdgeTypeSchema]) -> String {
    let name = &label.name;
    let mut fields = vec!["  id: ID!".to_string()];

    for (prop_name, prop_def) in &label.properties {
        let gql_type = property_type_to_graphql(&prop_def.property_type, prop_def.not_null);
        // Skip vector fields in type definition (search-only)
        if matches!(prop_def.property_type, PropertyType::Vector { .. }) {
            continue;
        }
        fields.push(format!("  {prop_name}: {gql_type}"));
    }

    // Add relationship fields from edge types
    for et in edge_types {
        // Forward: this label has outgoing edges of this type
        let field_name = to_camel_case(&et.name);
        fields.push(format!("  {field_name}: [{name}!]!",));
    }

    format!("type {name} {{\n{}\n}}\n\n", fields.join("\n"))
}

/// Generate filter input type for a label.
fn generate_filter_type(label: &LabelSchema) -> String {
    let name = &label.name;
    let mut fields = Vec::new();

    for (prop_name, prop_def) in &label.properties {
        if matches!(prop_def.property_type, PropertyType::Vector { .. }) {
            continue;
        }
        let filter_type = property_type_to_filter(&prop_def.property_type);
        fields.push(format!("  {prop_name}: {filter_type}"));
    }

    fields.push(format!("  AND: [{name}Filter!]"));
    fields.push(format!("  OR: [{name}Filter!]"));
    fields.push(format!("  NOT: {name}Filter"));

    format!("input {name}Filter {{\n{}\n}}\n\n", fields.join("\n"))
}

/// Generate order-by input type.
fn generate_order_type(label: &LabelSchema) -> String {
    let name = &label.name;
    let mut fields = Vec::new();

    for (prop_name, prop_def) in &label.properties {
        if matches!(prop_def.property_type, PropertyType::Vector { .. }) {
            continue;
        }
        fields.push(format!("  {prop_name}: SortDirection"));
    }

    let mut sdl = format!("input {name}OrderBy {{\n{}\n}}\n\n", fields.join("\n"));

    // SortDirection enum (generated once, but safe to repeat)
    sdl.push_str("enum SortDirection {\n  ASC\n  DESC\n}\n\n");

    sdl
}

/// Generate Relay connection type.
fn generate_connection_type(label: &LabelSchema) -> String {
    let name = &label.name;
    format!(
        "type {name}Connection {{\n  edges: [{name}Edge!]!\n  pageInfo: PageInfo!\n  totalCount: Int\n}}\n\n\
         type {name}Edge {{\n  node: {name}!\n  cursor: String!\n}}\n\n"
    )
}

/// Generate create/update input types.
fn generate_input_types(label: &LabelSchema) -> String {
    let name = &label.name;
    let mut create_fields = Vec::new();
    let mut update_fields = Vec::new();

    for (prop_name, prop_def) in &label.properties {
        if matches!(prop_def.property_type, PropertyType::Vector { .. }) {
            continue;
        }
        let gql_type = property_type_to_graphql(&prop_def.property_type, false);
        let required = if prop_def.not_null {
            format!("{gql_type}!")
        } else {
            gql_type.clone()
        };
        create_fields.push(format!("  {prop_name}: {required}"));
        update_fields.push(format!("  {prop_name}: {gql_type}"));
    }

    format!(
        "input Create{name}Input {{\n{}\n}}\n\n\
         input Update{name}Input {{\n{}\n}}\n\n",
        create_fields.join("\n"),
        update_fields.join("\n"),
    )
}

/// Generate unique input type for upsert WHERE clause.
/// Only includes properties marked as unique.
fn generate_unique_input(label: &LabelSchema) -> String {
    let name = &label.name;
    let unique_props: Vec<(&String, &coordinode_core::schema::definition::PropertyDef)> = label
        .properties
        .iter()
        .filter(|(_, pd)| pd.unique)
        .collect();

    if unique_props.is_empty() {
        return String::new();
    }

    let fields: Vec<String> = unique_props
        .iter()
        .map(|(prop_name, prop_def)| {
            let gql_type = property_type_to_graphql(&prop_def.property_type, false);
            format!("  {prop_name}: {gql_type}")
        })
        .collect();

    format!("input {name}UniqueInput {{\n{}\n}}\n\n", fields.join("\n"))
}

/// Generate edge property input type for edge mutations with properties.
fn generate_edge_input_type(et: &EdgeTypeSchema) -> String {
    let type_name = to_pascal_case(&et.name);
    let fields: Vec<String> = et
        .properties
        .iter()
        .map(|(prop_name, prop_def)| {
            let gql_type = property_type_to_graphql(&prop_def.property_type, false);
            format!("  {prop_name}: {gql_type}")
        })
        .collect();

    format!(
        "input {type_name}PropertiesInput {{\n{}\n}}\n\n",
        fields.join("\n")
    )
}

/// Generate edge type SDL for edges with properties.
fn generate_edge_type_sdl(et: &EdgeTypeSchema) -> String {
    let name = &et.name;
    let type_name = to_pascal_case(name);
    let mut fields = Vec::new();

    for (prop_name, prop_def) in &et.properties {
        let gql_type = property_type_to_graphql(&prop_def.property_type, prop_def.not_null);
        fields.push(format!("  {prop_name}: {gql_type}"));
    }

    format!("type {type_name}Edge {{\n{}\n}}\n\n", fields.join("\n"))
}

/// Generate the Query type.
fn generate_query_type(labels: &[&LabelSchema]) -> String {
    let mut fields = Vec::new();

    for label in labels {
        let name = &label.name;
        let lower = to_camel_case(name);
        let lower_plural = format!("{lower}s");

        // Single by ID
        fields.push(format!("  {lower}(id: ID!): {name}"));

        // Filtered list with pagination
        fields.push(format!(
            "  {lower_plural}(filter: {name}Filter, orderBy: {name}OrderBy, first: Int, after: String): {name}Connection!"
        ));
    }

    format!("type Query {{\n{}\n}}\n\n", fields.join("\n"))
}

/// Generate the Mutation type.
fn generate_mutation_type(labels: &[&LabelSchema], edge_types: &[&EdgeTypeSchema]) -> String {
    let mut fields = Vec::new();

    for label in labels {
        let name = &label.name;

        // CRUD mutations
        fields.push(format!(
            "  create{name}(input: Create{name}Input!): {name}!"
        ));
        fields.push(format!(
            "  update{name}(id: ID!, input: Update{name}Input!): {name}!"
        ));
        fields.push(format!("  delete{name}(id: ID!): Boolean!"));

        // Upsert mutation (only if label has unique properties)
        let has_unique = label.properties.values().any(|pd| pd.unique);
        if has_unique {
            fields.push(format!(
                "  upsert{name}(where: {name}UniqueInput!, onCreate: Create{name}Input!, onMatch: Update{name}Input!): {name}!"
            ));
        }
    }

    for et in edge_types {
        let name = to_pascal_case(&et.name);
        let has_props = !et.properties.is_empty();

        if has_props {
            fields.push(format!(
                "  create{name}(from: ID!, to: ID!, properties: {name}PropertiesInput): Boolean!"
            ));
        } else {
            fields.push(format!("  create{name}(from: ID!, to: ID!): Boolean!"));
        }
        fields.push(format!("  delete{name}(from: ID!, to: ID!): Boolean!"));
    }

    format!("type Mutation {{\n{}\n}}\n\n", fields.join("\n"))
}

/// Generate the Subscription type (WebSocket subscriptions).
///
/// CE: WebSocket transport (`graphql-transport-ws` protocol) on port 7083.
/// WAL-based change detection: mutations trigger subscription notifications.
///
/// # Cluster-ready notes
/// - Subscriptions are per-node (client connects to one node).
/// - In CE 3-node HA: local WAL events trigger local subscriptions.
/// - Cross-node subscription propagation via CDC/NATS in Phase 2+.
fn generate_subscription_type(labels: &[&LabelSchema], edge_types: &[&EdgeTypeSchema]) -> String {
    let mut fields = Vec::new();

    for label in labels {
        let name = &label.name;
        let lower = to_camel_case(name);

        // Subscribe to changes on a specific node
        fields.push(format!("  {lower}Changed(id: ID!): {name}ChangeEvent!"));

        // Subscribe to new nodes of a label
        fields.push(format!("  {lower}Created: {name}!"));

        // Subscribe to deleted nodes
        fields.push(format!("  {lower}Deleted: ID!"));
    }

    for et in edge_types {
        let lower = to_camel_case(&et.name);

        // Subscribe to edge changes
        fields.push(format!("  {lower}Created(nodeId: ID): Boolean!"));
        fields.push(format!("  {lower}Deleted(nodeId: ID): Boolean!"));
    }

    if fields.is_empty() {
        return String::new();
    }

    format!("type Subscription {{\n{}\n}}\n\n", fields.join("\n"))
}

/// Generate change event type for a label (used by subscriptions).
fn generate_change_event_type(label: &LabelSchema) -> String {
    let name = &label.name;
    format!(
        "type {name}ChangeEvent {{\n\
         \x20 operation: ChangeOperation!\n\
         \x20 node: {name}!\n\
         \x20 changedFields: [String!]\n\
         \x20 timestamp: DateTime!\n\
         }}\n\n"
    )
}

/// Map PropertyType to GraphQL type string.
fn property_type_to_graphql(pt: &PropertyType, required: bool) -> String {
    let base = match pt {
        PropertyType::String => "String",
        PropertyType::Int => "Int",
        PropertyType::Float => "Float",
        PropertyType::Bool => "Boolean",
        PropertyType::Timestamp => "DateTime",
        PropertyType::Vector { .. } => "[Float!]",
        PropertyType::Blob => "String",
        PropertyType::Array(_) => "[JSON]",
        PropertyType::Map => "JSON",
        PropertyType::Geo => "JSON",
        PropertyType::Binary => "String",
        PropertyType::Document => "JSON",
        PropertyType::Computed(_) => "Float", // evaluated at query time, returns numeric
    };
    if required {
        format!("{base}!")
    } else {
        base.to_string()
    }
}

/// Map PropertyType to filter input type.
fn property_type_to_filter(pt: &PropertyType) -> &'static str {
    match pt {
        PropertyType::String => "StringFilter",
        PropertyType::Int => "IntFilter",
        PropertyType::Float => "FloatFilter",
        PropertyType::Bool => "BoolFilter",
        PropertyType::Timestamp => "DateTimeFilter",
        _ => "StringFilter",
    }
}

/// Convert UPPER_SNAKE to camelCase.
fn to_camel_case(s: &str) -> String {
    let lower = s.to_lowercase();
    let mut chars = lower.chars();
    match chars.next() {
        Some(c) => c.to_string() + chars.as_str(),
        None => String::new(),
    }
}

/// Convert UPPER_SNAKE to PascalCase.
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().to_string() + &chars.as_str().to_lowercase(),
                None => String::new(),
            }
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use coordinode_core::schema::definition::PropertyDef;

    fn make_user_schema() -> LabelSchema {
        let mut schema = LabelSchema::new_node_id("User");
        schema.add_property(PropertyDef::new("name", PropertyType::String).not_null());
        schema.add_property(
            PropertyDef::new("email", PropertyType::String)
                .not_null()
                .unique(),
        );
        schema.add_property(PropertyDef::new("age", PropertyType::Int));
        schema
    }

    fn make_movie_schema() -> LabelSchema {
        let mut schema = LabelSchema::new_node_id("Movie");
        schema.add_property(PropertyDef::new("title", PropertyType::String).not_null());
        schema.add_property(PropertyDef::new("rating", PropertyType::Float));
        schema
    }

    fn make_follows_edge() -> EdgeTypeSchema {
        let mut et = EdgeTypeSchema::new("FOLLOWS");
        et.add_property(PropertyDef::new("since", PropertyType::Timestamp));
        et
    }

    #[test]
    fn generate_basic_sdl() {
        let user = make_user_schema();
        let movie = make_movie_schema();
        let follows = make_follows_edge();

        let sdl = generate_graphql_sdl(&[&user, &movie], &[&follows]);

        // Should contain type definitions
        assert!(sdl.contains("type User {"), "missing User type");
        assert!(sdl.contains("type Movie {"), "missing Movie type");
        assert!(sdl.contains("id: ID!"), "missing id field");
        assert!(sdl.contains("name: String!"), "missing name field");
        assert!(sdl.contains("age: Int"), "missing age field");

        // Should contain query type
        assert!(sdl.contains("type Query {"), "missing Query type");
        assert!(sdl.contains("user(id: ID!): User"), "missing user query");
        assert!(sdl.contains("users("), "missing users query");

        // Should contain mutation type
        assert!(sdl.contains("type Mutation {"), "missing Mutation type");
        assert!(sdl.contains("createUser("), "missing createUser");
        assert!(sdl.contains("deleteUser("), "missing deleteUser");

        // Should contain filter types
        assert!(sdl.contains("input UserFilter {"), "missing UserFilter");
        assert!(sdl.contains("AND: [UserFilter!]"), "missing AND filter");

        // Should contain connection types
        assert!(sdl.contains("type UserConnection {"), "missing connection");
        assert!(sdl.contains("type UserEdge {"), "missing edge type");
        assert!(sdl.contains("pageInfo: PageInfo!"), "missing pageInfo");

        // Should contain input types
        assert!(
            sdl.contains("input CreateUserInput {"),
            "missing create input"
        );
        assert!(
            sdl.contains("input UpdateUserInput {"),
            "missing update input"
        );

        // Should contain edge type SDL
        assert!(sdl.contains("FollowsEdge"), "missing follows edge type");
    }

    #[test]
    fn generate_empty_schema() {
        let sdl = generate_graphql_sdl(&[], &[]);
        assert!(sdl.contains("scalar DateTime"));
        assert!(sdl.contains("type PageInfo"));
        assert!(sdl.contains("type Query {"));
        assert!(sdl.contains("type Mutation {"));
    }

    #[test]
    fn required_fields_have_bang() {
        let user = make_user_schema();
        let sdl = generate_graphql_sdl(&[&user], &[]);

        // name is NOT NULL → String!
        assert!(sdl.contains("name: String!"));
        // age is nullable → Int (no !)
        assert!(sdl.contains("age: Int\n") || sdl.contains("age: Int\r"));
    }

    #[test]
    fn filter_types_match_properties() {
        let user = make_user_schema();
        let sdl = generate_graphql_sdl(&[&user], &[]);

        assert!(sdl.contains("name: StringFilter"));
        assert!(sdl.contains("age: IntFilter"));
    }

    #[test]
    fn edge_mutations_generated() {
        let user = make_user_schema();
        let follows = make_follows_edge(); // has 'since' property
        let sdl = generate_graphql_sdl(&[&user], &[&follows]);

        // FOLLOWS has properties → createFollows includes properties param
        assert!(
            sdl.contains("createFollows(from: ID!, to: ID!, properties: FollowsPropertiesInput)")
        );
        assert!(sdl.contains("deleteFollows(from: ID!, to: ID!)"));
    }

    #[test]
    fn pascal_case_conversion() {
        assert_eq!(to_pascal_case("FOLLOWS"), "Follows");
        assert_eq!(to_pascal_case("WORKED_AT"), "WorkedAt");
        assert_eq!(to_pascal_case("user"), "User");
    }

    #[test]
    fn camel_case_conversion() {
        assert_eq!(to_camel_case("User"), "user");
        assert_eq!(to_camel_case("FOLLOWS"), "follows");
    }

    #[test]
    fn vector_fields_excluded_from_type() {
        let mut schema = LabelSchema::new_node_id("Document");
        schema.add_property(PropertyDef::new("title", PropertyType::String));
        schema.add_property(PropertyDef::new(
            "embedding",
            PropertyType::Vector {
                dimensions: 384,
                metric: coordinode_core::graph::types::VectorMetric::Cosine,
            },
        ));

        let sdl = generate_graphql_sdl(&[&schema], &[]);

        // type should have title but NOT embedding
        assert!(sdl.contains("title: String"));
        assert!(
            !sdl.contains("embedding:"),
            "vector field should be excluded from type definition"
        );
    }

    #[test]
    fn sdl_is_valid_looking() {
        let user = make_user_schema();
        let movie = make_movie_schema();
        let follows = make_follows_edge();

        let sdl = generate_graphql_sdl(&[&user, &movie], &[&follows]);

        // No empty types
        assert!(!sdl.contains("{\n}"), "empty type body");

        // Balanced braces
        let opens = sdl.chars().filter(|c| *c == '{').count();
        let closes = sdl.chars().filter(|c| *c == '}').count();
        assert_eq!(opens, closes, "unbalanced braces");
    }

    // ====== Mutations auto-gen ======

    #[test]
    fn upsert_mutation_generated_for_unique_props() {
        let user = make_user_schema(); // email is unique
        let sdl = generate_graphql_sdl(&[&user], &[]);

        assert!(
            sdl.contains("upsertUser(where: UserUniqueInput!"),
            "missing upsert mutation"
        );
        assert!(
            sdl.contains("onCreate: CreateUserInput!"),
            "missing onCreate in upsert"
        );
        assert!(
            sdl.contains("onMatch: UpdateUserInput!"),
            "missing onMatch in upsert"
        );
    }

    #[test]
    fn unique_input_contains_unique_props_only() {
        let user = make_user_schema(); // email is unique, name/age are not
        let sdl = generate_graphql_sdl(&[&user], &[]);

        assert!(
            sdl.contains("input UserUniqueInput {"),
            "missing UniqueInput type"
        );
        // email is unique → should be in UniqueInput
        assert!(sdl.contains("email: String"), "missing unique field email");
    }

    #[test]
    fn no_upsert_without_unique_props() {
        let movie = make_movie_schema(); // no unique properties
        let sdl = generate_graphql_sdl(&[&movie], &[]);

        assert!(
            !sdl.contains("upsertMovie"),
            "upsert should not exist without unique props"
        );
        assert!(
            !sdl.contains("MovieUniqueInput"),
            "UniqueInput should not exist without unique props"
        );
    }

    #[test]
    fn edge_mutation_with_properties() {
        let user = make_user_schema();
        let follows = make_follows_edge(); // has 'since' property

        let sdl = generate_graphql_sdl(&[&user], &[&follows]);

        assert!(
            sdl.contains("createFollows(from: ID!, to: ID!, properties: FollowsPropertiesInput)"),
            "edge mutation should accept properties input"
        );
        assert!(
            sdl.contains("input FollowsPropertiesInput {"),
            "missing edge properties input type"
        );
        assert!(
            sdl.contains("since: DateTime"),
            "missing edge property in input"
        );
    }

    #[test]
    fn edge_mutation_without_properties() {
        let user = make_user_schema();
        let simple_edge = EdgeTypeSchema::new("BLOCKS");
        // No properties

        let sdl = generate_graphql_sdl(&[&user], &[&simple_edge]);

        // Simple edge mutation without properties argument
        assert!(
            sdl.contains("createBlocks(from: ID!, to: ID!): Boolean!"),
            "simple edge mutation should not have properties param"
        );
        assert!(
            !sdl.contains("BlocksPropertiesInput"),
            "no input type for propertyless edge"
        );
    }

    // ====== Subscription types ======

    #[test]
    fn subscription_type_generated() {
        let user = make_user_schema();
        let follows = make_follows_edge();
        let sdl = generate_graphql_sdl(&[&user], &[&follows]);

        assert!(
            sdl.contains("type Subscription {"),
            "missing Subscription type"
        );
        assert!(
            sdl.contains("userChanged(id: ID!): UserChangeEvent!"),
            "missing userChanged"
        );
        assert!(sdl.contains("userCreated: User!"), "missing userCreated");
        assert!(sdl.contains("userDeleted: ID!"), "missing userDeleted");
        assert!(
            sdl.contains("followsCreated(nodeId: ID): Boolean!"),
            "missing followsCreated"
        );
    }

    #[test]
    fn change_event_type_generated() {
        let user = make_user_schema();
        let sdl = generate_graphql_sdl(&[&user], &[]);

        assert!(
            sdl.contains("type UserChangeEvent {"),
            "missing ChangeEvent type"
        );
        assert!(
            sdl.contains("operation: ChangeOperation!"),
            "missing operation field"
        );
        assert!(sdl.contains("node: User!"), "missing node field");
        assert!(
            sdl.contains("changedFields: [String!]"),
            "missing changedFields"
        );
        assert!(sdl.contains("timestamp: DateTime!"), "missing timestamp");
    }

    #[test]
    fn empty_schema_no_subscription() {
        let sdl = generate_graphql_sdl(&[], &[]);
        assert!(
            !sdl.contains("type Subscription"),
            "should not have Subscription for empty schema"
        );
    }
}
