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
mod tests;
