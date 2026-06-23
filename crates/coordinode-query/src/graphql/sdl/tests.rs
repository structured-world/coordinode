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
    assert!(sdl.contains("createFollows(from: ID!, to: ID!, properties: FollowsPropertiesInput)"));
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
