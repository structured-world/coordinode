use std::sync::Arc;

// no-std: spin::RwLock (drop-in).
use parking_lot::RwLock;

use tonic::{Request, Response, Status};

use coordinode_core::graph::types::{Value, VectorMetric};
use coordinode_core::schema::computed::{ComputedSpec, DecayFormula, TtlScope};
use coordinode_core::schema::definition::{
    EdgeTypeSchema, LabelSchema, PropertyDef, PropertyType, SchemaMode,
};
use coordinode_embed::Database;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;

use crate::proto::graph;
use crate::services::db_err_to_status;

/// Map proto `PropertyType` integer value to internal `PropertyType`.
///
/// Proto integers:
///   0 = UNSPECIFIED, 1 = INT64, 2 = FLOAT64, 3 = STRING, 4 = BOOL,
///   5 = BYTES, 6 = TIMESTAMP, 7 = VECTOR, 8 = LIST, 9 = MAP
fn proto_type_to_property_type(t: i32) -> PropertyType {
    match t {
        1 => PropertyType::Int,
        2 => PropertyType::Float,
        3 => PropertyType::String,
        4 => PropertyType::Bool,
        5 => PropertyType::Binary,
        6 => PropertyType::Timestamp,
        // VECTOR without explicit dimensions — schema stores structural intent;
        // HNSW index creation (R-API3) will fill dimensions from CREATE VECTOR INDEX.
        7 => PropertyType::Vector {
            dimensions: 0,
            metric: VectorMetric::Cosine,
        },
        8 => PropertyType::Array(Box::new(PropertyType::String)),
        9 => PropertyType::Map,
        _ => PropertyType::String, // UNSPECIFIED → String (most permissive)
    }
}

/// Convert a proto `ComputedPropertyDefinition` to an internal `ComputedSpec`.
///
/// Returns `Err(Status::invalid_argument)` if required fields are missing or
/// the computed_type is UNSPECIFIED.
fn proto_to_computed_spec(def: &graph::ComputedPropertyDefinition) -> Result<ComputedSpec, Status> {
    let computed_type = def.computed_type;
    let formula_type = def.formula_type;

    let formula = match formula_type {
        // 0 = UNSPECIFIED → default to Linear (acceptable for TTL where formula is unused)
        0 | 1 => DecayFormula::Linear,
        2 => DecayFormula::Exponential { lambda: def.lambda },
        3 => DecayFormula::PowerLaw {
            tau: def.tau,
            alpha: def.alpha,
        },
        4 => DecayFormula::Step,
        _ => {
            return Err(Status::invalid_argument(format!(
                "unknown decay_formula_type: {formula_type}"
            )));
        }
    };

    let anchor = def.anchor_field.clone();
    if anchor.is_empty() {
        return Err(Status::invalid_argument(
            "computed_property: anchor_field must not be empty",
        ));
    }
    if def.duration_secs == 0 {
        return Err(Status::invalid_argument(
            "computed_property: duration_secs must be > 0",
        ));
    }

    match computed_type {
        1 => {
            // TTL
            let scope = match def.scope {
                0 | 3 => TtlScope::Node, // UNSPECIFIED defaults to Node
                1 => TtlScope::Field,
                2 => TtlScope::Subtree,
                _ => {
                    return Err(Status::invalid_argument(format!(
                        "unknown ttl_scope: {}",
                        def.scope
                    )));
                }
            };
            let target_field = if def.target_field.is_empty() {
                None
            } else {
                Some(def.target_field.clone())
            };
            Ok(ComputedSpec::Ttl {
                duration_secs: def.duration_secs,
                anchor_field: anchor,
                scope,
                target_field,
            })
        }
        2 => {
            // Decay
            Ok(ComputedSpec::Decay {
                formula,
                initial: def.initial,
                target: def.target,
                duration_secs: def.duration_secs,
                anchor_field: anchor,
            })
        }
        3 => {
            // VectorDecay
            Ok(ComputedSpec::VectorDecay {
                formula,
                duration_secs: def.duration_secs,
                anchor_field: anchor,
            })
        }
        0 => Err(Status::invalid_argument(
            "computed_property: computed_type must not be UNSPECIFIED",
        )),
        _ => Err(Status::invalid_argument(format!(
            "unknown computed_type: {computed_type}"
        ))),
    }
}

/// Convert an internal `ComputedSpec` to a proto `ComputedPropertyDefinition`.
fn computed_spec_to_proto(name: &str, spec: &ComputedSpec) -> graph::ComputedPropertyDefinition {
    let mut def = graph::ComputedPropertyDefinition {
        name: name.to_string(),
        ..Default::default()
    };

    match spec {
        ComputedSpec::Ttl {
            duration_secs,
            anchor_field,
            scope,
            target_field,
        } => {
            def.computed_type = 1; // TTL
            def.duration_secs = *duration_secs;
            def.anchor_field = anchor_field.clone();
            def.scope = match scope {
                TtlScope::Field => 1,
                TtlScope::Subtree => 2,
                TtlScope::Node => 3,
            };
            def.target_field = target_field.clone().unwrap_or_default();
        }
        ComputedSpec::Decay {
            formula,
            initial,
            target,
            duration_secs,
            anchor_field,
        } => {
            def.computed_type = 2; // Decay
            def.duration_secs = *duration_secs;
            def.anchor_field = anchor_field.clone();
            def.initial = *initial;
            def.target = *target;
            set_formula_fields(&mut def, formula);
        }
        ComputedSpec::VectorDecay {
            formula,
            duration_secs,
            anchor_field,
        } => {
            def.computed_type = 3; // VectorDecay
            def.duration_secs = *duration_secs;
            def.anchor_field = anchor_field.clone();
            set_formula_fields(&mut def, formula);
        }
    }

    def
}

fn set_formula_fields(def: &mut graph::ComputedPropertyDefinition, formula: &DecayFormula) {
    match formula {
        DecayFormula::Linear => {
            def.formula_type = 1;
        }
        DecayFormula::Exponential { lambda } => {
            def.formula_type = 2;
            def.lambda = *lambda;
        }
        DecayFormula::PowerLaw { tau, alpha } => {
            def.formula_type = 3;
            def.tau = *tau;
            def.alpha = *alpha;
        }
        DecayFormula::Step => {
            def.formula_type = 4;
        }
    }
}

/// Convert a proto `SchemaMode` integer to the internal `SchemaMode`.
///
/// Proto values:
///   0 = UNSPECIFIED → defaults to STRICT (most type-safe)
///   1 = STRICT, 2 = VALIDATED, 3 = FLEXIBLE
fn proto_to_schema_mode(v: i32) -> SchemaMode {
    match v {
        2 => SchemaMode::Validated,
        3 => SchemaMode::Flexible,
        _ => SchemaMode::Strict, // UNSPECIFIED (0) and STRICT (1) both → Strict
    }
}

/// Convert an internal `SchemaMode` to a proto integer.
fn schema_mode_to_proto(mode: SchemaMode) -> i32 {
    match mode {
        SchemaMode::Strict => 1,
        SchemaMode::Validated => 2,
        SchemaMode::Flexible => 3,
    }
}

/// Map internal `PropertyType` to proto integer for list_labels responses.
fn property_type_to_proto(pt: &PropertyType) -> i32 {
    match pt {
        PropertyType::Int => 1,
        PropertyType::Float => 2,
        PropertyType::String => 3,
        PropertyType::Bool => 4,
        PropertyType::Binary | PropertyType::Blob => 5,
        PropertyType::Timestamp => 6,
        PropertyType::Vector { .. } => 7,
        PropertyType::Array(_) => 8,
        PropertyType::Map => 9,
        // Document, Geo, Computed have no direct proto representation → MAP as closest.
        _ => 9,
    }
}

pub struct SchemaServiceImpl {
    database: Arc<RwLock<Database>>,
}

impl SchemaServiceImpl {
    pub fn new(database: Arc<RwLock<Database>>) -> Self {
        Self { database }
    }
}

#[tonic::async_trait]
impl graph::schema_service_server::SchemaService for SchemaServiceImpl {
    async fn create_label(
        &self,
        request: Request<graph::CreateLabelRequest>,
    ) -> Result<Response<graph::Label>, Status> {
        let req = request.into_inner();
        // Build internal LabelSchema from the proto request.
        let mut schema = LabelSchema::new_node_id(&req.name);

        // Apply schema mode (defaults to STRICT when unspecified).
        let mode = proto_to_schema_mode(req.schema_mode);
        schema.set_mode(mode);

        for prop_def in &req.properties {
            let property_type = proto_type_to_property_type(prop_def.r#type);
            let mut prop = PropertyDef::new(&prop_def.name, property_type);
            if prop_def.required {
                prop = prop.not_null();
            }
            if prop_def.unique {
                prop = prop.unique();
            }
            schema.add_property(prop);
        }

        // Wire COMPUTED property specs into the schema.
        // TtlReaper picks up TTL specs automatically; DECAY/VECTOR_DECAY are
        // evaluated inline at query time during MATCH/RETURN execution.
        let mut echo_computed: Vec<graph::ComputedPropertyDefinition> =
            Vec::with_capacity(req.computed_properties.len());
        for cp in &req.computed_properties {
            let spec = proto_to_computed_spec(cp)?;
            schema.add_property(PropertyDef::computed(&cp.name, spec));
            echo_computed.push(cp.clone());
        }

        let schema_revision = {
            let mut db = self.database.write();
            db.create_label_schema(schema)
                .map_err(|e| db_err_to_status("create_label", e))?
        };

        Ok(Response::new(graph::Label {
            name: req.name,
            properties: req.properties,
            schema_revision,
            computed_properties: echo_computed,
            schema_mode: schema_mode_to_proto(mode),
        }))
    }

    async fn create_edge_type(
        &self,
        request: Request<graph::CreateEdgeTypeRequest>,
    ) -> Result<Response<graph::EdgeType>, Status> {
        let req = request.into_inner();

        // Build internal EdgeTypeSchema from the proto request.
        let mut schema = EdgeTypeSchema::new(&req.name);
        for prop_def in &req.properties {
            let property_type = proto_type_to_property_type(prop_def.r#type);
            let mut prop = PropertyDef::new(&prop_def.name, property_type);
            if prop_def.required {
                prop = prop.not_null();
            }
            if prop_def.unique {
                prop = prop.unique();
            }
            schema.add_property(prop);
        }

        let schema_revision = {
            let mut db = self.database.write();
            db.create_edge_type_schema(schema)
                .map_err(|e| db_err_to_status("create_edge_type", e))?
        };

        Ok(Response::new(graph::EdgeType {
            name: req.name,
            properties: req.properties,
            schema_revision,
        }))
    }

    async fn list_labels(
        &self,
        _request: Request<graph::ListLabelsRequest>,
    ) -> Result<Response<graph::ListLabelsResponse>, Status> {
        // Two-pass: first load declared schemas (with property metadata),
        // then add any undeclared labels discovered from existing nodes.
        const SCHEMA_PREFIX: &[u8] = b"schema:label:";

        let mut db = self.database.write();

        // Pass 1: scan `schema:label:*` for persisted LabelSchema entries.
        let mut label_map: std::collections::BTreeMap<String, graph::Label> = {
            let iter = db
                .engine()
                .prefix_scan(Partition::Schema, SCHEMA_PREFIX)
                .map_err(|e| Status::internal(format!("list_labels scan error: {e}")))?;

            let mut map = std::collections::BTreeMap::new();
            for guard in iter {
                let Ok((key, val_bytes)) = guard.into_inner() else {
                    continue;
                };
                let Ok(name) = std::str::from_utf8(&key[SCHEMA_PREFIX.len()..]) else {
                    continue;
                };
                if name.is_empty() {
                    continue;
                }
                // Decode the persisted LabelSchema and convert properties.
                if let Ok(schema) =
                    coordinode_core::schema::definition::LabelSchema::from_msgpack(&val_bytes)
                {
                    let mut properties = Vec::new();
                    let mut computed_properties = Vec::new();

                    for p in schema.properties.values() {
                        if let PropertyType::Computed(ref spec) = p.property_type {
                            computed_properties.push(computed_spec_to_proto(&p.name, spec));
                        } else {
                            properties.push(graph::PropertyDefinition {
                                name: p.name.clone(),
                                r#type: property_type_to_proto(&p.property_type),
                                required: p.not_null,
                                unique: p.unique,
                            });
                        }
                    }

                    map.insert(
                        schema.name.clone(),
                        graph::Label {
                            name: schema.name.clone(),
                            properties,
                            schema_revision: schema.schema_revision,
                            computed_properties,
                            schema_mode: schema_mode_to_proto(schema.mode),
                        },
                    );
                } else {
                    // Unreadable schema entry — still expose the name.
                    map.entry(name.to_string()).or_insert_with(|| graph::Label {
                        name: name.to_string(),
                        properties: vec![],
                        schema_revision: 0,
                        computed_properties: vec![],
                        schema_mode: schema_mode_to_proto(SchemaMode::Strict),
                    });
                }
            }
            map
        };

        // Pass 2: discover undeclared labels from existing nodes via Cypher.
        // Note: `label` is a Cypher reserved keyword — use `lbl` as alias.
        let rows = db
            .execute_cypher("MATCH (n) RETURN DISTINCT n.__label__ AS lbl ORDER BY lbl")
            .map_err(|e| db_err_to_status("list_labels cypher", e))?;

        for row in rows {
            if let Some(Value::String(name)) = row.get("lbl") {
                if !name.is_empty() {
                    label_map
                        .entry(name.clone())
                        .or_insert_with(|| graph::Label {
                            name: name.clone(),
                            properties: vec![],
                            schema_revision: 0,
                            computed_properties: vec![],
                            schema_mode: 0, // UNSPECIFIED — no declared schema
                        });
                }
            }
        }

        let mut labels: Vec<graph::Label> = label_map.into_values().collect();
        labels.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(Response::new(graph::ListLabelsResponse { labels }))
    }

    async fn list_edge_types(
        &self,
        _request: Request<graph::ListEdgeTypesRequest>,
    ) -> Result<Response<graph::ListEdgeTypesResponse>, Status> {
        // Edge types are registered in the Schema partition under the key prefix
        // `schema:edge_type:<name>` whenever an edge is created. Read directly
        // from there — wildcard MATCH ()-[r]->() won't work because the executor
        // only traverses explicitly-typed edges (empty edge_types slice = no-op).
        // Versioned schema keys are `schema:edge_type:<name>:<version>`.
        // Strip the trailing `:<version>` suffix and dedup by name.
        const PREFIX: &[u8] = b"schema:edge_type:";
        let names: Vec<String> = {
            let db = self.database.write();
            let iter = db
                .engine()
                .prefix_scan(Partition::Schema, PREFIX)
                .map_err(|e| Status::internal(format!("list_edge_types scan error: {e}")))?;

            let mut types: Vec<String> = Vec::new();
            for guard in iter {
                if let Ok((key, _)) = guard.into_inner() {
                    let suffix = match std::str::from_utf8(&key[PREFIX.len()..]) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    let name = match suffix.rsplit_once(':') {
                        Some((name, _version)) if !name.is_empty() => name.to_string(),
                        _ => continue,
                    };
                    if !types.contains(&name) {
                        types.push(name);
                    }
                }
            }
            types.sort();
            types
        };

        let edge_types: Vec<graph::EdgeType> = names
            .into_iter()
            .map(|name| graph::EdgeType {
                name,
                properties: vec![],
                schema_revision: 0,
            })
            .collect();

        Ok(Response::new(graph::ListEdgeTypesResponse { edge_types }))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
