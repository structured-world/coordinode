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
use coordinode_storage::Guard;
use coordinode_storage::engine::partition::Partition;

use crate::proto::graph;
use crate::services::db_err_to_status;

/// Map a proto `PropertyType` to the internal `PropertyType`.
///
/// UNSPECIFIED means STRING, the most permissive type. A value the protocol
/// does not define is refused rather than guessed.
fn proto_type_to_property_type(t: i32) -> Result<PropertyType, Status> {
    use graph::PropertyType as P;
    let Ok(t) = P::try_from(t) else {
        return Err(Status::invalid_argument(format!(
            "unknown property type: {t}"
        )));
    };
    Ok(match t {
        P::Unspecified | P::String => PropertyType::String,
        P::Int64 => PropertyType::Int,
        P::Float64 => PropertyType::Float,
        P::Bool => PropertyType::Bool,
        P::Bytes => PropertyType::Binary,
        P::Timestamp => PropertyType::Timestamp,
        // The proto definition carries no dimensions, so the schema records 0
        // ("unset"): writes accept any vector length, and the vector index
        // takes its dimensions from CREATE VECTOR INDEX.
        P::Vector => PropertyType::Vector {
            dimensions: 0,
            metric: VectorMetric::Cosine,
        },
        P::List => PropertyType::Array(Box::new(PropertyType::String)),
        P::Map => PropertyType::Map,
    })
}

/// Convert a proto `ComputedPropertyDefinition` to an internal `ComputedSpec`.
///
/// Returns `Err(Status::invalid_argument)` if required fields are missing or
/// the computed_type is UNSPECIFIED.
fn proto_to_computed_spec(def: &graph::ComputedPropertyDefinition) -> Result<ComputedSpec, Status> {
    use graph::{ComputedType, DecayFormulaType, TtlScopeType};

    let formula = match DecayFormulaType::try_from(def.formula_type) {
        // UNSPECIFIED means Linear: a TTL ignores the formula anyway.
        Ok(DecayFormulaType::Unspecified | DecayFormulaType::Linear) => DecayFormula::Linear,
        Ok(DecayFormulaType::Exponential) => DecayFormula::Exponential { lambda: def.lambda },
        Ok(DecayFormulaType::PowerLaw) => DecayFormula::PowerLaw {
            tau: def.tau,
            alpha: def.alpha,
        },
        Ok(DecayFormulaType::Step) => DecayFormula::Step,
        Err(_) => {
            return Err(Status::invalid_argument(format!(
                "unknown decay_formula_type: {}",
                def.formula_type
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

    match ComputedType::try_from(def.computed_type) {
        Ok(ComputedType::Ttl) => {
            let scope = match TtlScopeType::try_from(def.scope) {
                // UNSPECIFIED means the whole node.
                Ok(TtlScopeType::Unspecified | TtlScopeType::Node) => TtlScope::Node,
                Ok(TtlScopeType::Field) => TtlScope::Field,
                Ok(TtlScopeType::Subtree) => TtlScope::Subtree,
                Err(_) => {
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
        Ok(ComputedType::Decay) => Ok(ComputedSpec::Decay {
            formula,
            initial: def.initial,
            target: def.target,
            duration_secs: def.duration_secs,
            anchor_field: anchor,
        }),
        Ok(ComputedType::VectorDecay) => Ok(ComputedSpec::VectorDecay {
            formula,
            duration_secs: def.duration_secs,
            anchor_field: anchor,
        }),
        Ok(ComputedType::Unspecified) => Err(Status::invalid_argument(
            "computed_property: computed_type must not be UNSPECIFIED",
        )),
        Err(_) => Err(Status::invalid_argument(format!(
            "unknown computed_type: {}",
            def.computed_type
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
            def.computed_type = graph::ComputedType::Ttl as i32;
            def.duration_secs = *duration_secs;
            def.anchor_field = anchor_field.clone();
            def.scope = match scope {
                TtlScope::Field => graph::TtlScopeType::Field,
                TtlScope::Subtree => graph::TtlScopeType::Subtree,
                TtlScope::Node => graph::TtlScopeType::Node,
            } as i32;
            def.target_field = target_field.clone().unwrap_or_default();
        }
        ComputedSpec::Decay {
            formula,
            initial,
            target,
            duration_secs,
            anchor_field,
        } => {
            def.computed_type = graph::ComputedType::Decay as i32;
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
            def.computed_type = graph::ComputedType::VectorDecay as i32;
            def.duration_secs = *duration_secs;
            def.anchor_field = anchor_field.clone();
            set_formula_fields(&mut def, formula);
        }
    }

    def
}

fn set_formula_fields(def: &mut graph::ComputedPropertyDefinition, formula: &DecayFormula) {
    use graph::DecayFormulaType as F;
    def.formula_type = match formula {
        DecayFormula::Linear => F::Linear,
        DecayFormula::Exponential { lambda } => {
            def.lambda = *lambda;
            F::Exponential
        }
        DecayFormula::PowerLaw { tau, alpha } => {
            def.tau = *tau;
            def.alpha = *alpha;
            F::PowerLaw
        }
        DecayFormula::Step => F::Step,
    } as i32;
}

/// Convert a proto `SchemaMode` to the internal `SchemaMode`.
///
/// UNSPECIFIED means STRICT, the most type-safe mode. A value the protocol
/// does not define is refused rather than guessed.
fn proto_to_schema_mode(v: i32) -> Result<SchemaMode, Status> {
    use graph::SchemaMode as M;
    match M::try_from(v) {
        Ok(M::Unspecified | M::Strict) => Ok(SchemaMode::Strict),
        Ok(M::Validated) => Ok(SchemaMode::Validated),
        Ok(M::Flexible) => Ok(SchemaMode::Flexible),
        Err(_) => Err(Status::invalid_argument(format!(
            "unknown schema mode: {v}"
        ))),
    }
}

/// Convert an internal `SchemaMode` to its proto value.
fn schema_mode_to_proto(mode: SchemaMode) -> i32 {
    let proto = match mode {
        SchemaMode::Strict => graph::SchemaMode::Strict,
        SchemaMode::Validated => graph::SchemaMode::Validated,
        SchemaMode::Flexible => graph::SchemaMode::Flexible,
    };
    proto as i32
}

/// Map an internal `PropertyType` to its proto value for list_labels.
fn property_type_to_proto(pt: &PropertyType) -> i32 {
    use graph::PropertyType as P;
    let proto = match pt {
        PropertyType::Int => P::Int64,
        PropertyType::Float => P::Float64,
        PropertyType::String => P::String,
        PropertyType::Bool => P::Bool,
        PropertyType::Binary | PropertyType::Blob => P::Bytes,
        PropertyType::Timestamp => P::Timestamp,
        PropertyType::Vector { .. } => P::Vector,
        PropertyType::Array(_) => P::List,
        // The protocol has no document, geo or computed type; MAP is the
        // closest shape a client can decode.
        PropertyType::Map
        | PropertyType::Document
        | PropertyType::Geo
        | PropertyType::Computed(_) => P::Map,
    };
    proto as i32
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
        let mode = proto_to_schema_mode(req.schema_mode)?;
        schema.set_mode(mode);

        for prop_def in &req.properties {
            let property_type = proto_type_to_property_type(prop_def.r#type)?;
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
            let property_type = proto_type_to_property_type(prop_def.r#type)?;
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
                            // No declared schema.
                            schema_mode: graph::SchemaMode::Unspecified as i32,
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
