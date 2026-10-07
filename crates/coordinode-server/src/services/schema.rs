//! Schema management over gRPC: label and edge type definitions, and the
//! named constraints over them.
//!
//! A definition carries type facts only. Constraints are separate objects
//! that go through the same catalog and admission as `CREATE CONSTRAINT` and
//! `DROP CONSTRAINT` in a query.

use core::time::Duration;
use std::collections::HashSet;
use std::sync::Arc;

// no-std: spin::RwLock (drop-in).
use parking_lot::RwLock;

use tonic::{Request, Response, Status};

use coordinode_core::graph::types::{Value, VectorMetric};
use coordinode_core::schema::computed::{ComputedSpec, DecayFormula, TtlScope};
use coordinode_core::schema::definition::{
    ConstraintKind, ConstraintState, EdgeTypeSchema, LabelSchema, PropertyDef, PropertyType,
    SchemaMode,
};
use coordinode_embed::{ConstraintDeclaration, Database, LabelConstraint};
use coordinode_query::index::definition::PartialFilter;
use coordinode_query::index::{BuildPhase, BuildState, BuildStatus, GenerationId};

use crate::proto::v1::query::DistanceMetric;
use crate::proto::v2::graph as schema;
use crate::services::cypher::{db_error_to_status, proto_to_value_pub, value_to_proto_pub};
use crate::services::error_details::{Reason, catalog_object_status, invalid_field};

/// The largest vector a property may declare.
const MAX_VECTOR_DIMENSIONS: u32 = 65_536;

/// The engine type a wire type names. `field` is the request path of the
/// type, for the refusal.
fn property_type_from_proto(
    field: &str,
    t: Option<&schema::PropertyType>,
) -> Result<PropertyType, Status> {
    use schema::property_type::Type;
    let Some(kind) = t.and_then(|t| t.r#type.as_ref()) else {
        return Err(invalid_field(field, "a property type is required"));
    };
    match kind {
        Type::Scalar(s) => {
            use schema::ScalarType as S;
            let Ok(scalar) = S::try_from(*s) else {
                return Err(invalid_field(field, format!("unknown scalar type {s}")));
            };
            Ok(match scalar {
                S::Unspecified => {
                    return Err(invalid_field(field, "the scalar type is unspecified"));
                }
                S::String => PropertyType::String,
                S::Int64 => PropertyType::Int,
                S::Float64 => PropertyType::Float,
                S::Bool => PropertyType::Bool,
                S::Timestamp => PropertyType::Timestamp,
                S::Blob => PropertyType::Blob,
                S::Binary => PropertyType::Binary,
                S::Map => PropertyType::Map,
                S::Geo => PropertyType::Geo,
                S::Document => PropertyType::Document,
            })
        }
        Type::Vector(v) => {
            if v.dimensions > MAX_VECTOR_DIMENSIONS {
                return Err(invalid_field(
                    format!("{field}.vector.dimensions"),
                    format!("at most {MAX_VECTOR_DIMENSIONS} dimensions"),
                ));
            }
            let Ok(metric) = DistanceMetric::try_from(v.metric) else {
                return Err(invalid_field(
                    format!("{field}.vector.metric"),
                    format!("unknown metric {}", v.metric),
                ));
            };
            let metric = match metric {
                DistanceMetric::Unspecified | DistanceMetric::Cosine => VectorMetric::Cosine,
                DistanceMetric::L2 => VectorMetric::L2,
                DistanceMetric::Dot => VectorMetric::DotProduct,
                DistanceMetric::L1 => VectorMetric::L1,
            };
            Ok(PropertyType::Vector {
                dimensions: v.dimensions,
                metric,
            })
        }
        Type::Array(a) => {
            let element =
                property_type_from_proto(&format!("{field}.array.element"), a.element.as_deref())?;
            Ok(PropertyType::Array(Box::new(element)))
        }
    }
}

/// The wire form of a stored property type. Computed properties are listed
/// apart and never reach here.
fn property_type_to_proto(t: &PropertyType) -> schema::PropertyType {
    use schema::property_type::Type;
    let scalar = |s: schema::ScalarType| Type::Scalar(s as i32);
    let kind = match t {
        PropertyType::String => scalar(schema::ScalarType::String),
        PropertyType::Int => scalar(schema::ScalarType::Int64),
        PropertyType::Float => scalar(schema::ScalarType::Float64),
        PropertyType::Bool => scalar(schema::ScalarType::Bool),
        PropertyType::Timestamp => scalar(schema::ScalarType::Timestamp),
        PropertyType::Blob => scalar(schema::ScalarType::Blob),
        PropertyType::Binary => scalar(schema::ScalarType::Binary),
        PropertyType::Map => scalar(schema::ScalarType::Map),
        PropertyType::Geo => scalar(schema::ScalarType::Geo),
        PropertyType::Document => scalar(schema::ScalarType::Document),
        PropertyType::Vector { dimensions, metric } => Type::Vector(schema::VectorType {
            dimensions: *dimensions,
            metric: match metric {
                VectorMetric::Cosine => DistanceMetric::Cosine,
                VectorMetric::L2 => DistanceMetric::L2,
                VectorMetric::DotProduct => DistanceMetric::Dot,
                VectorMetric::L1 => DistanceMetric::L1,
            } as i32,
        }),
        PropertyType::Array(element) => Type::Array(Box::new(schema::ArrayType {
            element: Some(Box::new(property_type_to_proto(element))),
        })),
        // A computed property is listed apart; never a stored type.
        PropertyType::Computed(_) => scalar(schema::ScalarType::Unspecified),
    };
    schema::PropertyType { r#type: Some(kind) }
}

/// The stored properties a request declares, refusing a nameless or repeated
/// one. `field` is the request path of the list.
fn properties_from_proto(
    field: &str,
    properties: &[schema::PropertyDefinition],
    names: &mut HashSet<String>,
) -> Result<Vec<PropertyDef>, Status> {
    let mut out = Vec::with_capacity(properties.len());
    for (i, p) in properties.iter().enumerate() {
        let at = format!("{field}[{i}]");
        if p.name.is_empty() {
            return Err(invalid_field(format!("{at}.name"), "a name is required"));
        }
        if !names.insert(p.name.clone()) {
            return Err(invalid_field(
                format!("{at}.name"),
                format!("property `{}` is declared twice", p.name),
            ));
        }
        let property_type = property_type_from_proto(&format!("{at}.type"), p.r#type.as_ref())?;
        let mut def = PropertyDef::new(&p.name, property_type);
        if p.required {
            def = def.not_null();
        }
        if let Some(default) = &p.default_value {
            def = def.with_default(proto_to_value_pub(default));
        }
        out.push(def);
    }
    Ok(out)
}

fn property_to_proto(p: &PropertyDef) -> schema::PropertyDefinition {
    schema::PropertyDefinition {
        name: p.name.clone(),
        r#type: Some(property_type_to_proto(&p.property_type)),
        required: p.not_null,
        default_value: p.default.as_ref().map(value_to_proto_pub),
    }
}

/// The engine form of a wire computed property. `field` is its request path.
fn computed_from_proto(
    field: &str,
    def: &schema::ComputedPropertyDefinition,
) -> Result<ComputedSpec, Status> {
    use schema::{ComputedType, DecayFormulaType, TtlScopeType};

    let formula = match DecayFormulaType::try_from(def.formula_type) {
        Ok(DecayFormulaType::Unspecified | DecayFormulaType::Linear) => DecayFormula::Linear,
        Ok(DecayFormulaType::Exponential) => DecayFormula::Exponential { lambda: def.lambda },
        Ok(DecayFormulaType::PowerLaw) => DecayFormula::PowerLaw {
            tau: def.tau,
            alpha: def.alpha,
        },
        Ok(DecayFormulaType::Step) => DecayFormula::Step,
        Err(_) => {
            return Err(invalid_field(
                format!("{field}.formula_type"),
                format!("unknown decay formula {}", def.formula_type),
            ));
        }
    };
    if def.anchor_field.is_empty() {
        return Err(invalid_field(
            format!("{field}.anchor_field"),
            "an anchor property is required",
        ));
    }
    if def.duration_secs == 0 {
        return Err(invalid_field(
            format!("{field}.duration_secs"),
            "the duration must be greater than 0",
        ));
    }
    let anchor_field = def.anchor_field.clone();
    match ComputedType::try_from(def.computed_type) {
        Ok(ComputedType::Ttl) => {
            let scope = match TtlScopeType::try_from(def.scope) {
                Ok(TtlScopeType::Unspecified | TtlScopeType::Node) => TtlScope::Node,
                Ok(TtlScopeType::Field) => TtlScope::Field,
                Ok(TtlScopeType::Subtree) => TtlScope::Subtree,
                Err(_) => {
                    return Err(invalid_field(
                        format!("{field}.scope"),
                        format!("unknown TTL scope {}", def.scope),
                    ));
                }
            };
            Ok(ComputedSpec::Ttl {
                duration_secs: def.duration_secs,
                anchor_field,
                scope,
                target_field: (!def.target_field.is_empty()).then(|| def.target_field.clone()),
            })
        }
        Ok(ComputedType::Decay) => Ok(ComputedSpec::Decay {
            formula,
            initial: def.initial,
            target: def.target,
            duration_secs: def.duration_secs,
            anchor_field,
        }),
        Ok(ComputedType::VectorDecay) => Ok(ComputedSpec::VectorDecay {
            formula,
            duration_secs: def.duration_secs,
            anchor_field,
        }),
        Ok(ComputedType::Unspecified) => Err(invalid_field(
            format!("{field}.computed_type"),
            "the computed type is unspecified",
        )),
        Err(_) => Err(invalid_field(
            format!("{field}.computed_type"),
            format!("unknown computed type {}", def.computed_type),
        )),
    }
}

fn computed_to_proto(name: &str, spec: &ComputedSpec) -> schema::ComputedPropertyDefinition {
    let mut def = schema::ComputedPropertyDefinition {
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
            def.computed_type = schema::ComputedType::Ttl as i32;
            def.duration_secs = *duration_secs;
            def.anchor_field = anchor_field.clone();
            def.scope = match scope {
                TtlScope::Field => schema::TtlScopeType::Field,
                TtlScope::Subtree => schema::TtlScopeType::Subtree,
                TtlScope::Node => schema::TtlScopeType::Node,
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
            def.computed_type = schema::ComputedType::Decay as i32;
            def.duration_secs = *duration_secs;
            def.anchor_field = anchor_field.clone();
            def.initial = *initial;
            def.target = *target;
            set_formula(&mut def, formula);
        }
        ComputedSpec::VectorDecay {
            formula,
            duration_secs,
            anchor_field,
        } => {
            def.computed_type = schema::ComputedType::VectorDecay as i32;
            def.duration_secs = *duration_secs;
            def.anchor_field = anchor_field.clone();
            set_formula(&mut def, formula);
        }
    }
    def
}

fn set_formula(def: &mut schema::ComputedPropertyDefinition, formula: &DecayFormula) {
    use schema::DecayFormulaType as F;
    let wire = match formula {
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
    };
    def.formula_type = wire as i32;
}

fn schema_mode_from_proto(v: i32) -> Result<SchemaMode, Status> {
    use schema::SchemaMode as M;
    match M::try_from(v) {
        Ok(M::Unspecified | M::Strict) => Ok(SchemaMode::Strict),
        Ok(M::Validated) => Ok(SchemaMode::Validated),
        Ok(M::Flexible) => Ok(SchemaMode::Flexible),
        Err(_) => Err(invalid_field(
            "schema_mode",
            format!("unknown schema mode {v}"),
        )),
    }
}

fn schema_mode_to_proto(mode: SchemaMode) -> i32 {
    let wire = match mode {
        SchemaMode::Strict => schema::SchemaMode::Strict,
        SchemaMode::Validated => schema::SchemaMode::Validated,
        SchemaMode::Flexible => schema::SchemaMode::Flexible,
    };
    wire as i32
}

fn label_to_proto(s: &LabelSchema) -> schema::Label {
    let mut properties = Vec::new();
    let mut computed_properties = Vec::new();
    for p in s.properties.values() {
        match &p.property_type {
            PropertyType::Computed(spec) => {
                computed_properties.push(computed_to_proto(&p.name, spec))
            }
            _ => properties.push(property_to_proto(p)),
        }
    }
    schema::Label {
        name: s.name.clone(),
        declared: true,
        properties,
        computed_properties,
        schema_mode: schema_mode_to_proto(s.mode),
        temporal: s.temporal,
        schema_revision: s.schema_revision,
    }
}

fn edge_type_to_proto(s: &EdgeTypeSchema) -> schema::EdgeType {
    schema::EdgeType {
        name: s.name.clone(),
        declared: true,
        properties: s.properties.values().map(property_to_proto).collect(),
        temporal: s.temporal,
        schema_revision: s.schema_revision,
        discriminator: s
            .discriminator()
            .map(|d| d.column.clone())
            .unwrap_or_default(),
    }
}

/// An index build as the wire shows it.
fn build_to_proto(status: &BuildStatus) -> schema::IndexBuild {
    use schema::{IndexBuildPhase as P, IndexBuildState as S};
    let state = match status.record.as_ref().map(|r| &r.state) {
        Some(BuildState::Accepted) => S::Accepted,
        // A build a member runs for itself keeps no record.
        Some(BuildState::Running { .. }) | None => S::Running,
        Some(BuildState::Published) => S::Published,
        Some(BuildState::Failed { .. }) => S::Failed,
        Some(BuildState::Cancelled) => S::Cancelled,
    };
    let phase = match status.phase {
        None => P::Unspecified,
        Some(BuildPhase::AwaitingSeat) => P::AwaitingSeat,
        Some(BuildPhase::AwaitingOlderTransactions) => P::AwaitingOlderTransactions,
        Some(BuildPhase::Indexing { .. }) => P::Indexing,
    };
    let index = status.index.as_ref();
    schema::IndexBuild {
        operation: status.generation.as_raw(),
        index: index.and_then(|i| i.name.clone()).unwrap_or_default(),
        label: index.map(|i| i.label.clone()).unwrap_or_default(),
        state: state as i32,
        phase: phase as i32,
        indexed: status.indexed().unwrap_or(0),
        failure: status.failure().unwrap_or_default().to_string(),
        rename_property: status
            .record
            .as_ref()
            .and_then(|r| r.on_duplicate.as_ref())
            .map(|r| r.property.clone())
            .unwrap_or_default(),
        repaired: status.record.as_ref().map_or(0, |r| r.repaired),
    }
}

/// `NOT_FOUND` for an index build operation no build has.
fn unknown_build(operation: GenerationId) -> Status {
    let operation = operation.as_raw().to_string();
    catalog_object_status(
        tonic::Code::NotFound,
        format!("no index build has operation {operation}"),
        Reason::CatalogObjectNotFound,
        "index_build",
        &operation,
    )
}

/// The wait a request names, refusing a negative one.
fn wait_from_proto(
    field: &str,
    wait: Option<prost_types::Duration>,
) -> Result<Option<Duration>, Status> {
    wait.map(|w| {
        Duration::try_from(w).map_err(|_| invalid_field(field, "a wait is a non-negative duration"))
    })
    .transpose()
}

/// The build operation a request names. Operation 0 is a real build, so an
/// absent field is refused rather than read as it.
fn operation_from_proto(operation: Option<u64>) -> Result<GenerationId, Status> {
    operation
        .map(GenerationId::from_raw)
        .ok_or_else(|| invalid_field("operation", "an operation is required"))
}

/// A constraint's scope as the wire shows it.
fn scope_to_proto(scope: &PartialFilter) -> schema::ConstraintScope {
    use schema::constraint_scope::Test;
    let (property, test) = match scope {
        PartialFilter::PropertyEquals { property, value } => {
            (property, Test::EqualsString(value.clone()))
        }
        PartialFilter::PropertyEqualsInt { property, value } => (property, Test::EqualsInt(*value)),
        PartialFilter::PropertyEqualsBool { property, value } => {
            (property, Test::EqualsBool(*value))
        }
        PartialFilter::PropertyExists { property } => (property, Test::Present(())),
    };
    schema::ConstraintScope {
        property: property.clone(),
        test: Some(test),
    }
}

/// The scope a request declares, refusing one without a property or a test.
fn scope_from_proto(
    scope: Option<schema::ConstraintScope>,
) -> Result<Option<PartialFilter>, Status> {
    use schema::constraint_scope::Test;
    let Some(scope) = scope else {
        return Ok(None);
    };
    if scope.property.is_empty() {
        return Err(invalid_field("scope.property", "a property is required"));
    }
    let property = scope.property;
    Ok(Some(match scope.test {
        Some(Test::EqualsString(value)) => PartialFilter::PropertyEquals { property, value },
        Some(Test::EqualsInt(value)) => PartialFilter::PropertyEqualsInt { property, value },
        Some(Test::EqualsBool(value)) => PartialFilter::PropertyEqualsBool { property, value },
        Some(Test::Present(())) => PartialFilter::PropertyExists { property },
        None => return Err(invalid_field("scope.test", "a test is required")),
    }))
}

fn constraint_to_proto(c: &LabelConstraint, build: Option<&BuildStatus>) -> schema::Constraint {
    use schema::ConstraintKind as K;
    let (kind, property_type) = match &c.constraint.kind {
        ConstraintKind::Unique => (K::Unique, None),
        ConstraintKind::NotNull => (K::NotNull, None),
        ConstraintKind::NodeKey => (K::NodeKey, None),
        ConstraintKind::Type(t) => (K::PropertyType, Some(property_type_to_proto(t))),
    };
    schema::Constraint {
        name: c.constraint.name.clone(),
        target: Some(schema::constraint::Target::Label(c.label.clone())),
        properties: c.constraint.properties.clone(),
        kind: kind as i32,
        property_type,
        state: match c.constraint.state {
            ConstraintState::Validating => schema::ConstraintState::Validating,
            ConstraintState::Active => schema::ConstraintState::Active,
        } as i32,
        backing_index: c.backing_index.clone().unwrap_or_default(),
        build: build.map(build_to_proto),
        scope: c.constraint.scope.as_ref().map(scope_to_proto),
    }
}

/// The constraint a request declares, refusing one whose shape its kind does
/// not allow.
fn declaration_from_proto(
    req: schema::CreateConstraintRequest,
) -> Result<ConstraintDeclaration, Status> {
    use schema::ConstraintKind as K;
    let label = match req.target {
        Some(schema::create_constraint_request::Target::Label(label)) if !label.is_empty() => label,
        _ => return Err(invalid_field("label", "the constrained label is required")),
    };
    let wire_kind = match K::try_from(req.kind) {
        Ok(K::Unspecified) => {
            return Err(invalid_field("kind", "the constraint kind is unspecified"));
        }
        Err(_) => {
            return Err(invalid_field(
                "kind",
                format!("unknown constraint kind {}", req.kind),
            ));
        }
        Ok(kind) => kind,
    };
    if req.properties.is_empty() {
        return Err(invalid_field(
            "properties",
            "at least one property is required",
        ));
    }
    if let Some((i, _)) = req
        .properties
        .iter()
        .enumerate()
        .find(|(_, p)| p.is_empty())
    {
        return Err(invalid_field(
            format!("properties[{i}]"),
            "a name is required",
        ));
    }
    let mut seen = HashSet::new();
    if let Some((i, p)) = req
        .properties
        .iter()
        .enumerate()
        .find(|(_, p)| !seen.insert(p.as_str()))
    {
        return Err(invalid_field(
            format!("properties[{i}]"),
            format!("property `{p}` is named twice"),
        ));
    }
    if matches!(wire_kind, K::NotNull | K::PropertyType) && req.properties.len() != 1 {
        return Err(invalid_field(
            "properties",
            "this kind constrains exactly one property",
        ));
    }
    if wire_kind != K::PropertyType && req.property_type.is_some() {
        return Err(invalid_field(
            "property_type",
            "only a PROPERTY_TYPE constraint names a type",
        ));
    }
    let kind = match wire_kind {
        K::PropertyType => ConstraintKind::Type(property_type_from_proto(
            "property_type",
            req.property_type.as_ref(),
        )?),
        K::NotNull => ConstraintKind::NotNull,
        K::NodeKey => ConstraintKind::NodeKey,
        // UNSPECIFIED was refused when the kind was read.
        K::Unique | K::Unspecified => ConstraintKind::Unique,
    };
    Ok(ConstraintDeclaration {
        scope: scope_from_proto(req.scope)?,
        wait: wait_from_proto("wait", req.wait)?,
        on_duplicate_rename: (!req.on_duplicate_rename.is_empty())
            .then_some(req.on_duplicate_rename),
        name: (!req.name.is_empty()).then_some(req.name),
        label,
        properties: req.properties,
        kind,
        if_not_exists: req.if_not_exists,
    })
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
impl schema::schema_service_server::SchemaService for SchemaServiceImpl {
    async fn create_label(
        &self,
        request: Request<schema::CreateLabelRequest>,
    ) -> Result<Response<schema::Label>, Status> {
        let req = request.into_inner();
        if req.name.is_empty() {
            return Err(invalid_field("name", "a label name is required"));
        }
        let mut definition = LabelSchema::new_node_id(&req.name);
        definition.set_mode(schema_mode_from_proto(req.schema_mode)?);
        definition.set_temporal(req.temporal);
        let mut names = HashSet::new();
        for p in properties_from_proto("properties", &req.properties, &mut names)? {
            definition.add_property(p);
        }
        for (i, cp) in req.computed_properties.iter().enumerate() {
            let at = format!("computed_properties[{i}]");
            if cp.name.is_empty() {
                return Err(invalid_field(format!("{at}.name"), "a name is required"));
            }
            if !names.insert(cp.name.clone()) {
                return Err(invalid_field(
                    format!("{at}.name"),
                    format!("property `{}` is declared twice", cp.name),
                ));
            }
            definition.add_property(PropertyDef::computed(
                &cp.name,
                computed_from_proto(&at, cp)?,
            ));
        }

        let name = req.name;
        let label = super::blocking(|| -> Result<schema::Label, Status> {
            let db = self.database.write();
            db.define_label(definition).map_err(db_error_to_status)?;
            db.label_schemas()
                .map_err(db_error_to_status)?
                .iter()
                .find(|s| s.name == name)
                .map(label_to_proto)
                .ok_or_else(|| {
                    Status::internal(format!("label '{name}' vanished after its definition"))
                })
        })?;
        Ok(Response::new(label))
    }

    async fn list_labels(
        &self,
        _request: Request<schema::ListLabelsRequest>,
    ) -> Result<Response<schema::ListLabelsResponse>, Status> {
        let labels = super::blocking(|| -> Result<Vec<schema::Label>, Status> {
            let mut db = self.database.write();
            let mut labels: std::collections::BTreeMap<String, schema::Label> = db
                .label_schemas()
                .map_err(db_error_to_status)?
                .iter()
                .map(|s| (s.name.clone(), label_to_proto(s)))
                .collect();
            // Labels the stored nodes carry without a definition.
            let rows = db
                .execute_cypher("MATCH (n) RETURN DISTINCT n.__label__ AS lbl")
                .map_err(db_error_to_status)?;
            for row in rows {
                if let Some(Value::String(name)) = row.get("lbl") {
                    if !name.is_empty() {
                        labels.entry(name.clone()).or_insert_with(|| schema::Label {
                            name: name.clone(),
                            ..Default::default()
                        });
                    }
                }
            }
            Ok(labels.into_values().collect())
        })?;
        Ok(Response::new(schema::ListLabelsResponse { labels }))
    }

    async fn create_edge_type(
        &self,
        request: Request<schema::CreateEdgeTypeRequest>,
    ) -> Result<Response<schema::EdgeType>, Status> {
        let req = request.into_inner();
        if req.name.is_empty() {
            return Err(invalid_field("name", "an edge type name is required"));
        }
        let mut definition = EdgeTypeSchema::new(&req.name);
        definition.set_temporal(req.temporal);
        for p in properties_from_proto("properties", &req.properties, &mut HashSet::new())? {
            definition.add_property(p);
        }
        definition
            .resolve_identity(Some(req.discriminator.as_str()).filter(|d| !d.is_empty()))
            .map_err(|why| invalid_field("discriminator", why))?;

        let name = req.name;
        let edge_type = super::blocking(|| -> Result<schema::EdgeType, Status> {
            let db = self.database.write();
            db.define_edge_type(definition)
                .map_err(db_error_to_status)?;
            db.edge_type_schemas()
                .map_err(db_error_to_status)?
                .iter()
                .find(|s| s.name == name)
                .map(edge_type_to_proto)
                .ok_or_else(|| {
                    Status::internal(format!("edge type '{name}' vanished after its definition"))
                })
        })?;
        Ok(Response::new(edge_type))
    }

    async fn list_edge_types(
        &self,
        _request: Request<schema::ListEdgeTypesRequest>,
    ) -> Result<Response<schema::ListEdgeTypesResponse>, Status> {
        let edge_types = super::blocking(|| -> Result<Vec<schema::EdgeType>, Status> {
            let db = self.database.read();
            let mut types: std::collections::BTreeMap<String, schema::EdgeType> = db
                .edge_type_schemas()
                .map_err(db_error_to_status)?
                .iter()
                .map(|s| (s.name.clone(), edge_type_to_proto(s)))
                .collect();
            // Edge types the stored edges carry without a definition.
            for name in db.edge_type_names().map_err(db_error_to_status)? {
                types
                    .entry(name.clone())
                    .or_insert_with(|| schema::EdgeType {
                        name,
                        ..Default::default()
                    });
            }
            Ok(types.into_values().collect())
        })?;
        Ok(Response::new(schema::ListEdgeTypesResponse { edge_types }))
    }

    async fn create_constraint(
        &self,
        request: Request<schema::CreateConstraintRequest>,
    ) -> Result<Response<schema::Constraint>, Status> {
        let declaration = declaration_from_proto(request.into_inner())?;
        let created = super::blocking(|| -> Result<schema::Constraint, Status> {
            // Shared: the call may wait for the build, and the database
            // serves everyone else meanwhile.
            let db = self.database.read();
            let created = db
                .create_constraint(declaration)
                .map_err(db_error_to_status)?;
            let build = match created.operation {
                Some(operation) => db
                    .index_build(operation, Duration::ZERO)
                    .map_err(db_error_to_status)?,
                None => None,
            };
            Ok(constraint_to_proto(&created, build.as_ref()))
        })?;
        Ok(Response::new(created))
    }

    async fn drop_constraint(
        &self,
        request: Request<schema::DropConstraintRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        if req.name.is_empty() {
            return Err(invalid_field("name", "a constraint name is required"));
        }
        super::blocking(|| {
            self.database
                .write()
                .drop_constraint(&req.name, req.if_exists)
                .map_err(db_error_to_status)
        })?;
        Ok(Response::new(()))
    }

    async fn list_constraints(
        &self,
        _request: Request<schema::ListConstraintsRequest>,
    ) -> Result<Response<schema::ListConstraintsResponse>, Status> {
        let constraints = super::blocking(|| -> Result<Vec<schema::Constraint>, Status> {
            let db = self.database.read();
            let constraints = db.constraints().map_err(db_error_to_status)?;
            let builds: rustc_hash::FxHashMap<GenerationId, BuildStatus> = db
                .index_build_status()
                .map_err(db_error_to_status)?
                .into_iter()
                .map(|s| (s.generation, s))
                .collect();
            Ok(constraints
                .iter()
                .map(|c| constraint_to_proto(c, c.operation.and_then(|op| builds.get(&op))))
                .collect())
        })?;
        Ok(Response::new(schema::ListConstraintsResponse {
            constraints,
        }))
    }

    async fn list_index_builds(
        &self,
        _request: Request<schema::ListIndexBuildsRequest>,
    ) -> Result<Response<schema::ListIndexBuildsResponse>, Status> {
        let builds = super::blocking(|| {
            self.database
                .read()
                .index_build_status()
                .map_err(db_error_to_status)
        })?;
        Ok(Response::new(schema::ListIndexBuildsResponse {
            builds: builds.iter().map(build_to_proto).collect(),
        }))
    }

    async fn get_index_build(
        &self,
        request: Request<schema::GetIndexBuildRequest>,
    ) -> Result<Response<schema::IndexBuild>, Status> {
        let req = request.into_inner();
        let operation = operation_from_proto(req.operation)?;
        let wait = wait_from_proto("wait", req.wait)?.unwrap_or(Duration::ZERO);
        let build = super::blocking(|| {
            self.database
                .read()
                .index_build(operation, wait)
                .map_err(db_error_to_status)
        })?
        .ok_or_else(|| unknown_build(operation))?;
        Ok(Response::new(build_to_proto(&build)))
    }

    async fn cancel_index_build(
        &self,
        request: Request<schema::CancelIndexBuildRequest>,
    ) -> Result<Response<schema::IndexBuild>, Status> {
        let req = request.into_inner();
        let operation = operation_from_proto(req.operation)?;
        let build = super::blocking(|| -> Result<schema::IndexBuild, Status> {
            let db = self.database.read();
            let cancelled = db
                .cancel_index_build(operation)
                .map_err(db_error_to_status)?;
            match db
                .index_build(operation, Duration::ZERO)
                .map_err(db_error_to_status)?
            {
                Some(status) => Ok(build_to_proto(&status)),
                // A vector build keeps no record: stopped here, nothing of
                // it remains to inspect but the cancellation itself.
                None if cancelled => Ok(schema::IndexBuild {
                    operation: operation.as_raw(),
                    state: schema::IndexBuildState::Cancelled as i32,
                    ..Default::default()
                }),
                None => Err(unknown_build(operation)),
            }
        })?;
        Ok(Response::new(build))
    }

    async fn list_index_build_repairs(
        &self,
        request: Request<schema::ListIndexBuildRepairsRequest>,
    ) -> Result<Response<schema::ListIndexBuildRepairsResponse>, Status> {
        let operation = operation_from_proto(request.into_inner().operation)?;
        let repairs = super::blocking(|| {
            let db = self.database.read();
            if db
                .index_build(operation, Duration::ZERO)
                .map_err(db_error_to_status)?
                .is_none()
            {
                return Err(unknown_build(operation));
            }
            db.index_build_repairs(operation)
                .map_err(db_error_to_status)
        })?;
        Ok(Response::new(schema::ListIndexBuildRepairsResponse {
            repairs: repairs
                .into_iter()
                .map(|r| schema::IndexBuildRepair {
                    element_id: coordinode_core::graph::node::NodeId::from_raw(r.node)
                        .to_element_id(),
                    property: r.property,
                    old_value: r.old,
                    new_value: r.new,
                })
                .collect(),
        }))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
