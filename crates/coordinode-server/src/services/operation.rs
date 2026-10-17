//! Client operation identity on the wire: the reference a write carries, the
//! outcomes a client resolves, and `OperationService`, which enrolls issuers
//! and resolves, acknowledges and retires their operations.
//!
//! A call that changes operation records is decided by the group's log; one
//! reaching a member that does not lead is passed to the leader, as a Cypher
//! write is, so a client never has to know which member that is.

use coordinode_core::group::DeploymentId;
use coordinode_embed::DatabaseError;
use coordinode_embed::operation::{
    IssuerId, OperationDescriptor, OperationId, OperationOutcome, Resolve,
};
use tonic::{Code, Request, Response, Status};

use super::cypher::db_error_to_status;
use super::error_details::{Reason, invalid_field, status_with_reason};
use super::statement::{FORWARDED_HEADER, StatementExecutor, leader_hint};
use crate::proto::operation as wire;

/// The descriptor `reference` names, refusing a malformed one. `field` is the
/// request path of the reference, for the refusal.
pub(crate) fn descriptor_from_proto(
    field: &str,
    reference: &wire::OperationRef,
) -> Result<OperationDescriptor, Status> {
    Ok(OperationDescriptor {
        op: OperationId {
            issuer: issuer_from_proto(&format!("{field}.issuer"), &reference.issuer)?,
            incarnation: reference.incarnation,
            sequence: reference.sequence,
        },
        deployment: deployment_from_proto(&format!("{field}.deployment"), &reference.deployment)?,
    })
}

/// The descriptor of an optional reference.
pub(crate) fn optional_descriptor(
    field: &str,
    reference: Option<&wire::OperationRef>,
) -> Result<Option<OperationDescriptor>, Status> {
    reference
        .map(|reference| descriptor_from_proto(field, reference))
        .transpose()
}

/// The wire form of `descriptor`.
pub(crate) fn descriptor_to_proto(descriptor: &OperationDescriptor) -> wire::OperationRef {
    wire::OperationRef {
        issuer: descriptor.op.issuer.0.to_vec(),
        incarnation: descriptor.op.incarnation,
        sequence: descriptor.op.sequence,
        deployment: descriptor.deployment.as_bytes().to_vec(),
    }
}

fn issuer_from_proto(field: &str, bytes: &[u8]) -> Result<IssuerId, Status> {
    <[u8; 16]>::try_from(bytes)
        .map(IssuerId)
        .map_err(|_| invalid_field(field, "an issuer is 16 bytes"))
}

fn deployment_from_proto(field: &str, bytes: &[u8]) -> Result<DeploymentId, Status> {
    DeploymentId::from_slice(bytes).ok_or_else(|| {
        invalid_field(
            field,
            "a deployment is 16 bytes: the one that enrolled the incarnation",
        )
    })
}

/// The wire state of `outcome`, and the commit timestamp of a committed one.
pub(crate) fn outcome_to_proto(outcome: OperationOutcome) -> (wire::OperationState, u64) {
    match outcome {
        OperationOutcome::Committed { commit_ts, .. } => {
            (wire::OperationState::Committed, commit_ts)
        }
        OperationOutcome::Aborted => (wire::OperationState::Aborted, 0),
        OperationOutcome::Unknown => (wire::OperationState::Unknown, 0),
        OperationOutcome::Retired => (wire::OperationState::Retired, 0),
    }
}

/// The status of a write refused because its operation is settled.
pub(crate) fn settled_status(
    message: String,
    outcome: OperationOutcome,
    intent_differs: bool,
) -> Status {
    let (state, commit_ts) = outcome_to_proto(outcome);
    let name = state
        .as_str_name()
        .trim_start_matches("OPERATION_STATE_")
        .to_string();
    let mut metadata = vec![
        ("outcome", name),
        ("intent_differs", intent_differs.to_string()),
    ];
    if state == wire::OperationState::Committed {
        metadata.push(("commit_ts", commit_ts.to_string()));
    }
    status_with_reason(
        Code::FailedPrecondition,
        message,
        Reason::OperationSettled,
        metadata,
    )
}

/// The status of a reference naming another deployment.
pub(crate) fn foreign_status(message: String, deployment: DeploymentId) -> Status {
    status_with_reason(
        Code::InvalidArgument,
        message,
        Reason::ForeignOperation,
        [("deployment", deployment.to_string())],
    )
}

/// Enrolls issuers and resolves, acknowledges and retires their operations.
pub struct OperationServiceImpl {
    executor: StatementExecutor,
}

impl OperationServiceImpl {
    /// Serve over the database and consensus node `executor` runs statements
    /// against, so a call needing the leader goes where its statements go.
    pub fn new(executor: StatementExecutor) -> Self {
        Self { executor }
    }

    /// Run `call` on the database; a call refused because this member does
    /// not lead is passed, once, to the leader through `forward`.
    async fn decided<T, F, Fut>(
        &self,
        forwarded: bool,
        call: impl FnOnce(&coordinode_embed::Database) -> Result<T, DatabaseError>,
        forward: F,
    ) -> Result<T, Status>
    where
        F: FnOnce(
            wire::operation_service_client::OperationServiceClient<tonic::transport::Channel>,
        ) -> Fut,
        Fut: std::future::Future<Output = Result<T, Status>>,
    {
        let result = super::blocking(|| call(&self.executor.database().read()));
        match result {
            Ok(value) => Ok(value),
            Err(e) => match leader_hint(&e) {
                Some(leader_id) if !forwarded => {
                    let channel = self.executor.leader_channel(leader_id)?;
                    forward(wire::operation_service_client::OperationServiceClient::new(
                        channel,
                    ))
                    .await
                }
                _ => Err(db_error_to_status(e)),
            },
        }
    }
}

/// A request passed to the leader, marked so it is not passed again.
fn onward<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request.metadata_mut().insert(
        FORWARDED_HEADER,
        tonic::metadata::MetadataValue::from_static("1"),
    );
    request
}

#[tonic::async_trait]
impl wire::operation_service_server::OperationService for OperationServiceImpl {
    async fn enroll_issuer(
        &self,
        request: Request<wire::EnrollIssuerRequest>,
    ) -> Result<Response<wire::EnrollIssuerResponse>, Status> {
        let forwarded = request.metadata().contains_key(FORWARDED_HEADER);
        let req = request.into_inner();
        let issuer = issuer_from_proto("issuer", &req.issuer)?;
        self.decided(
            forwarded,
            |db| {
                let enrolled = db.enroll_issuer(issuer)?;
                Ok(wire::EnrollIssuerResponse {
                    incarnation: enrolled.incarnation(),
                    next_sequence: enrolled.next_sequence(),
                    deployment: enrolled.deployment().as_bytes().to_vec(),
                })
            },
            |mut leader| async move { Ok(leader.enroll_issuer(onward(req)).await?.into_inner()) },
        )
        .await
        .map(Response::new)
    }

    async fn resume_issuer(
        &self,
        request: Request<wire::ResumeIssuerRequest>,
    ) -> Result<Response<wire::ResumeIssuerResponse>, Status> {
        let req = request.into_inner();
        let issuer = issuer_from_proto("issuer", &req.issuer)?;
        let resumed = super::blocking(|| {
            self.executor.database().read().resume_issuer(
                issuer,
                req.incarnation,
                req.next_sequence,
            )
        })
        .map_err(|e| match e {
            // Closed or never enrolled: the client enrolls a new incarnation.
            DatabaseError::Other(message) => Status::failed_precondition(message),
            other => db_error_to_status(other),
        })?;
        Ok(Response::new(wire::ResumeIssuerResponse {
            next_sequence: resumed.next_sequence(),
            deployment: resumed.deployment().as_bytes().to_vec(),
        }))
    }

    async fn resolve_operation(
        &self,
        request: Request<wire::ResolveOperationRequest>,
    ) -> Result<Response<wire::ResolveOperationResponse>, Status> {
        let forwarded = request.metadata().contains_key(FORWARDED_HEADER);
        let req = request.into_inner();
        let reference = req
            .operation
            .as_ref()
            .ok_or_else(|| invalid_field("operation", "the operation to resolve is required"))?;
        let descriptor = descriptor_from_proto("operation", reference)?;
        let resolve = if req.exclude {
            Resolve::Exclude
        } else {
            Resolve::Lookup
        };
        self.decided(
            forwarded,
            |db| {
                let (state, commit_ts) =
                    outcome_to_proto(db.resolve_operation(&descriptor, resolve)?);
                Ok(wire::ResolveOperationResponse {
                    state: state.into(),
                    commit_ts,
                })
            },
            |mut leader| async move {
                Ok(leader
                    .resolve_operation(onward(req))
                    .await?
                    .into_inner())
            },
        )
        .await
        .map(Response::new)
    }

    async fn acknowledge_operations(
        &self,
        request: Request<wire::AcknowledgeOperationsRequest>,
    ) -> Result<Response<wire::AcknowledgeOperationsResponse>, Status> {
        let forwarded = request.metadata().contains_key(FORWARDED_HEADER);
        let req = request.into_inner();
        let issuer = issuer_from_proto("issuer", &req.issuer)?;
        let deployment = deployment_from_proto("deployment", &req.deployment)?;
        let descriptors: Vec<OperationDescriptor> = req
            .sequences
            .iter()
            .map(|&sequence| OperationDescriptor {
                op: OperationId {
                    issuer,
                    incarnation: req.incarnation,
                    sequence,
                },
                deployment,
            })
            .collect();
        self.decided(
            forwarded,
            |db| {
                for descriptor in &descriptors {
                    db.acknowledge_operation(descriptor)?;
                }
                Ok(wire::AcknowledgeOperationsResponse {})
            },
            |mut leader| async move {
                Ok(leader
                    .acknowledge_operations(onward(req))
                    .await?
                    .into_inner())
            },
        )
        .await
        .map(Response::new)
    }

    async fn retire_operations(
        &self,
        request: Request<wire::RetireOperationsRequest>,
    ) -> Result<Response<wire::RetireOperationsResponse>, Status> {
        let forwarded = request.metadata().contains_key(FORWARDED_HEADER);
        let req = request.into_inner();
        let bound = OperationDescriptor {
            op: OperationId {
                issuer: issuer_from_proto("issuer", &req.issuer)?,
                incarnation: req.incarnation,
                sequence: req.below,
            },
            deployment: deployment_from_proto("deployment", &req.deployment)?,
        };
        self.decided(
            forwarded,
            |db| {
                db.retire_operations(&bound)?;
                Ok(wire::RetireOperationsResponse {})
            },
            |mut leader| async move {
                Ok(leader
                    .retire_operations(onward(req))
                    .await?
                    .into_inner())
            },
        )
        .await
        .map(Response::new)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
