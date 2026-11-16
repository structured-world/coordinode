//! Change-stream subscriptions over the client's persistent session.
//!
//! A subscription reads as a consumer registered on the server with a
//! retention policy; the registration outlives the subscription and the
//! connection, so a consumer resumes it later by its incarnation. Committed
//! transactions arrive as units of logical events, in batches with the
//! cursor after them. The driver grants the server credit as the caller takes
//! batches, so a caller that stops reading stops the server sending, and
//! drains at once when the server says applied entries are waiting.

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::error::ClientError;
use crate::proto::replication::{
    AcknowledgeSubscriptionRequest, CancelSubscriptionRequest, ChangeInterest, ConsumerRetention,
    NameList, OrdinaryRetention, StrictRetention, SubscribeRequest, consumer_retention,
    logical_change,
};
use crate::proto::session::server_frame::Event;
use crate::proto::session::{Cancel, Credit, Subscribe, client_frame};
use crate::session::{SessionLink, error_status};

/// Log entries granted to the server at a time unless set otherwise.
pub const DEFAULT_WINDOW: u32 = 256;

/// Whether a consumer's registration protects the history it still needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retention {
    /// The server keeps every entry from the acknowledged cursor until the
    /// registration is cancelled.
    Strict,
    /// The registration records the acknowledged cursor only; the server's
    /// own log retention decides how far back it reaches. A position it no
    /// longer holds is refused as retention lost, and the consumer rebuilds.
    Ordinary,
}

/// Which consumer a subscription reads as, what it follows and from where.
#[derive(Debug, Clone)]
pub struct SubscribeOptions {
    consumer_id: String,
    /// `None` resumes `incarnation`.
    retention: Option<Retention>,
    incarnation: u64,
    from: Option<Cursor>,
    labels: Option<Vec<String>>,
    edge_types: Option<Vec<String>>,
    blobs: bool,
    window: u32,
}

impl SubscribeOptions {
    /// Register `consumer_id` anew with `retention`, following every change
    /// from the oldest the server holds.
    pub fn register(consumer_id: impl Into<String>, retention: Retention) -> Self {
        Self {
            consumer_id: consumer_id.into(),
            retention: Some(retention),
            incarnation: 0,
            from: None,
            labels: None,
            edge_types: None,
            blobs: false,
            window: DEFAULT_WINDOW,
        }
    }

    /// Resume the live registration `incarnation` of `consumer_id`, reading
    /// from its last acknowledged cursor.
    pub fn resume(consumer_id: impl Into<String>, incarnation: u64) -> Self {
        Self {
            retention: None,
            incarnation,
            ..Self::register(consumer_id, Retention::Strict)
        }
    }

    /// Read from `cursor` instead: the cursor of a batch received with the
    /// same interest.
    pub fn from(mut self, cursor: Cursor) -> Self {
        self.from = Some(cursor);
        self
    }

    /// Follow the nodes of `labels` only: a node event arrives when the node
    /// carries one of them before or after it.
    pub fn labels(mut self, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.labels = Some(labels.into_iter().map(Into::into).collect());
        self
    }

    /// Follow the relationships of `edge_types` only.
    pub fn edge_types(mut self, edge_types: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.edge_types = Some(edge_types.into_iter().map(Into::into).collect());
        self
    }

    /// Follow blob objects too: what each owner holds, put and removed.
    pub fn blobs(mut self, blobs: bool) -> Self {
        self.blobs = blobs;
        self
    }

    /// Grant the server `entries` log entries at a time (at least 1).
    pub fn window(mut self, entries: u32) -> Self {
        self.window = entries.max(1);
        self
    }

    fn request(&self) -> SubscribeRequest {
        let retention = self.retention.map(|retention| ConsumerRetention {
            policy: Some(match retention {
                Retention::Strict => consumer_retention::Policy::Strict(StrictRetention {}),
                Retention::Ordinary => consumer_retention::Policy::Ordinary(OrdinaryRetention {}),
            }),
        });
        let names = |list: &Option<Vec<String>>| {
            list.as_ref().map(|names| NameList {
                names: names.clone(),
            })
        };
        SubscribeRequest {
            cursor: self.from.as_ref().map(|c| c.0.clone()).unwrap_or_default(),
            interest: Some(ChangeInterest {
                labels: names(&self.labels),
                edge_types: names(&self.edge_types),
                blobs: self.blobs,
            }),
            consumer_id: self.consumer_id.clone(),
            incarnation: self.incarnation,
            retention,
        }
    }
}

/// Where a stream continues after a batch: pass it to
/// [`Subscription::acknowledge`] once the batch is held, or to
/// [`SubscribeOptions::from`] to read from there. Opaque bytes to save with
/// what the caller built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cursor(Vec<u8>);

impl Cursor {
    /// The cursor's bytes, to save.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// A cursor saved as `bytes`.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

/// What an event does to its target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeAction {
    /// The target is written whole: replaced, or it comes to exist.
    Put,
    /// The target changes by the payload's delta.
    Patch,
    /// The target ceases to exist.
    Remove,
    /// What a type implies changed: its nodes enter and leave label scopes
    /// with no node event of their own. A reader keeping state by label
    /// reconciles the type's nodes, or rebuilds the scope.
    ScopeChange,
    /// A label is dropped: its identity and scope end.
    Drop,
    /// An action this driver does not know.
    Other(i32),
}

/// How a node event reaches the labels the subscription follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeDelivery {
    /// The node stays in the subscription's scope: the event applies.
    Apply,
    /// The node joins the scope with the record the event writes.
    Enter,
    /// The node leaves the scope: deleted, relabelled or retyped out of it.
    Leave,
    /// A delivery this driver does not know.
    Other(i32),
}

/// A relationship by type and endpoints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relationship {
    /// The relationship type's name.
    pub edge_type: String,
    /// The source node's id.
    pub source_id: u64,
    /// The target node's id.
    pub target_id: u64,
}

/// A label as the transaction knew it. Two labels of the same name and
/// different identities are different labels: one was dropped and the
/// other created later, and they have different scopes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Label {
    /// The label's identity, never reused.
    pub id: u32,
    /// The name it was created under.
    pub name: String,
}

impl From<crate::proto::replication::Label> for Label {
    fn from(label: crate::proto::replication::Label) -> Self {
        Self {
            id: label.id,
            name: label.name,
        }
    }
}

/// What an event changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeTarget {
    /// A node's record, or one version of a temporal node.
    Node {
        /// The node's id.
        node_id: u64,
        /// The version's start for a temporal node, in milliseconds.
        valid_from: Option<i64>,
    },
    /// A relationship's presence.
    Relationship(Relationship),
    /// The properties of a relationship that is neither versioned nor
    /// discriminated.
    Properties(Relationship),
    /// One version or discriminated instance of a relationship.
    Instance {
        /// The relationship.
        relationship: Relationship,
        /// The version or discriminator, in its key encoding.
        instance: Vec<u8>,
    },
    /// What a blob owner holds, by its owner key; a put's payload is the
    /// owner record.
    BlobOwner(Vec<u8>),
    /// A label a drop ended, which a label created later under its name is
    /// not.
    DroppedLabel(Label),
    /// A type whose implied labels changed: the event's labels before and
    /// after hold it and what it implied before and implies after.
    Scope {
        /// The type label.
        label: Label,
        /// The activation of the change, distinct for every change.
        activation: u64,
    },
}

/// One effect of a committed transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeEvent {
    /// Its place among the transaction's events, the same for every
    /// subscription and delivery.
    pub ordinal: u32,
    /// What it does.
    pub action: ChangeAction,
    /// What it changes; `None` for a target this driver does not know.
    pub target: Option<ChangeTarget>,
    /// The record or properties a put writes, the delta a patch applies, in
    /// the storage encoding.
    pub payload: Vec<u8>,
    /// How a node event reaches the subscription's labels.
    pub delivery: ChangeDelivery,
    /// The labels a node carried before the transaction.
    pub labels_before: Vec<Label>,
    /// The labels a node carries after the event.
    pub labels_after: Vec<Label>,
}

/// One committed transaction, or a fragment of one: the events of it in the
/// ordinal range `[first, end)` the subscription takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeUnit {
    /// Its commit timestamp (HLC microseconds).
    pub commit_ts: u64,
    /// The log index of the entry carrying it.
    pub log_index: u64,
    /// Its place among the transactions of that entry.
    pub unit: u32,
    /// How many events it has in all, taken by the subscription or not.
    pub projected: u32,
    /// The events the subscription takes, in ordinal order.
    pub events: Vec<ChangeEvent>,
    /// The first ordinal this delivery covers.
    pub first: u32,
    /// The ordinal after the last one this delivery covers: `projected`
    /// when it reaches the transaction's end.
    pub end: u32,
}

impl ChangeUnit {
    /// Whether this delivery reaches the end of the transaction.
    #[must_use]
    pub fn completes_transaction(&self) -> bool {
        self.end == self.projected
    }
}

/// Units read from the log and the cursor after them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeBatch {
    /// The units, in log order; none when the entries read carried none.
    pub units: Vec<ChangeUnit>,
    /// Where the stream continues after them.
    pub cursor: Cursor,
}

fn relationship(target: crate::proto::replication::RelationshipTarget) -> Relationship {
    Relationship {
        edge_type: target.edge_type,
        source_id: target.source_id,
        target_id: target.target_id,
    }
}

impl ChangeEvent {
    fn from_proto(event: crate::proto::replication::LogicalChange) -> Self {
        use crate::proto::replication::{
            ChangeAction as ProtoAction, ChangeDelivery as ProtoDelivery,
        };
        let target = event.target.and_then(|target| {
            Some(match target {
                logical_change::Target::Node(node) => ChangeTarget::Node {
                    node_id: node.node_id,
                    valid_from: node.valid_from,
                },
                logical_change::Target::Relationship(r) => {
                    ChangeTarget::Relationship(relationship(r))
                }
                logical_change::Target::Properties(r) => ChangeTarget::Properties(relationship(r)),
                logical_change::Target::Instance(r) => {
                    let instance = r.instance.clone();
                    ChangeTarget::Instance {
                        relationship: relationship(r),
                        instance,
                    }
                }
                // A scope naming no type is not one this driver knows.
                logical_change::Target::Scope(s) => ChangeTarget::Scope {
                    label: Label::from(s.label?),
                    activation: s.activation,
                },
                logical_change::Target::BlobOwner(key) => ChangeTarget::BlobOwner(key),
                logical_change::Target::DroppedLabel(label) => {
                    ChangeTarget::DroppedLabel(Label::from(label))
                }
            })
        });
        Self {
            ordinal: event.ordinal,
            action: match ProtoAction::try_from(event.action) {
                Ok(ProtoAction::Put) => ChangeAction::Put,
                Ok(ProtoAction::Patch) => ChangeAction::Patch,
                Ok(ProtoAction::Remove) => ChangeAction::Remove,
                Ok(ProtoAction::ScopeChange) => ChangeAction::ScopeChange,
                Ok(ProtoAction::Drop) => ChangeAction::Drop,
                Ok(ProtoAction::Unspecified) | Err(_) => ChangeAction::Other(event.action),
            },
            target,
            payload: event.payload,
            delivery: match ProtoDelivery::try_from(event.delivery) {
                Ok(ProtoDelivery::Apply) => ChangeDelivery::Apply,
                Ok(ProtoDelivery::Enter) => ChangeDelivery::Enter,
                Ok(ProtoDelivery::Leave) => ChangeDelivery::Leave,
                Ok(ProtoDelivery::Unspecified) | Err(_) => ChangeDelivery::Other(event.delivery),
            },
            labels_before: event.labels_before.into_iter().map(Label::from).collect(),
            labels_after: event.labels_after.into_iter().map(Label::from).collect(),
        }
    }
}

impl ChangeBatch {
    fn from_proto(batch: crate::proto::replication::ChangeBatch) -> Self {
        Self {
            units: batch
                .units
                .into_iter()
                .map(|unit| ChangeUnit {
                    commit_ts: unit.commit_ts,
                    log_index: unit.log_index,
                    unit: unit.unit,
                    projected: unit.projected,
                    first: unit.first_ordinal,
                    end: unit.end_ordinal,
                    events: unit
                        .events
                        .into_iter()
                        .map(ChangeEvent::from_proto)
                        .collect(),
                })
                .collect(),
            cursor: Cursor(batch.cursor),
        }
    }
}

/// An open subscription. Dropping it ends delivery; the registration stays.
pub struct Subscription {
    link: Arc<SessionLink>,
    request_id: u64,
    frames: mpsc::Receiver<crate::proto::session::ServerFrame>,
    consumer_id: String,
    incarnation: u64,
    window: u32,
    /// Credit granted and not yet used by a batch taken.
    outstanding: u32,
}

impl Subscription {
    pub(crate) async fn open(
        link: Arc<SessionLink>,
        options: &SubscribeOptions,
    ) -> Result<Self, ClientError> {
        // Room for every batch the window allows (a batch passes at least one
        // entry) and the answers around them, so the session never waits on
        // a subscription its caller is not reading.
        let (request_id, mut frames) = link.request(options.window as usize + 4);
        link.send(
            request_id,
            client_frame::Op::Subscribe(Subscribe {
                request: Some(options.request()),
                credit: options.window,
            }),
        )
        .await?;
        let incarnation = match frames.recv().await.and_then(|f| f.event) {
            Some(Event::Subscribed(subscribed)) => subscribed.incarnation,
            Some(Event::Error(error)) => {
                link.forget(request_id);
                return Err(error_status(error).into());
            }
            Some(other) => {
                link.forget(request_id);
                return Err(ClientError::UnexpectedAnswer(format!("{other:?}")));
            }
            None => return Err(ClientError::SessionClosed),
        };
        Ok(Self {
            link,
            request_id,
            frames,
            consumer_id: options.consumer_id.clone(),
            incarnation,
            window: options.window,
            outstanding: options.window,
        })
    }

    /// The incarnation of the registration this subscription reads as: pass
    /// it to [`SubscribeOptions::resume`] to continue it later.
    pub fn incarnation(&self) -> u64 {
        self.incarnation
    }

    /// The next batch, waiting until one arrives. `Ok(None)` when the
    /// session ended.
    ///
    /// # Errors
    ///
    /// The subscription ended on the server: the registration was cancelled
    /// or ended (CONSUMER_TERMINATED), the server no longer holds where it
    /// reads (RETENTION_LOST), or it reached a change it cannot read
    /// (CHANGE_FEED_UNSUPPORTED); the status carries the ErrorInfo reason.
    pub async fn next(&mut self) -> Result<Option<ChangeBatch>, ClientError> {
        loop {
            let Some(frame) = self.frames.recv().await else {
                return Ok(None);
            };
            match frame.event {
                Some(Event::ChangeEvents(frame)) => {
                    // The server reads no more than the credit granted.
                    self.outstanding =
                        self.outstanding.checked_sub(frame.entries).ok_or_else(|| {
                            ClientError::UnexpectedAnswer(format!(
                                "a batch of {} entries with {} granted",
                                frame.entries, self.outstanding
                            ))
                        })?;
                    // Applied entries are waiting: grant at once to drain
                    // them. Otherwise top the window up once half is used.
                    if frame.more || self.outstanding <= self.window / 2 {
                        self.grant().await?;
                    }
                    let batch = frame.batch.ok_or_else(|| {
                        ClientError::UnexpectedAnswer("a change frame without a batch".into())
                    })?;
                    return Ok(Some(ChangeBatch::from_proto(batch)));
                }
                Some(Event::Error(error)) => return Err(error_status(error).into()),
                // Nothing else is answered on the subscription's request.
                _ => {}
            }
        }
    }

    /// Record that the caller holds every unit before `cursor` durably: the
    /// server may stop keeping them, and a resumed subscription starts there.
    ///
    /// # Errors
    ///
    /// The server refused it: a cursor past what it applied or of another
    /// history, or an ended registration.
    pub async fn acknowledge(&self, cursor: &Cursor) -> Result<(), ClientError> {
        let answer = self
            .link
            .call(client_frame::Op::Acknowledge(
                AcknowledgeSubscriptionRequest {
                    consumer_id: self.consumer_id.clone(),
                    incarnation: self.incarnation,
                    cursor: cursor.0.clone(),
                },
            ))
            .await?;
        match answer.event {
            Some(Event::Acknowledged(_)) => Ok(()),
            Some(Event::Error(error)) => Err(error_status(error).into()),
            other => Err(ClientError::UnexpectedAnswer(format!("{other:?}"))),
        }
    }

    /// End the registration on the server: it stops keeping history for the
    /// consumer, and its incarnation cannot be resumed.
    ///
    /// # Errors
    ///
    /// The server refused it, for an already ended registration.
    pub async fn cancel_registration(self) -> Result<(), ClientError> {
        let answer = self
            .link
            .call(client_frame::Op::CancelSubscription(
                CancelSubscriptionRequest {
                    consumer_id: self.consumer_id.clone(),
                    incarnation: self.incarnation,
                },
            ))
            .await?;
        match answer.event {
            Some(Event::SubscriptionCancelled(_)) => Ok(()),
            Some(Event::Error(error)) => Err(error_status(error).into()),
            other => Err(ClientError::UnexpectedAnswer(format!("{other:?}"))),
        }
    }

    /// Grant the server the window back.
    async fn grant(&mut self) -> Result<(), ClientError> {
        let entries = self.window - self.outstanding;
        if entries == 0 {
            return Ok(());
        }
        // Answered only when refused, and a refusal of a credit for an open
        // subscription cannot happen; nobody waits for it.
        self.link
            .send(
                self.link.fresh_id(),
                client_frame::Op::Credit(Credit {
                    target_request_id: self.request_id,
                    events: entries,
                }),
            )
            .await?;
        self.outstanding = self.window;
        Ok(())
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.link.forget(self.request_id);
        // Ends delivery on the server; the registration stays. A full
        // outbound queue or an ended session leaves nothing to stop.
        let _ = self.link.try_send(
            self.link.fresh_id(),
            client_frame::Op::Cancel(Cancel {
                target_request_id: self.request_id,
            }),
        );
    }
}
