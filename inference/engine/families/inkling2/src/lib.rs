//! Execution schedule for Inkling 2 Mini's physically shared loop core.
//!
//! The decoder contract already represents one logical visit per block. A
//! family can therefore expand the physical layer range into logical visits,
//! point each visit at its physical tensor names, and let the state layout
//! assign a separate state slot to each visit. This crate currently provides
//! that schedule foundation; it does not yet build an executable family
//! definition for Inkling's additional operators.

use std::{error, fmt};

#[cfg(not(feature = "standalone"))]
mod inventory;
#[cfg(not(feature = "standalone"))]
mod metadata;

#[cfg(not(feature = "standalone"))]
pub use inventory::{TensorInventory, inspect_tensors};
#[cfg(not(feature = "standalone"))]
pub use metadata::{ARCHITECTURE, InklingMetadata, InspectionError, inspect_metadata};

/// One visit to a physical layer. `logical_slot` is also the state identity:
/// the state layout owns one cache slot for every block in execution order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayerVisit {
    pub logical_slot: u32,
    pub physical_layer: u32,
    /// `None` for the one-time prelude and coda; otherwise the zero-based
    /// visit within the repeated core.
    pub loop_pass: Option<u32>,
}

/// A fixed-count loop schedule expanded from physical layer metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopSchedule {
    core_start: u32,
    core_end: u32,
    core_visits: u32,
    logical_visit_count: u32,
}

impl LoopSchedule {
    /// Expands a physical stack into the logical order: prelude, each core
    /// pass, then coda. The core boundaries must align to attention-residual
    /// blocks, and a fixed loop must have at least two passes.
    pub fn new(
        physical_layers: u32,
        core_start: u32,
        core_end: u32,
        passes: u32,
        attention_residual_block_size: u32,
    ) -> Result<Self, ScheduleError> {
        if physical_layers == 0
            || core_start == 0
            || core_start >= core_end
            || core_end > physical_layers
            || passes < 2
            || attention_residual_block_size == 0
            || !core_start.is_multiple_of(attention_residual_block_size)
            || !core_end.is_multiple_of(attention_residual_block_size)
        {
            return Err(ScheduleError::InvalidGeometry);
        }

        let core_layers = core_end - core_start;
        let extra_visits = core_layers
            .checked_mul(passes - 1)
            .ok_or(ScheduleError::TooManyVisits)?;
        let visit_count = physical_layers
            .checked_add(extra_visits)
            .ok_or(ScheduleError::TooManyVisits)?;
        let core_visits = core_layers
            .checked_mul(passes)
            .ok_or(ScheduleError::TooManyVisits)?;
        Ok(Self {
            core_start,
            core_end,
            core_visits,
            logical_visit_count: visit_count,
        })
    }

    /// The number of logical visits and independent state slots.
    pub fn logical_visit_count(&self) -> u32 {
        self.logical_visit_count
    }

    /// The visit at `logical_slot`, or `None` when it is out of range.
    pub fn visit_at(&self, logical_slot: u32) -> Option<LayerVisit> {
        if logical_slot >= self.logical_visit_count {
            return None;
        }
        if logical_slot < self.core_start {
            return Some(LayerVisit {
                logical_slot,
                physical_layer: logical_slot,
                loop_pass: None,
            });
        }

        let core_offset = logical_slot - self.core_start;
        if core_offset < self.core_visits {
            let core_layers = self.core_end - self.core_start;
            return Some(LayerVisit {
                logical_slot,
                physical_layer: self.core_start + core_offset % core_layers,
                loop_pass: Some(core_offset / core_layers),
            });
        }

        Some(LayerVisit {
            logical_slot,
            physical_layer: self.core_end + core_offset - self.core_visits,
            loop_pass: None,
        })
    }

    /// Logical visits in execution order. Each visit's logical slot is its
    /// state identity, while its physical layer identifies shared weights.
    pub fn visits(&self) -> impl Iterator<Item = LayerVisit> + '_ {
        (0..self.logical_visit_count).map(|slot| {
            self.visit_at(slot)
                .expect("slot is within the validated schedule")
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    InvalidGeometry,
    TooManyVisits,
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry => formatter.write_str("invalid fixed loop geometry"),
            Self::TooManyVisits => formatter.write_str("logical loop visit count overflows"),
        }
    }
}

impl error::Error for ScheduleError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mini_schedule_has_48_visits_and_three_passes_of_each_core_layer() {
        let schedule = LoopSchedule::new(24, 6, 18, 3, 6).unwrap();
        assert_eq!(schedule.logical_visit_count(), 48);
        assert_eq!(
            schedule
                .visits()
                .map(|visit| visit.physical_layer)
                .collect::<Vec<_>>(),
            (0..6)
                .chain(6..18)
                .chain(6..18)
                .chain(6..18)
                .chain(18..24)
                .collect::<Vec<_>>()
        );
        assert!(
            schedule
                .visits()
                .enumerate()
                .all(|(slot, visit)| { visit.logical_slot == slot as u32 })
        );
        for layer in 6..18 {
            assert_eq!(
                schedule
                    .visits()
                    .filter(|visit| visit.physical_layer == layer)
                    .map(|visit| visit.loop_pass)
                    .collect::<Vec<_>>(),
                [Some(0), Some(1), Some(2)]
            );
        }
        assert_eq!(schedule.visit_at(48), None);
    }

    #[test]
    fn rejects_invalid_boundaries_passes_and_alignment() {
        assert_eq!(
            LoopSchedule::new(24, 0, 18, 3, 1),
            Err(ScheduleError::InvalidGeometry)
        );
        assert_eq!(
            LoopSchedule::new(24, 6, 18, 1, 1),
            Err(ScheduleError::InvalidGeometry)
        );
        assert_eq!(
            LoopSchedule::new(24, 6, 25, 3, 1),
            Err(ScheduleError::InvalidGeometry)
        );
        assert_eq!(
            LoopSchedule::new(24, 6, 17, 3, 2),
            Err(ScheduleError::InvalidGeometry)
        );
        assert_eq!(
            LoopSchedule::new(u32::MAX, 1, u32::MAX, u32::MAX, 1),
            Err(ScheduleError::TooManyVisits)
        );
    }
}

#[cfg(all(test, not(feature = "standalone")))]
mod state_layout_tests {
    use super::*;
    use magnitude_family_contracts::{
        ActivationDType, Attention, AttentionGate, Block, Decoder, EmbeddingScale, EntryForm,
        ExitForm, ExitNorm, FamilyId, HeadNorm, HistoryRows, HistorySource, InputNorm,
        InputSemantics, MediaRowAttention, ModelDefinition, Operator, OutputForm, OutputProjection,
        ProjectedRows, QueryKeys, QueryProjection, ResidualForm, RmsNorm, Rotary, RowRounding,
        Sublayer, TokenDomain, ValueNorm, ValueSource, WeightDescriptor,
    };
    use magnitude_state::{KvCodec, LayerRef, ModelStateLayout};

    fn weight(name: impl Into<String>, shape: &[u64]) -> WeightDescriptor {
        WeightDescriptor::stored(name, shape)
    }

    fn attention_block(physical_layer: u32) -> Block {
        let prefix = format!("blk.{physical_layer}.");
        let name = |suffix: &str| format!("{prefix}{suffix}");
        let attention = Attention {
            heads: 1,
            kv_heads: 1,
            width: 2,
            query: QueryProjection::Full(weight(name("attn_q.weight"), &[2, 2])),
            query_keys: QueryKeys::Direct,
            gate: AttentionGate::None,
            query_norm: HeadNorm::None,
            history: vec![HistorySource::visible(HistoryRows::Own(
                magnitude_family_contracts::RowProducer::Projected(ProjectedRows {
                    key: weight(name("attn_k.weight"), &[2, 2]),
                    value: ValueSource::Projected(weight(name("attn_v.weight"), &[2, 2])),
                    key_norm: HeadNorm::None,
                    value_norm: ValueNorm::None,
                    domain: TokenDomain::Token,
                    rounding: RowRounding::None,
                }),
            ))],
            rotary: Rotary::None,
            scale: 2.0_f64.sqrt().recip(),
            media_rows: MediaRowAttention::Causal,
            sinks: None,
            output: OutputProjection::Full(weight(name("attn_out.weight"), &[2, 2])),
        };
        Block {
            sublayers: vec![Sublayer {
                input: InputNorm::Rms(RmsNorm {
                    weight: weight(name("attn_norm.weight"), &[2]),
                    epsilon: 1e-6,
                }),
                op: Operator::Attention(Box::new(attention)),
                output: OutputForm::Residual,
            }],
        }
    }

    fn definition(blocks: Vec<Block>) -> ModelDefinition {
        let embedding = weight("token_embd.weight", &[8, 2]);
        ModelDefinition {
            family: FamilyId("inkling2_loop".into()),
            artifact_identity: magnitude_artifacts::PackageIdentity {
                target: magnitude_artifacts::ArtifactIdentity([0; 32]),
                projector: None,
            },
            inputs: InputSemantics { coordinate_axes: 1 },
            decoder: Decoder {
                activation_dtype: ActivationDType::BF16,
                hidden: 2,
                vocabulary: 8,
                context_limit: 64,
                residual: ResidualForm::Single,
                entry: EntryForm {
                    embedding: embedding.clone(),
                    scale: EmbeddingScale::Unit,
                    norm: None,
                    per_layer: None,
                },
                blocks,
                exit: ExitForm {
                    norm: ExitNorm::Rms(RmsNorm {
                        weight: weight("output_norm.weight", &[2]),
                        epsilon: 1e-6,
                    }),
                    output: weight("output.weight", &[8, 2]),
                    softcap: None,
                },
            },
            head: None,
            vision: None,
            draft: None,
        }
    }

    #[test]
    fn expanded_decoder_reuses_physical_weights_and_allocates_per_visit_history() {
        let schedule = LoopSchedule::new(24, 6, 18, 3, 6).unwrap();
        let blocks = schedule
            .visits()
            .map(|visit| attention_block(visit.physical_layer))
            .collect::<Vec<_>>();

        let core_weight = |block: &Block| match &block.sublayers[0].op {
            Operator::Attention(attention) => match &attention.query {
                QueryProjection::Full(weight) => &weight.name,
                _ => unreachable!("dummy attention uses a full query"),
            },
            _ => unreachable!("dummy block contains attention"),
        };
        assert_eq!(core_weight(&blocks[6]), core_weight(&blocks[18]));
        assert_eq!(core_weight(&blocks[18]), core_weight(&blocks[30]));

        let model = definition(blocks);
        let state = ModelStateLayout::derive(&model.decoder, &[], KvCodec::Dense, 0).unwrap();
        let [magnitude_state::HistoryDomainLayout::Token { components }] =
            state.target_history.as_slice()
        else {
            panic!("dummy decoder should have one token history domain");
        };
        assert_eq!(components.len(), 48);
        assert_eq!(
            components
                .iter()
                .map(|component| component.layer)
                .collect::<Vec<_>>(),
            (0..48).map(LayerRef::Target).collect::<Vec<_>>()
        );
        for layer in [6, 18, 30] {
            assert!(
                components
                    .iter()
                    .any(|component| component.layer == LayerRef::Target(layer))
            );
        }
    }
}
