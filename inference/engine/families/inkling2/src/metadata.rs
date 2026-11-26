//! Strict, header-only interpretation of Inkling 2 Mini's GGUF layout.
//!
//! This records the architecture's real per-layer geometry and fixed loop
//! schedule before a numerical definition exists. It deliberately does not
//! claim that the current executor can run the model.

use crate::{LoopSchedule, ScheduleError};
use magnitude_artifacts::gguf::{Directory, Kind, Value};
use magnitude_family_common::{HeaderError, Metadata};
use std::{error, fmt};

pub const ARCHITECTURE: &str = "inkling2_loop";

/// The dimensions and execution metadata that shape an Inkling decoder.
#[derive(Clone, Debug, PartialEq)]
pub struct InklingMetadata {
    pub physical_layers: u32,
    pub hidden_size: u32,
    pub context_length: u64,
    pub feed_forward_size: u32,
    pub query_heads: Vec<u32>,
    pub key_value_heads: Vec<u32>,
    pub key_length: u32,
    pub value_length: u32,
    pub sliding_key_length: u32,
    pub sliding_value_length: u32,
    pub sliding_window: u32,
    /// `true` for a local window and `false` for global attention.
    pub local_attention: Vec<bool>,
    pub experts: u32,
    pub selected_experts: u32,
    pub expert_feed_forward_size: u32,
    pub loop_start: u32,
    pub loop_end: u32,
    pub loop_passes: u32,
    pub loop_gate_rank: u32,
    pub schedule: LoopSchedule,
    pub attention_residual_block_size: u32,
    pub convolution_cache_length: u32,
    pub padded_vocabulary: u32,
    pub tokenizer_vocabulary: u32,
    pub rope_dimensions: u32,
    pub rms_epsilon: f64,
    pub expert_weights_scale: f64,
    pub situ_beta: f64,
    pub situ_linear_beta: f64,
    pub final_logit_softcap: f64,
    pub relative_length: u32,
    pub relative_extent: u32,
    pub qk_clip: f64,
    pub relative_clip: f64,
    pub log_scale_floor: u32,
    pub log_scale_alpha: f64,
}

/// Why a GGUF cannot be described as this fixed Inkling architecture.
#[derive(Clone, Debug, PartialEq)]
pub enum InspectionError {
    Header(HeaderError),
    Architecture(String),
    Invalid(String),
}

impl fmt::Display for InspectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Header(error) => error.fmt(formatter),
            Self::Architecture(architecture) => write!(
                formatter,
                "expected GGUF architecture {ARCHITECTURE:?}, received {architecture:?}"
            ),
            Self::Invalid(detail) => formatter.write_str(detail),
        }
    }
}

impl error::Error for InspectionError {}

impl From<HeaderError> for InspectionError {
    fn from(error: HeaderError) -> Self {
        Self::Header(error)
    }
}

fn invalid(detail: impl Into<String>) -> InspectionError {
    InspectionError::Invalid(detail.into())
}

fn narrow(key: &str, value: u64) -> Result<u32, InspectionError> {
    u32::try_from(value).map_err(|_| invalid(format!("{ARCHITECTURE}.{key} exceeds u32")))
}

fn narrow_layers(key: &str, values: Vec<u64>) -> Result<Vec<u32>, InspectionError> {
    values.into_iter().map(|value| narrow(key, value)).collect()
}

/// Read and cross-check the architecture metadata needed to plan the decoder.
/// Tensor payloads are never accessed.
pub fn inspect_metadata(directory: &Directory) -> Result<InklingMetadata, InspectionError> {
    let architecture = directory
        .value("general.architecture")
        .and_then(Value::string)
        .ok_or_else(|| {
            InspectionError::Header(HeaderError::MissingMetadata("general.architecture".into()))
        })?;
    if architecture != ARCHITECTURE {
        return Err(InspectionError::Architecture(architecture.to_owned()));
    }

    let metadata = Metadata::new(directory, ARCHITECTURE);
    let physical_layers = metadata.bounded_block_count("block_count")?;
    let physical_layers_u32 = narrow("block_count", physical_layers)?;
    let hidden_size = narrow("embedding_length", metadata.integer("embedding_length")?)?;
    let context_length = metadata.integer("context_length")?;
    let feed_forward_size = narrow(
        "feed_forward_length",
        metadata.integer("feed_forward_length")?,
    )?;
    let query_heads = narrow_layers(
        "attention.head_count",
        metadata.per_layer("attention.head_count", physical_layers)?,
    )?;
    let key_value_heads = narrow_layers(
        "attention.head_count_kv",
        metadata.per_layer("attention.head_count_kv", physical_layers)?,
    )?;
    let rms_epsilon = metadata.positive_number("attention.layer_norm_rms_epsilon")?;
    let experts = narrow("expert_count", metadata.integer("expert_count")?)?;
    let selected_experts = narrow("expert_used_count", metadata.integer("expert_used_count")?)?;
    let key_length = narrow(
        "attention.key_length",
        metadata.integer("attention.key_length")?,
    )?;
    let value_length = narrow(
        "attention.value_length",
        metadata.integer("attention.value_length")?,
    )?;
    let passes = narrow("loop.count", metadata.integer("loop.count")?)?;
    let loop_start = narrow("loop.start", metadata.count("loop.start")?)?;
    let loop_end = narrow("loop.end", metadata.integer("loop.end")?)?;
    let loop_gate_rank = narrow("loop.gate_rank", metadata.integer("loop.gate_rank")?)?;
    let padded_vocabulary = narrow(
        "vocab_size",
        metadata.vocab_size(metadata.integer("vocab_size")?)?,
    )?;
    let tokenizer_vocabulary = narrow(
        "vocab_size_unpadded",
        metadata.integer("vocab_size_unpadded")?,
    )?;
    let expert_feed_forward_size = narrow(
        "expert_feed_forward_length",
        metadata.integer("expert_feed_forward_length")?,
    )?;
    let sliding_key_length = narrow(
        "attention.key_length_swa",
        metadata.integer("attention.key_length_swa")?,
    )?;
    let sliding_value_length = narrow(
        "attention.value_length_swa",
        metadata.integer("attention.value_length_swa")?,
    )?;
    let rope_dimensions = narrow(
        "rope.dimension_count",
        metadata.count("rope.dimension_count")?,
    )?;
    let sliding_window = narrow(
        "attention.sliding_window",
        metadata.integer("attention.sliding_window")?,
    )?;
    let local_attention = metadata.flags("attention.sliding_window_pattern", physical_layers)?;
    let convolution_cache_length =
        narrow("shortconv.l_cache", metadata.integer("shortconv.l_cache")?)?;
    let expert_weights_scale = metadata.positive_number("expert_weights_scale")?;
    let situ_beta = metadata.positive_number("activation.situ_beta")?;
    let situ_linear_beta = metadata.positive_number("activation.situ_linear_beta")?;
    let attention_residual_block_size = narrow(
        "attn_res.block_size",
        metadata.integer("attn_res.block_size")?,
    )?;
    let final_logit_softcap = metadata.positive_number("final_logit_softcapping")?;
    let relative_length = narrow(
        "attention.relative_length",
        metadata.integer("attention.relative_length")?,
    )?;
    let relative_extent = narrow(
        "attention.relative_extent",
        metadata.integer("attention.relative_extent")?,
    )?;
    let qk_clip = metadata.positive_number("attention.qk_clip")?;
    let relative_clip = metadata.positive_number("attention.relative_clip")?;
    let log_scale_floor = narrow(
        "attention.log_scale_floor",
        metadata.integer("attention.log_scale_floor")?,
    )?;
    let log_scale_alpha = metadata.positive_number("attention.log_scale_alpha")?;
    if metadata.flag("loop.varying")? {
        return Err(invalid(
            "Inkling loop.varying must be false for a fixed schedule",
        ));
    }

    let token_count = match directory.value("tokenizer.ggml.tokens") {
        Some(Value::Array(tokens)) if tokens.kind() == Kind::String => tokens.len() as u64,
        Some(_) => {
            return Err(InspectionError::Header(HeaderError::MetadataType {
                key: "tokenizer.ggml.tokens".into(),
                expected: "an array of strings",
            }));
        }
        None => {
            return Err(InspectionError::Header(HeaderError::MissingMetadata(
                "tokenizer.ggml.tokens".into(),
            )));
        }
    };
    if token_count != u64::from(tokenizer_vocabulary) {
        return Err(invalid(format!(
            "tokenizer.ggml.tokens has {token_count} entries but {ARCHITECTURE}.vocab_size_unpadded is {tokenizer_vocabulary}"
        )));
    }
    if u64::from(tokenizer_vocabulary) > u64::from(padded_vocabulary) {
        return Err(invalid(format!(
            "{ARCHITECTURE}.vocab_size_unpadded exceeds vocab_size"
        )));
    }
    if experts == 0 || selected_experts == 0 || selected_experts > experts {
        return Err(invalid(format!(
            "{ARCHITECTURE}.expert_used_count must be within 1..=expert_count"
        )));
    }
    if context_length == 0
        || hidden_size == 0
        || feed_forward_size == 0
        || expert_feed_forward_size == 0
        || key_length == 0
        || value_length == 0
        || sliding_key_length == 0
        || sliding_value_length == 0
        || sliding_window == 0
        || sliding_window > context_length
        || convolution_cache_length < 2
        || convolution_cache_length > 64
        || padded_vocabulary == 0
        || tokenizer_vocabulary == 0
        || loop_gate_rank == 0
        || loop_gate_rank > hidden_size
        || relative_length == 0
        || relative_extent == 0
        || relative_length > relative_extent
        || log_scale_floor == 0
        || rope_dimensions != 0
    {
        return Err(invalid(format!(
            "{ARCHITECTURE} dimensions or non-RoPE position metadata are inconsistent"
        )));
    }
    for (layer, (&query, &key_value)) in query_heads.iter().zip(&key_value_heads).enumerate() {
        if key_value == 0 || query == 0 || query % key_value != 0 {
            return Err(invalid(format!(
                "{ARCHITECTURE} attention head counts are invalid at layer {layer}"
            )));
        }
    }
    if local_attention.iter().all(|local| *local) || local_attention.iter().all(|local| !local) {
        return Err(invalid(format!(
            "{ARCHITECTURE} must include both local and global attention layers"
        )));
    }

    let schedule = LoopSchedule::new(
        physical_layers_u32,
        loop_start,
        loop_end,
        passes,
        attention_residual_block_size,
    )
    .map_err(|error| match error {
        ScheduleError::InvalidGeometry => {
            invalid(format!("{ARCHITECTURE} loop metadata has invalid geometry"))
        }
        ScheduleError::TooManyVisits => invalid(format!(
            "{ARCHITECTURE} loop metadata overflows the visit count"
        )),
    })?;

    metadata.finish()?;
    Ok(InklingMetadata {
        physical_layers: physical_layers_u32,
        hidden_size,
        context_length,
        feed_forward_size,
        query_heads,
        key_value_heads,
        key_length,
        value_length,
        sliding_key_length,
        sliding_value_length,
        sliding_window,
        local_attention,
        experts,
        selected_experts,
        expert_feed_forward_size,
        loop_start,
        loop_end,
        loop_passes: passes,
        loop_gate_rank,
        schedule,
        attention_residual_block_size,
        convolution_cache_length,
        padded_vocabulary,
        tokenizer_vocabulary,
        rope_dimensions,
        rms_epsilon,
        expert_weights_scale,
        situ_beta,
        situ_linear_beta,
        final_logit_softcap,
        relative_length,
        relative_extent,
        qk_clip,
        relative_clip,
        log_scale_floor,
        log_scale_alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_artifacts::gguf::{
        Array, ByteOrder, Encoding, Metadata as Entry, Scalar, TensorDescriptor,
    };

    fn integer(key: &str, value: u64) -> Entry {
        Entry {
            name: format!("{ARCHITECTURE}.{key}"),
            value: Value::Scalar(Scalar::Unsigned(value)),
        }
    }

    fn number(key: &str, value: f64) -> Entry {
        Entry {
            name: format!("{ARCHITECTURE}.{key}"),
            value: Value::Scalar(Scalar::Float(value)),
        }
    }

    fn metadata() -> Vec<Entry> {
        let mut metadata = vec![
            Entry {
                name: "general.architecture".into(),
                value: Value::Scalar(Scalar::String(ARCHITECTURE.into())),
            },
            integer("block_count", 24),
            integer("context_length", 1_048_576),
            integer("embedding_length", 2304),
            integer("feed_forward_length", 1152),
            integer("expert_count", 128),
            integer("expert_used_count", 8),
            integer("attention.key_length", 128),
            integer("attention.value_length", 128),
            integer("loop.count", 3),
            integer("loop.start", 6),
            integer("loop.end", 18),
            integer("loop.gate_rank", 64),
            integer("vocab_size", 32),
            integer("vocab_size_unpadded", 32),
            integer("expert_feed_forward_length", 1152),
            integer("attention.key_length_swa", 128),
            integer("attention.value_length_swa", 128),
            integer("rope.dimension_count", 0),
            integer("attention.sliding_window", 1024),
            integer("shortconv.l_cache", 4),
            integer("attn_res.block_size", 6),
            integer("attention.relative_length", 16),
            integer("attention.relative_extent", 1024),
            integer("attention.log_scale_floor", 8192),
        ];
        for (key, value) in [
            ("attention.head_count", 24_u64),
            ("attention.head_count_kv", 8_u64),
        ] {
            metadata.push(Entry {
                name: format!("{ARCHITECTURE}.{key}"),
                value: Value::Array(Array::from(vec![value; 24])),
            });
        }
        let mut sliding = vec![true; 24];
        for layer in [5, 11, 17, 23] {
            sliding[layer] = false;
        }
        metadata.push(Entry {
            name: format!("{ARCHITECTURE}.attention.sliding_window_pattern"),
            value: Value::Array(Array::from(sliding)),
        });
        metadata.push(Entry {
            name: format!("{ARCHITECTURE}.loop.varying"),
            value: Value::Scalar(Scalar::Bool(false)),
        });
        metadata.extend([
            number("attention.layer_norm_rms_epsilon", 1e-6),
            number("expert_weights_scale", 8.0),
            number("activation.situ_beta", 4.0),
            number("activation.situ_linear_beta", 25.0),
            number("final_logit_softcapping", 30.0),
            number("attention.qk_clip", 22.627416610717773),
            number("attention.relative_clip", 8.0),
            number("attention.log_scale_alpha", 0.1),
        ]);
        metadata.push(Entry {
            name: "tokenizer.ggml.tokens".into(),
            value: Value::Array(Array::from(
                (0..32)
                    .map(|index| format!("token-{index}"))
                    .collect::<magnitude_artifacts::gguf::Strings>(),
            )),
        });
        metadata
    }

    fn directory() -> Directory {
        Directory {
            version: 3,
            byte_order: ByteOrder::Little,
            alignment: 32,
            data_offset: 0,
            metadata: metadata(),
            tensors: (0..24)
                .map(|layer| TensorDescriptor {
                    name: format!("blk.{layer}.sentinel"),
                    shape: vec![1],
                    encoding: Encoding::F32,
                    offset: 0,
                    nbytes: 4,
                })
                .collect(),
        }
    }

    #[test]
    fn reads_downloaded_models_metadata_profile_and_48_visit_schedule() {
        let config = inspect_metadata(&directory()).unwrap();
        assert_eq!(config.physical_layers, 24);
        assert_eq!(config.hidden_size, 2304);
        assert_eq!(config.context_length, 1_048_576);
        assert_eq!(config.experts, 128);
        assert_eq!(config.selected_experts, 8);
        assert_eq!(config.schedule.logical_visit_count(), 48);
        assert_eq!(
            config
                .schedule
                .visits()
                .filter(|visit| visit.loop_pass.is_some())
                .count(),
            36
        );
        assert_eq!(
            config
                .local_attention
                .iter()
                .filter(|local| !**local)
                .count(),
            4
        );
        assert_eq!(config.tokenizer_vocabulary, 32);
    }

    #[test]
    fn rejects_unread_architecture_keys_and_loop_geometry_that_breaks_residual_groups() {
        let mut directory = directory();
        directory.metadata.push(integer("new_operator_mode", 1));
        assert!(matches!(
            inspect_metadata(&directory),
            Err(InspectionError::Header(HeaderError::UnknownMetadata(_)))
        ));

        let mut directory = directory();
        directory
            .metadata
            .iter_mut()
            .find(|entry| entry.name == format!("{ARCHITECTURE}.loop.end"))
            .unwrap()
            .value = Value::Scalar(Scalar::Unsigned(17));
        assert!(matches!(
            inspect_metadata(&directory),
            Err(InspectionError::Invalid(detail)) if detail.contains("loop metadata")
        ));
    }
}
