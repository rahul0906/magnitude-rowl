//! Header-only inventory checks for the physical tensors in an Inkling GGUF.
//!
//! This checks the stable physical storage once, before a family expands the
//! layers into logical visits. It never loads tensor bytes or duplicates a
//! physical tensor for another loop pass.

use crate::{InklingMetadata, InspectionError};
use magnitude_artifacts::gguf::{Directory, TensorDescriptor};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInventory {
    pub total_tensors: u32,
    /// Number of stored tensors under each `blk.N.` physical layer prefix.
    pub tensors_per_physical_layer: Vec<u32>,
    /// Number of tensors of each GGML encoding in the GGUF directory.
    pub encodings: BTreeMap<&'static str, u32>,
}

fn invalid(detail: impl Into<String>) -> InspectionError {
    InspectionError::Invalid(detail.into())
}

fn product(left: u32, right: u32, role: &str) -> Result<u64, InspectionError> {
    u64::from(left)
        .checked_mul(u64::from(right))
        .ok_or_else(|| invalid(format!("Inkling {role} tensor dimension overflows")))
}

fn require_shape(directory: &Directory, name: &str, shape: &[u64]) -> Result<(), InspectionError> {
    let tensor = directory
        .tensor(name)
        .ok_or_else(|| invalid(format!("Inkling GGUF is missing required tensor {name:?}")))?;
    if tensor.shape != shape {
        return Err(invalid(format!(
            "Inkling tensor {name:?} has shape {:?}, expected {shape:?}",
            tensor.shape
        )));
    }
    Ok(())
}

fn expected_tensor_names(metadata: &InklingMetadata) -> BTreeSet<String> {
    let mut names = [
        "token_embd.weight",
        "token_embd_norm.weight",
        "output.weight",
        "output_norm.weight",
        "output_res_query.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();

    for layer in 0..metadata.physical_layers {
        let prefix = format!("blk.{layer}.");
        for suffix in [
            "attn_q.weight",
            "attn_gate.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_q_clip.weight",
            "attn_rel.weight",
            "attn_rel_bias.weight",
            "attn_rel_clip.weight",
            "attn_sink_proj.weight",
            "attn_k_conv.weight",
            "attn_v_conv.weight",
            "attn_post_conv.weight",
            "ffn_norm.weight",
            "ffn_post_conv.weight",
            "ffn_gate_inp.weight",
            "ffn_gate_up_exps.weight",
            "ffn_down_exps.weight",
            "exp_probs_b.bias",
            "ffn_res_query.weight",
        ] {
            names.insert(format!("{prefix}{suffix}"));
        }
        if layer > 0 {
            names.insert(format!("{prefix}attn_res_query.weight"));
        }
        if metadata.local_attention[layer as usize] {
            names.insert(format!("{prefix}attn_norm.weight"));
        } else {
            for suffix in [
                "attn_q_inp_norm.weight",
                "attn_k_inp_norm.weight",
                "attn_v_inp_norm.weight",
                "attn_k_conv_proj.weight",
                "attn_v_conv_proj.weight",
            ] {
                names.insert(format!("{prefix}{suffix}"));
            }
        }
        if (metadata.loop_start..metadata.loop_end).contains(&layer) {
            names.insert(format!("{prefix}loop_msg_query.weight"));
            for controller in ["q", "k", "v", "ffn"] {
                names.insert(format!("{prefix}loop_{controller}_ctrl_down.weight"));
                for projection in ["forget", "gain"] {
                    names.insert(format!(
                        "{prefix}loop_{controller}_ctrl_{projection}.weight"
                    ));
                }
            }
        }
    }
    names
}

/// Verify physical layer numbering and the model's routed and loop-controller
/// tensor geometry. Quantization support remains a separate backend decision.
pub fn inspect_tensors(
    directory: &Directory,
    metadata: &InklingMetadata,
) -> Result<TensorInventory, InspectionError> {
    if metadata.physical_layers as usize > directory.tensors.len() {
        return Err(invalid(format!(
            "Inkling block_count {} exceeds its {} stored tensors",
            metadata.physical_layers,
            directory.tensors.len()
        )));
    }
    let mut tensors_per_physical_layer = vec![0_u32; metadata.physical_layers as usize];
    let mut encodings = BTreeMap::new();
    let mut loop_tensors_per_layer = vec![0_u32; metadata.physical_layers as usize];
    for tensor in &directory.tensors {
        let encoding_count = encodings.entry(tensor.encoding.ggml_name()).or_insert(0);
        *encoding_count = encoding_count
            .checked_add(1)
            .ok_or_else(|| invalid("Inkling encoding count overflows"))?;
        if let Some(rest) = tensor.name.strip_prefix("blk.") {
            let (layer, suffix) = rest.split_once('.').ok_or_else(|| {
                invalid(format!(
                    "Inkling tensor name {:?} has no layer suffix",
                    tensor.name
                ))
            })?;
            let layer = layer.parse::<u32>().map_err(|_| {
                invalid(format!(
                    "Inkling tensor name {:?} has an invalid layer",
                    tensor.name
                ))
            })?;
            let count = tensors_per_physical_layer
                .get_mut(layer as usize)
                .ok_or_else(|| {
                    invalid(format!(
                        "Inkling tensor {:?} exceeds block_count",
                        tensor.name
                    ))
                })?;
            *count = count
                .checked_add(1)
                .ok_or_else(|| invalid("Inkling per-layer tensor count overflows"))?;
            if suffix.starts_with("loop_") {
                loop_tensors_per_layer[layer as usize] = loop_tensors_per_layer[layer as usize]
                    .checked_add(1)
                    .ok_or_else(|| invalid("Inkling loop tensor count overflows"))?;
            }
        }
    }
    if let Some(layer) = tensors_per_physical_layer
        .iter()
        .position(|count| *count == 0)
    {
        return Err(invalid(format!(
            "Inkling physical layer {layer} has no stored tensors"
        )));
    }

    let hidden = u64::from(metadata.hidden_size);
    let padded_vocabulary = u64::from(metadata.padded_vocabulary);
    require_shape(directory, "token_embd.weight", &[padded_vocabulary, hidden])?;
    require_shape(directory, "token_embd_norm.weight", &[hidden])?;
    require_shape(directory, "output.weight", &[padded_vocabulary, hidden])?;
    require_shape(directory, "output_norm.weight", &[hidden])?;
    require_shape(directory, "output_res_query.weight", &[hidden])?;

    for layer in 0..metadata.physical_layers {
        let layer_index = layer as usize;
        let query_heads = metadata.query_heads[layer_index];
        let kv_heads = metadata.key_value_heads[layer_index];
        let is_local = metadata.local_attention[layer_index];
        let key_length = if is_local {
            metadata.sliding_key_length
        } else {
            metadata.key_length
        };
        let value_length = if is_local {
            metadata.sliding_value_length
        } else {
            metadata.value_length
        };
        let query_width = product(query_heads, key_length, "query")?;
        let key_width = product(kv_heads, key_length, "key")?;
        let value_width = product(kv_heads, value_length, "value")?;
        let output_width = product(query_heads, value_length, "attention output")?;
        let layer_name = |suffix: &str| format!("blk.{layer}.{suffix}");
        require_shape(
            directory,
            &layer_name("attn_q.weight"),
            &[query_width, hidden],
        )?;
        require_shape(
            directory,
            &layer_name("attn_gate.weight"),
            &[query_width, hidden],
        )?;
        require_shape(
            directory,
            &layer_name("attn_k.weight"),
            &[key_width, hidden],
        )?;
        require_shape(
            directory,
            &layer_name("attn_v.weight"),
            &[value_width, hidden],
        )?;
        require_shape(
            directory,
            &layer_name("attn_output.weight"),
            &[hidden, output_width],
        )?;
        require_shape(
            directory,
            &layer_name("attn_q_clip.weight"),
            &[u64::from(query_heads), u64::from(key_length)],
        )?;
        require_shape(
            directory,
            &layer_name("attn_rel.weight"),
            &[
                product(metadata.relative_length, query_heads, "relative query")?,
                hidden,
            ],
        )?;
        require_shape(
            directory,
            &layer_name("attn_rel_bias.weight"),
            &[
                u64::from(if is_local {
                    metadata.sliding_window
                } else {
                    metadata.relative_extent
                }),
                u64::from(metadata.relative_length),
            ],
        )?;
        require_shape(
            directory,
            &layer_name("attn_rel_clip.weight"),
            &[u64::from(query_heads), u64::from(metadata.relative_length)],
        )?;
        require_shape(
            directory,
            &layer_name("attn_sink_proj.weight"),
            &[u64::from(query_heads), hidden],
        )?;
        require_shape(
            directory,
            &layer_name("attn_k_conv.weight"),
            &[key_width, u64::from(metadata.convolution_cache_length)],
        )?;
        require_shape(
            directory,
            &layer_name("attn_v_conv.weight"),
            &[value_width, u64::from(metadata.convolution_cache_length)],
        )?;
        require_shape(
            directory,
            &layer_name("attn_post_conv.weight"),
            &[hidden, u64::from(metadata.convolution_cache_length)],
        )?;
        if is_local {
            require_shape(directory, &layer_name("attn_norm.weight"), &[hidden])?;
        } else {
            require_shape(directory, &layer_name("attn_q_inp_norm.weight"), &[hidden])?;
            require_shape(directory, &layer_name("attn_k_inp_norm.weight"), &[hidden])?;
            require_shape(directory, &layer_name("attn_v_inp_norm.weight"), &[hidden])?;
            let dynamic_taps = product(
                metadata.convolution_cache_length,
                kv_heads,
                "global convolution taps",
            )?;
            require_shape(
                directory,
                &layer_name("attn_k_conv_proj.weight"),
                &[dynamic_taps, hidden],
            )?;
            require_shape(
                directory,
                &layer_name("attn_v_conv_proj.weight"),
                &[dynamic_taps, hidden],
            )?;
        }
        if layer > 0 {
            if is_local {
                require_shape(directory, &layer_name("attn_res_query.weight"), &[hidden])?;
            } else {
                require_shape(
                    directory,
                    &layer_name("attn_res_query.weight"),
                    &[3, hidden],
                )?;
            }
        }
        require_shape(directory, &layer_name("ffn_norm.weight"), &[hidden])?;
        require_shape(directory, &layer_name("ffn_res_query.weight"), &[hidden])?;
        require_shape(
            directory,
            &layer_name("ffn_post_conv.weight"),
            &[hidden, u64::from(metadata.convolution_cache_length)],
        )?;
        require_shape(
            directory,
            &layer_name("ffn_gate_inp.weight"),
            &[u64::from(metadata.experts), hidden],
        )?;
        require_shape(
            directory,
            &layer_name("ffn_gate_up_exps.weight"),
            &[
                u64::from(metadata.experts),
                product(metadata.expert_feed_forward_size, 2, "fused gate/up")?,
                hidden,
            ],
        )?;
        require_shape(
            directory,
            &layer_name("ffn_down_exps.weight"),
            &[
                u64::from(metadata.experts),
                hidden,
                u64::from(metadata.expert_feed_forward_size),
            ],
        )?;
        require_shape(
            directory,
            &layer_name("exp_probs_b.bias"),
            &[u64::from(metadata.experts)],
        )?;

        let in_loop = (metadata.loop_start..metadata.loop_end).contains(&layer);
        if !in_loop {
            if loop_tensors_per_layer[layer as usize] != 0 {
                return Err(invalid(format!(
                    "Inkling physical layer {layer} stores loop-controller tensors outside the repeated core"
                )));
            }
            continue;
        }

        require_shape(
            directory,
            &layer_name("loop_msg_query.weight"),
            &[4, hidden],
        )?;
        for controller in ["q", "k", "v", "ffn"] {
            require_shape(
                directory,
                &layer_name(&format!("loop_{controller}_ctrl_down.weight")),
                &[
                    u64::from(metadata.loop_gate_rank),
                    product(metadata.hidden_size, 2, "loop input")?,
                ],
            )?;
            for projection in ["forget", "gain"] {
                require_shape(
                    directory,
                    &layer_name(&format!("loop_{controller}_ctrl_{projection}.weight")),
                    &[hidden, u64::from(metadata.loop_gate_rank)],
                )?;
            }
        }
        if loop_tensors_per_layer[layer as usize] != 13 {
            return Err(invalid(format!(
                "Inkling physical layer {layer} must contain 13 loop-controller tensors, found {}",
                loop_tensors_per_layer[layer as usize]
            )));
        }
    }

    let expected_names = expected_tensor_names(metadata);
    let actual_names = directory
        .tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<BTreeSet<_>>();
    if actual_names.len() != directory.tensors.len() {
        return Err(invalid("Inkling tensor directory has duplicate names"));
    }
    if let Some(name) = actual_names.difference(&expected_names).next() {
        return Err(invalid(format!(
            "Inkling tensor {name:?} is not part of the supported layout"
        )));
    }
    if let Some(name) = expected_names.difference(&actual_names).next() {
        return Err(invalid(format!(
            "Inkling tensor directory is missing required tensor {name:?}"
        )));
    }

    let total_tensors = u32::try_from(directory.tensors.len())
        .map_err(|_| invalid("Inkling tensor directory exceeds u32"))?;
    Ok(TensorInventory {
        total_tensors,
        tensors_per_physical_layer,
        encodings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ARCHITECTURE, inspect_metadata};
    use magnitude_artifacts::gguf::{
        Array, ByteOrder, Encoding, Metadata as Entry, Scalar, Strings, TensorDescriptor, Value,
    };

    fn tensor(name: String, shape: Vec<u64>) -> TensorDescriptor {
        TensorDescriptor {
            name,
            shape,
            encoding: Encoding::F32,
            offset: 0,
            nbytes: 0,
        }
    }

    fn directory() -> Directory {
        let mut metadata = vec![
            Entry {
                name: "general.architecture".into(),
                value: Value::Scalar(Scalar::String(ARCHITECTURE.into())),
            },
            Entry {
                name: "tokenizer.ggml.tokens".into(),
                value: Value::Array(Array::from(
                    (0..32)
                        .map(|index| format!("token-{index}"))
                        .collect::<Strings>(),
                )),
            },
        ];
        macro_rules! unsigned {
            ($key:literal, $value:expr) => {
                metadata.push(Entry {
                    name: format!("{ARCHITECTURE}.{}", $key),
                    value: Value::Scalar(Scalar::Unsigned($value)),
                })
            };
        }
        unsigned!("block_count", 24);
        unsigned!("context_length", 1024);
        unsigned!("embedding_length", 2304);
        unsigned!("feed_forward_length", 1152);
        unsigned!("expert_count", 128);
        unsigned!("expert_used_count", 8);
        unsigned!("attention.key_length", 128);
        unsigned!("attention.value_length", 128);
        unsigned!("loop.count", 3);
        unsigned!("loop.start", 6);
        unsigned!("loop.end", 18);
        unsigned!("loop.gate_rank", 64);
        unsigned!("vocab_size", 32);
        unsigned!("vocab_size_unpadded", 32);
        unsigned!("expert_feed_forward_length", 1152);
        unsigned!("attention.key_length_swa", 128);
        unsigned!("attention.value_length_swa", 128);
        unsigned!("rope.dimension_count", 0);
        unsigned!("attention.sliding_window", 1024);
        unsigned!("shortconv.l_cache", 4);
        unsigned!("attn_res.block_size", 6);
        unsigned!("attention.relative_length", 16);
        unsigned!("attention.relative_extent", 1024);
        unsigned!("attention.log_scale_floor", 8192);
        for (key, value) in [
            ("attention.head_count", 24_u64),
            ("attention.head_count_kv", 8_u64),
        ] {
            metadata.push(Entry {
                name: format!("{ARCHITECTURE}.{key}"),
                value: Value::Array(Array::from(vec![value; 24])),
            });
        }
        let mut local = vec![true; 24];
        for layer in [5, 11, 17, 23] {
            local[layer] = false;
        }
        metadata.push(Entry {
            name: format!("{ARCHITECTURE}.attention.sliding_window_pattern"),
            value: Value::Array(Array::from(local)),
        });
        metadata.push(Entry {
            name: format!("{ARCHITECTURE}.loop.varying"),
            value: Value::Scalar(Scalar::Bool(false)),
        });
        metadata.extend(
            [
                ("attention.layer_norm_rms_epsilon", 1e-6),
                ("expert_weights_scale", 8.0),
                ("activation.situ_beta", 4.0),
                ("activation.situ_linear_beta", 25.0),
                ("final_logit_softcapping", 30.0),
                ("attention.qk_clip", 22.627416610717773),
                ("attention.relative_clip", 8.0),
                ("attention.log_scale_alpha", 0.1),
            ]
            .map(|(key, value)| Entry {
                name: format!("{ARCHITECTURE}.{key}"),
                value: Value::Scalar(Scalar::Float(value)),
            }),
        );

        let hidden = 2304_u64;
        let mut tensors = vec![
            tensor("token_embd.weight".into(), vec![32, hidden]),
            tensor("token_embd_norm.weight".into(), vec![hidden]),
            tensor("output.weight".into(), vec![32, hidden]),
            tensor("output_norm.weight".into(), vec![hidden]),
            tensor("output_res_query.weight".into(), vec![hidden]),
        ];
        for layer in 0..24 {
            let p = format!("blk.{layer}.");
            tensors.extend([
                tensor(format!("{p}attn_q.weight"), vec![3072, hidden]),
                tensor(format!("{p}attn_gate.weight"), vec![3072, hidden]),
                tensor(format!("{p}attn_k.weight"), vec![1024, hidden]),
                tensor(format!("{p}attn_v.weight"), vec![1024, hidden]),
                tensor(format!("{p}attn_output.weight"), vec![hidden, 3072]),
                tensor(format!("{p}attn_q_clip.weight"), vec![24, 128]),
                tensor(format!("{p}attn_rel.weight"), vec![384, hidden]),
                tensor(format!("{p}attn_rel_bias.weight"), vec![1024, 16]),
                tensor(format!("{p}attn_rel_clip.weight"), vec![24, 16]),
                tensor(format!("{p}attn_sink_proj.weight"), vec![24, hidden]),
                tensor(format!("{p}attn_k_conv.weight"), vec![1024, 4]),
                tensor(format!("{p}attn_v_conv.weight"), vec![1024, 4]),
                tensor(format!("{p}attn_post_conv.weight"), vec![hidden, 4]),
                tensor(format!("{p}ffn_norm.weight"), vec![hidden]),
                tensor(format!("{p}ffn_post_conv.weight"), vec![hidden, 4]),
                tensor(format!("{p}ffn_res_query.weight"), vec![hidden]),
                tensor(format!("{p}ffn_gate_inp.weight"), vec![128, hidden]),
                tensor(
                    format!("{p}ffn_gate_up_exps.weight"),
                    vec![128, 2304, hidden],
                ),
                tensor(format!("{p}ffn_down_exps.weight"), vec![128, hidden, 1152]),
                tensor(format!("{p}exp_probs_b.bias"), vec![128]),
            ]);
            if layer > 0 {
                tensors.push(tensor(
                    format!("{p}attn_res_query.weight"),
                    if [5, 11, 17, 23].contains(&layer) {
                        vec![3, hidden]
                    } else {
                        vec![hidden]
                    },
                ));
            }
            if [5, 11, 17, 23].contains(&layer) {
                tensors.extend([
                    tensor(format!("{p}attn_q_inp_norm.weight"), vec![hidden]),
                    tensor(format!("{p}attn_k_inp_norm.weight"), vec![hidden]),
                    tensor(format!("{p}attn_v_inp_norm.weight"), vec![hidden]),
                    tensor(format!("{p}attn_k_conv_proj.weight"), vec![32, hidden]),
                    tensor(format!("{p}attn_v_conv_proj.weight"), vec![32, hidden]),
                ]);
            } else {
                tensors.push(tensor(format!("{p}attn_norm.weight"), vec![hidden]));
            }
            if (6..18).contains(&layer) {
                tensors.push(tensor(format!("{p}loop_msg_query.weight"), vec![4, hidden]));
                for controller in ["q", "k", "v", "ffn"] {
                    tensors.push(tensor(
                        format!("{p}loop_{controller}_ctrl_down.weight"),
                        vec![64, 4608],
                    ));
                    for projection in ["forget", "gain"] {
                        tensors.push(tensor(
                            format!("{p}loop_{controller}_ctrl_{projection}.weight"),
                            vec![hidden, 64],
                        ));
                    }
                }
            }
        }
        Directory {
            version: 3,
            byte_order: ByteOrder::Little,
            alignment: 32,
            data_offset: 0,
            metadata,
            tensors,
        }
    }

    #[test]
    fn audits_physical_weight_layout_without_expanding_loop_weights() {
        let directory = directory();
        let metadata = inspect_metadata(&directory).unwrap();
        let inventory = inspect_tensors(&directory, &metadata).unwrap();
        assert_eq!(inventory.tensors_per_physical_layer.len(), 24);
        for (layer, &count) in inventory.tensors_per_physical_layer.iter().enumerate() {
            let local = ![5, 11, 17, 23].contains(&layer);
            let expected = if (6..18).contains(&layer) {
                if local { 35 } else { 39 }
            } else if layer == 0 {
                21
            } else if local {
                22
            } else {
                26
            };
            assert_eq!(count, expected, "physical layer {layer}");
        }
        assert_eq!(inventory.total_tensors, 704);
        assert_eq!(inventory.encodings.len(), 1);
        assert_eq!(
            inventory.encodings.get("F32"),
            Some(&inventory.total_tensors)
        );
    }

    #[test]
    fn rejects_a_malformed_routed_expert_shape() {
        let mut directory = directory();
        directory
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == "blk.6.ffn_down_exps.weight")
            .unwrap()
            .shape[2] = 1024;
        let metadata = inspect_metadata(&directory).unwrap();
        assert!(matches!(
            inspect_tensors(&directory, &metadata),
            Err(InspectionError::Invalid(detail)) if detail.contains("ffn_down_exps.weight")
        ));
    }
}
