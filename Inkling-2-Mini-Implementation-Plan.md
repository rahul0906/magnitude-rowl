# Inkling 2 Mini support in Magnitude

Engineering design and implementation plan  |  26 November 2026

Purpose: define the changes required to run Thinking Machines Lab’s Inkling 2 Mini in Magnitude, beginning with correct text inference and extending to SSD expert streaming and image input. This is an implementation plan, not a claim that the model currently runs in Magnitude.

Recommendation: add a dedicated model family and first prove a single GGUF checkpoint against a pinned reference. Treat the 24 physical weight layers and 48 logical layer visits as separate identities. Reuse physical weights and streaming records, but allocate independent attention and convolution state for each logical visit. Add streaming and vision only after the corresponding resident text path is correct.

## Scope and desired outcome

The first milestone serves text through the existing Magnitude interfaces using one supported GGUF layout, one supported quantization and one representative backend. It includes tokenization and chat formatting, prompt prefill, autoregressive decoding, and correct per-session state. Follow-on milestones add supported hardware backends, disk-backed experts, image input, and optional speculative decoding.

A successful release must load an explicitly supported checkpoint, produce reference-consistent logits and deterministic token sequences, preserve correctness across prefill and decoding, and report unsupported packages accurately. Performance goals must be established from measurements after correctness; no speed or laptop compatibility guarantee is assumed here.

## Model and checkpoint facts

- Inkling 2 Mini has 26.4 billion parameters and 3.4 billion unique active parameters. Its two additional loop passes increase effective activations to approximately 5.4 billion parameters per token.

- The model has 24 physical layers. Layers 0–5 run once, layers 6–17 run three times, and layers 18–23 run once: 48 logical layer visits in total. Its middle passes use learned loop-controller operations and each pass has its own KV and convolution state.

- Each layer routes each token to 8 of 128 experts. Attention mixes local and global layers. The model supports text and image input; a separate vision encoder and projector is available in published GGUF distributions.

- The ggml-org GGUF distribution documents BF16 at 52.2 GB, Q8_0 at 27.8 GB and IQ4_NL at 17.6 GB, plus a separate image projector. The original routed expert tensors use NVFP4 in layers 0–19 and MXFP8 in layers 20–23. These file sizes exclude runtime buffers, state and operating-system overhead.

The selected GGUF architecture is `inkling2_loop`. Other converters may use different layouts or tensor encodings. The official model is distributed in formats that are not automatically equivalent to Magnitude’s GGUF loading path.

## Reference audit and first artifact

Audit date: 26 November 2026. **Decision: use the locally available AtomicChat AD-IQ1_S file for the first resident text implementation and same-file regression tests.** Its small footprint enables work on the available machine and its header has already been inventoried in this checkout. This is a mixed, aggressively quantized community conversion, not the original vendor checkpoint or a high-fidelity model-quality baseline. It is a provisional implementation target, not an assertion that its complete encoding mix or numerical execution is supported. If a required encoding proves unavailable on the chosen backend, resolve that explicitly before claiming the milestone.

| Identity | Evidence |
| --- | --- |
| Repository and revision | [`AtomicChat/Inkling-2-Mini-GGUF@309509848b3151419aa26d069eb03db57fd29a00`](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF/tree/309509848b3151419aa26d069eb03db57fd29a00) |
| File | `Inkling-2-Mini-AD-IQ1_S.gguf`, 6,881,463,904 bytes |
| SHA-256 | `4d3cab7e88016548136e13f7ec465db7fc49ab8cf9559a60f3daad1f3f15c9d5` — previously calculated in the local audit; the cache blob has the same name. Not independently checked against a Hub checksum during this audit. |
| Local artifact | `/Users/rahul/.cache/huggingface/hub/models--AtomicChat--Inkling-2-Mini-GGUF/snapshots/309509848b3151419aa26d069eb03db57fd29a00/Inkling-2-Mini-AD-IQ1_S.gguf` |
| Header and tensor directory | Local prior header audit: GGUF v3, 62 metadata entries, 704 tensors, `general.architecture=inkling2_loop`; 24 physical layers, fixed loop `[6,18)`, three passes and 48 visits; 128 experts, 8 selected. Tensor encodings: F32 334, Q8_0 320, IQ1_S 24, Q2_0 24, Q4_K 1, Q6_K 1. Counts are tensor counts, not proportions of bytes. See `Inkling-2-Mini-Progress.md` and `inference/engine/families/inkling2/tests/artifact.rs`. The full header, tokenizer, per-layer shapes and controller inventory still need review against the pinned reference. |

**Conversion provenance.** The [AtomicChat model card](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF) says its BF16 is dequantized from the vendor’s NVFP4/MXFP8 checkpoint, and its AD files assign types per tensor, keeping attention and loop controllers at Q8_0. It reports an importance matrix made from 4.76 million chat-formatted tokens, two shards using BF16 and four Q8_0, and a patched importance-aware Q2_0 quantizer for the low-bit expert down projections. These are publisher claims from the current card; the exact source checkpoint revision, converter and quantizer commits, commands, calibration inputs and their hashes for this particular pinned file have **not** been independently verified. Do not describe the conversion as reproducible until those are recovered. The card reports 79.5% top-1 agreement and 0.278 mean KLD against its BF16 on its own corpus; these are publisher measurements, not Magnitude validation.

**Alternatives considered.** [ggml-org’s distribution](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/tree/main) lists BF16 (52.2 GB), Q8_0 (27.8 GB), IQ4_NL and NVFP4 (each about 17.6 GB), plus separate vision projectors. Q8_0 is a useful higher-precision follow-up: its [published pointer](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/blob/main/Inkling-2-Mini-Q8_0.gguf) gives 27,820,099,264 bytes and SHA-256 `d57542f8aea5d7639beba5fc4ae7f584754916f63ac5d5a69328c27f652f45da`; its full repository revision, complete header, exact conversion recipe and local content hash have not been established here. Do not substitute it into same-file parity tests without pinning and inspecting it. BF16 is larger still and represents dequantized original weights, not newly recovered precision. [Unsloth’s mainline layout notes](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/Mainline_Support/README.md) distinguish mainline files from its root layout; do not assume GGUFs from different publishers are interchangeable.

**Reference implementation.** Use ggml-org llama.cpp’s `inkling2_loop` as the GGUF execution reference; [PR #32694](https://github.com/ggml-org/llama.cpp/pull/32694) merged the architecture. AtomicChat documents llama.cpp commit [`e4b1cb5318f1ac55d148be228c6347c6a94cc4c6`](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6) for its AD-Q4_K plus optional drafter, so it is a concrete **candidate** reference build, not evidence that this IQ1_S file has been validated on it. Pin that commit for the first attempted same-file reference run and record build flags and any required fixes; replace the pin only with documented evidence. The accessible current [loop graph](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/inkling2-loop.cpp) shares physical weights, gives each cache slot separate KV and convolution state, restarts each loop pass from the prelude, and feeds messages from the previous pass to later passes. The current [base graph](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/inkling2.cpp) specifies relative-position bias instead of RoPE and convolution state holding K, V, attention and FFN outputs. These moving-source observations are leads, not a verified audit of the candidate commit. Exact controller equations, tensor mapping, rounding, token IDs, chat template and stopping behavior must be checked against that commit and the pinned file before the reference gate is closed.

**Gate status: open.** The selection, identity and partial artifact inventory are recorded. A reviewed operator inventory and exact pinned-reference comparison remain outstanding. The locally installed llama.cpp versions previously rejected `inkling2_loop`; no reference logits or generation have been established (see `Inkling-2-Mini-Progress.md`). Same-file parity establishes implementation agreement, not fidelity to the original checkpoint; compare with a pinned higher-precision artifact separately when feasible.

## Current Magnitude integration

The engine recognizes a GGUF architecture through the registered ModelFamily implementations in inference/engine/families/registry/src/lib.rs. A family inspects a package and produces a family-neutral ModelDefinition and input adapter; contracts live in inference/engine/families/contracts. An unregistered Inkling family crate now contains schedule and header inspection work; it does not yet provide a serving implementation. See `Inkling-2-Mini-Progress.md` for its current status and validation limits.

Existing families show reusable concepts such as routed experts, short convolutions and sliding-window attention. Their presence does not prove numerical compatibility with Inkling. Audit the actual contracts, executor, kernels, quantization support and streaming planner before deciding which parts can be reused. In particular, do not assume that the existing short convolution or attention operators implement Inkling’s precise equations or relative-position bias.

## Identity and execution design

### Physical weights and logical visits

Represent physical layer identity independently from logical execution position. The execution schedule visits physical layers in the sequence 0–5, then 6–17 three times, then 18–23. A visit must identify at least its physical layer, loop pass and logical state slot. Keep a single owner for each imported physical tensor and have visits refer to it. Validate that tied descriptors do not cause the importer, placement planner or device backend to duplicate the weights.

The loop controller requires saved activations and learned transformations across passes. Derive the exact equations, operation order, intermediate dtypes, normalization, bias, and first-pass behavior from the pinned model reference and GGUF conversion. Express those operations through general contracts only where their semantics are exact. If a contract cannot represent the computation, extend the shared contract and executor at the owning abstraction rather than hiding the loop in a family-specific serving path.

### State ownership and lifecycle

KV cache and short-convolution history belong to a logical visit and session, not just to a physical layer. The three visits to a looped layer must never alias state. Verify state allocation, indexing, batching, prefix reuse, reset, cancellation, rollback and eviction. Position and causal visibility must match the reference for both a multi-token prefill and incremental decoding. Attention windows must retain exactly the history required by that visit.

The runtime must account for logical visits when estimating cache memory and for unique physical tensors when estimating model weight memory. Confirm this distinction throughout capacity planning, admission, metrics and the UI. Long context is a supported model property, not evidence that a million-token cache fits a particular laptop.

## Model family and artifact handling

- Create an inkling2 family crate following the existing family contract. Register it once in the central registry and include the workspace dependencies required by the build.

- Inspect architecture metadata, layer count, loop boundaries and pass count, attention configuration, expert count and selection count, tokenizer, and required tensors. Validate dimensions and consistency; reject missing or unsupported features with actionable reasons.

- Map GGUF tensor names and layouts to weight descriptors, including loop-controller weights, attention projections, routing, experts, embedding and output head. Use exact import transformations only when the reference layout requires them.

- Audit loader and backend kernel support for every encoding in the selected mixed AD-IQ1_S artifact. Reject unsupported formats rather than silently interpreting them; use a separately pinned Q8_0 or BF16 artifact for higher-precision comparisons.

- Configure the input adapter, tokenizer markers, chat template, stop tokens, generation settings, and tool-call parsing as applicable. Verify prompt token IDs against the pinned reference. Do not equate successful loading with correct conversational behavior.

## Numerical operators and backend work

Build an operator inventory from the pinned reference and compare it with the family contracts, execution lowering, and each desired backend. Check gated grouped-query attention, local and global masks, learned relative-position bias, short convolution, router scores and top-k normalization, expert activation and scaling, residual paths, and loop-controller math. Record exact equations and tensor shapes before deciding whether an existing operator is reusable.

First implement a clear correct path using existing primitive operations where possible. Add shared operators or kernels for missing semantics and cover them with focused numerical tests. The attention implementation must apply relative bias at the correct point and preserve causal and local-window masking. Confirm whether the reference uses additional numerical details such as clipping or special normalization before declaring parity.

Backend coverage must follow implemented capabilities rather than assuming every kernel exists everywhere. Make unsupported backend and quantization combinations fail during assessment with a precise reason. Add optimized fused paths only after the unfused path and end-to-end reference comparisons pass.

## SSD and memory tier streaming

After resident inference is correct, connect Inkling’s routed expert tensors to Magnitude’s existing expert directory, placement and streaming machinery. A physical expert must have one stable identity across loop visits; cache keys, records and residency accounting should reference physical weights, while routing and completion state remain associated with the current logical visit and request. Verify that a later visit cannot consume stale routing results from an earlier one.

Audit packed expert layouts and mixed per-layer encodings against the streaming importer and kernels. Ensure nonexpert weights, controller weights, caches and active scratch memory fit the relevant tiers. If prefetch or prediction assumes a strictly linear sequence of unique layers, adapt it to the loop schedule. Correctness must not depend on a prediction being right: missing experts must be fetched or the request must fail explicitly.

Test resident, RAM-tier and SSD-tier execution against the same deterministic reference, including repeated visits to the same expert, changing routes, simultaneous sessions, partial cache residency, cancellation and read errors. Measure actual disk reads and resident bytes to detect unintended weight triplication or needless repeated loading. Treat throughput and responsiveness as measured properties, not guarantees.

## Vision and optional acceleration

For image support, inspect the specific published projector package and the reference preprocessing path. Implement the vision encoder and projection using shared vision contracts where semantically adequate, plus the family’s image markers, patch preparation, positioning and embedding insertion. Validate preprocessing and projected image embeddings before end-to-end image prompts. Text support should explicitly reject image requests until this path is implemented.

A separate draft model or multi-token prediction path is an optimization, not a prerequisite for correct target inference. Add it only after target parity, then validate acceptance, rollback and cache state for looped visits. Do not describe speculative decoding as supported merely because a community draft artifact exists.

## Validation and acceptance

### Reference and reproducibility

Pin a checkpoint revision, GGUF file hash, conversion provenance, tokenizer/template and reference engine commit. Record the exact prompt tokens, sampling parameters, context length, backend, dtype and comparison tolerance. Use reference logits or intermediate tensors where available; generated prose alone can conceal numerical errors.

### Correctness matrix

- Artifact inspection: valid pinned headers load; wrong architecture, missing tensors, malformed shapes, incompatible loop metadata and unsupported encodings fail clearly.

- Operator parity: compare each new operation, including attention bias and masking, router outputs, convolution state and loop-controller outputs, against a small reference fixture. Include boundary positions and representative quantized inputs.

- Loop and cache: verify all 48 logical visits, three distinct states for each shared middle layer, correct weight aliasing, and reset/isolation across requests. Compare single-token and multi-token prefill followed by several decode steps.

- End to end: compare reference logits at selected prompt and decode positions with tolerances justified for the selected precision; compare greedy token sequences on fixed prompts. Include sliding-window boundary, multiple sessions and a context length that exercises both local and global attention.

- Streaming: compare output against the same resident checkpoint with expert tiers forced separately; exercise cache hits, misses, repeated visits and errors. Verify recorded memory allocation and disk reads reflect physical weight sharing.

- Vision, when implemented: compare preprocessed patches, projector output and text-plus-image logits against the reference. Verify text-only behavior remains unchanged.

Follow the repository’s check policy: inspect existing mrig runs first; run only checks that resolve a decision or verify a change; use the cheapest representative check; run hardware work in the background with purpose and estimate; establish correctness before timing. Run the required mrig check for the completed change and the Windows cross-compile when OS-facing code changes. Review failures and overruns before repeating a run.

## Delivery sequence and decision gates

1. Reference audit: pin one GGUF and reference implementation; document exact metadata, tensor mapping, loop equations, state layout and unsupported encodings. Gate: a reviewed operator and artifact inventory.

2. Family and resident text: implement recognition, package inspection, execution schedule and correct operators on one backend. Gate: deterministic prefill and decode match the reference and state is isolated per logical visit.

3. Product integration: complete tokenizer, template, stopping, session lifecycle and capacity accounting. Gate: fixed chat and concurrent-session cases pass and unsupported inputs fail clearly.

4. Expert streaming: enable supported RAM and SSD tiers and verify shared physical records. Gate: streamed and resident outputs agree and memory accounting shows no accidental triplication.

5. Vision and other backends: implement each supported path with its own parity checks. Gate: representative end-to-end checks pass per path and backend.

6. Performance: only after correctness, use scoped mrig measurements to identify bottlenecks and tune. Gate: report hardware, checkpoint, context, source revision and measured variability; do not promise a speedup without evidence.

## Open questions and risks

- Do all tensor encodings in the selected AD-IQ1_S artifact work in the existing import and routed kernels on the first backend? What pinned higher-precision artifact and hardware will be used to assess quantization effects?

- Can current decoder contracts and execution planning represent the loop controller, shared physical weights and per-visit cache identity without extending common abstractions?

- Does each intended backend support the needed attention bias, convolution, expert layouts and mixed quantization? Which combinations should initially be withheld?

- How do prefix caching, batching, cancellation, memory admission and streaming prediction treat repeated physical layers? These require code-level audit and targeted tests.

- What memory and throughput are achievable on a specific laptop at realistic context lengths? The advertised context limit and a model file size do not answer this.

## References

Thinking Machines Lab, Inkling 2 Mini model card: https://thinkingmachines.ai/model-card/inkling-2-mini/

Thinking Machines Lab, original model repository: https://huggingface.co/thinkingmachines/Inkling-2-Mini

ggml-org, GGUF model card and checkpoint notes: https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/blob/main/README.md

Unsloth, mainline GGUF layout notes: https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/Mainline_Support/README.md

Magnitude source: inference/engine/families/registry/src/lib.rs and inference/engine/families/contracts/src/family.rs; root AGENTS.md and inference/AGENTS.md for check policy.
