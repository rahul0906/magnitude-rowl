# Large MoE inference on an 18 GB M3 Pro

Prepared 26 November 2026 for Rahul. Primary workload: long documents and coding agents at 32K or more context, one active generation at a time. Preserve model quality and optimize execution before considering more aggressive quantization.

The recommended first path is **Magnitude for the existing Gemma 4 checkpoint, and MoeShift for a quality-preserving Inkling checkpoint**. Establish those baselines before changing either engine. Atomic Engine is an especially relevant development candidate because it streams from the original GGUF and understands Inkling's repeated layers, but its inspected memory-budget path currently rejects Inkling contexts above 4,096 tokens. Completing Inkling inside Magnitude is a larger project and should follow evidence that the existing alternatives cannot meet the workload.

No inference throughput was measured during this planning pass. The inventory below comes from local files, hardware information, source inspection, and existing run records. Future settings and performance targets are proposals, with explicit qualification gates.

## 1 The actual starting point

| Item | Observed state | Consequence |
|---|---|---|
| Laptop | MacBook Pro, Apple M3 Pro, 12 CPU cores comprising 6 performance and 6 efficiency cores, 18 GPU cores | Use native arm64 and Metal. Test CPU worker counts around the performance-core count only when CPU work is limiting. |
| Memory | 18 GB unified memory | CPU and GPU allocations consume one physical pool. There is no additional independent GPU memory allowance. |
| Storage | Internal Apple AP0512Z SSD; about 29 GiB free at inspection | Extracting another complete expert store can exhaust practical headroom. Resolve storage before new quality-model downloads. |
| Magnitude | Installed CLI/app 0.3.3; this checkout starts at `e8d793836` and has uncommitted Inkling work | Installed binaries and this source tree are different experimental subjects. Record exactly which is measured. |
| MoeShift | Homebrew `0.1.0-alpha.6` | An Inkling preset is installed. The active user configuration still has placeholder paths for two Qwen profiles. |
| Atomic Chat | App 2.4.7; bundled upstream backend version file says `b12532` | An installed app does not establish which newer Atomic Engine features it exposes. Qualify the actual selected backend. |
| Existing Gemma | `HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP`, snapshot `f9093662a2e7ae0503f637088bc96f77a1a70c83`, Q4_K_M | This is a particular fine-tune and mixed quantization, not an interchangeable stock Google checkpoint. Keep it fixed for engine comparisons. |
| Gemma files | Main GGUF 16,796,015,520 bytes, approximately 15.64 GiB; MTP companion 251,937,728 bytes; vision projector 1,194,827,776 bytes | The main file alone leaves little space for macOS, editor, KV state, and scratch if fully resident. Text tests should not load the vision projector. |
| Existing Inkling | `AtomicChat/Inkling-2-Mini-GGUF@309509848b3151419aa26d069eb03db57fd29a00`, `Inkling-2-Mini-AD-IQ1_S.gguf` | Useful for small, inexpensive compatibility and resident-versus-streamed correctness checks. |
| Inkling file | 6,881,463,904 bytes, approximately 6.41 GiB; architecture `inkling2_loop` | It can plausibly fit at useful contexts, but its aggressive weight quantization conflicts with the preferred quality objective. |

The Gemma file is under:

```text
/Users/rahul/.magnitude/models/hub/models--HauhauCS--Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP/snapshots/f9093662a2e7ae0503f637088bc96f77a1a70c83/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf
```

The existing Inkling file is:

```text
/Users/rahul/.cache/huggingface/hub/models--AtomicChat--Inkling-2-Mini-GGUF/snapshots/309509848b3151419aa26d069eb03db57fd29a00/Inkling-2-Mini-AD-IQ1_S.gguf
```

Header inspection confirms 30 Gemma blocks and 128 experts with 8 selected per routed block. Inkling stores 24 physical blocks, with blocks 6 through 17 visited three times: 48 logical visits in total, also selecting 8 of 128 experts. Physical weight identity and logical state identity must be treated separately.

### Existing uncommitted work

The new `inference/engine/families/inkling2/` crate inspects metadata and tensors and expands the loop schedule. The tokenizer change recognizes the Inkling profile. The executor test change adds coverage for mixed IQ1_S gate/up and Q2_0 down tensors. These are useful foundations, but the family registry does not register Inkling and the crate does not build an executable model definition.

The existing run at `inference/evaluation/results/20261126T074753Z-script-ff57df/` failed with `unknown model architecture: 'inkling2_loop'` in an older llama.cpp runtime. It produced no generation; it is not a failed speed measurement or evidence that SSD streaming itself is broken.

Preserve the dirty tree while making the baseline independent of it. Future implementation should use an isolated checkout or `mrig --build` selection. If cleanup is desired, archive both tracked diffs and the untracked family/docs before removing anything. A diff alone does not preserve untracked work. Keep only foundations that pass independent parity checks; do not assume they need to become the product implementation.

## 2 Runtime selection

| Runtime | Role in this plan | Qualification needed |
|---|---|---|
| Magnitude | Primary Gemma baseline and a mature local place to optimize streaming | Confirm the installed/source build, selected quant, context allocation, actual streaming, and MTP activation. |
| MoeShift alpha.6 | First practical Inkling SSD-streaming candidate for 32K and 64K | Its tested Inkling quant is Unsloth UD-Q4_K_XL. The existing AtomicChat IQ1_S and a proposed AtomicChat AD-Q4_K need separate mixed-format and template qualification. |
| Atomic Engine | Short-context Inkling comparison and a focused long-context development candidate | Lift and qualify its Inkling context restriction before treating it as a 32K solution. |
| Current compatible llama.cpp | Reference for GGUF loading, tokenizer, model math, and resident execution where memory permits | Pin a build supporting the exact merged loop layout and all tensor types. |
| Atomic Chat application | Possible daily interface after its backend is qualified | Verify backend identity, local-model import, context controls, and external-endpoint support if using it as a client. Do not assume these from installation alone. |

MoeShift explicitly optimizes SSD streaming for `qwen4exp`, `qwen35moe`, and `inkling2_loop`. Gemma is listed as ordinary loading, without trained-model qualification in that document. Routing Gemma through MoeShift does not automatically enable its expert-streaming path. Its server handles one inference request at a time. [MoeShift supported architectures](https://github.com/davide-troiani/moeshift/blob/v0.1.0-alpha.6/docs/supported-architectures.md)

Atomic Engine's `engine-development` commit `fb69fff02101b048a0a165a3b502712dd2ca978f` rejects budgeted contexts above 4,096 except a separately enabled extended Flash profile. Inkling's DSpark enablement does not grant that exception. Therefore a command combining Inkling, `--engine-budget-mib`, and `-c 32768` is currently invalid in that source. [Exact context admission code](https://github.com/AtomicBot-ai/atomic-engine/blob/fb69fff02101b048a0a165a3b502712dd2ca978f/src/llama-context.cpp#L181-L208)

The AtomicChat model card requires llama.cpp b12600 or newer for its merged-loop files and warns against mixing the older PR layout with the merged implementation. The installed Atomic Chat backend version file predates that threshold; establish compatibility rather than assuming the app contains the required new backend. [Model compatibility instructions](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF#run-it)

### What combining the applications means

Use one selected inference engine behind an OpenAI-compatible endpoint and connect the preferred coding/chat client to it. Reuse original GGUF files through paths wherever supported. Stop or unload competing model workers during measurements and memory-constrained use.

The applications do not share live expert caches or KV state simply because they use the same GGUF. Combining them inside one forward pass would require a new execution protocol and introduce transfers and synchronization. There is no evidence that this would help this laptop. Borrow implementation ideas, telemetry, and tests between repositories; retain one owner of model execution and memory accounting.

## 3 Quality and memory targets

### Weight precision

Retain the existing Gemma Q4_K_M as the first fixed-quality target. For Inkling, make a four-bit quality model the intended daily target: first evaluate AtomicChat AD-Q4_K compatibility, or use MoeShift's already-tested Unsloth UD-Q4_K_XL when minimizing integration risk. Different publishers' quants are different experimental subjects.

AtomicChat reports 79.5% next-token top-1 agreement and mean KLD 0.278 for the existing IQ1_S, versus 96.3% and 0.0080 for AD-Q4_K. Its AD-Q4_K is 16.53 decimal GB. These are publisher measurements against their BF16 reference, not task accuracy or measurements on this laptop. They are enough to reject IQ1_S as the default quality-preserving choice. A four-bit quant itself still needs a workload quality check. [AtomicChat quantization measurements](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF#files)

Keep these changes separate:

1. Exact expert caching, scheduling, and read overlap: preserve selected experts and weight bytes.
2. Weight quantization and KV quantization: alter numerical precision and require a quality decision.
3. Expert dropping, lower top-k, fewer loop passes, or approximate attention: alter model computation and are outside the primary plan.

Use FP16 KV as the initial quality reference. Q8 KV is a candidate after a paired long-context quality check. Do not inherit Q4 KV from a published speed command merely to reproduce a large tokens/s number. Preserve the model's actual sliding-window attention semantics and full-attention layers.

### Working memory budget

Start with an approximate 9–10 GiB engine allocation target, measured against whole-process footprint, and leave room for macOS and the actual editor/browser workload. Treat 11 GiB as a later candidate only when measurements show sufficient headroom. These are initial experimental budgets, not guaranteed safe limits.

Account for:

```text
physical memory = OS and other apps
                + shared model weights
                + resident expert cache
                + KV and recurrent state
                + graph scratch and conversion buffers
                + in-flight reads and speculative state
                + allocator/runtime overhead and safety margin
```

Count CPU/Metal aliases once when they share physical storage. Count separate repacks, scratch pools, and old-plus-new allocations during resizing separately. A configured engine budget may omit OS file cache and caller-owned buffers; RSS and Metal allocation totals cannot simply be added together.

Inkling has 8 full-attention logical visits and 40 sliding-window visits. Assuming a compact 1,024-token window, 8 KV heads, head width 128, separate K/V, and FP16 storage:

```text
KV bytes = (8 × context + 40 × 1024) × 8 × 128 × 2 × 2
```

That is about **1.16 GiB at 32,768 tokens** and **2.16 GiB at 65,536 tokens**, before alignment, checkpoints, convolution state, temporary copies, or draft state. This is an idealized capacity calculation, not the allocation of a particular runtime. A backend that allocates full context for every window layer can use much more. Require its allocation logs to explain the difference.

Measure memory during initial prefill, the prefill-to-decode transition, context growth, and a later agent follow-up. A low steady decode footprint can conceal a failing prefill peak.

### Storage budget

Magnitude can stream from model artifacts; MoeShift prepares a separate expert store. The installed Inkling setup documentation estimates roughly 34 decimal GB for its full download-and-preparation path. Current free space is about 31 decimal GB before build outputs and working headroom. [MoeShift installation requirements](https://github.com/davide-troiani/moeshift/tree/v0.1.0-alpha.6#requirements)

Before downloading a quality Inkling file, budget the original, extracted experts, temporary output, and a practical 15–20 GiB free-space reserve. Use exact sizes once the quant is selected. A reasonable preparation target is 50–60 GiB free, rather than filling the remaining disk. Preserve models and user data; review regenerable build output and stale artifacts for cleanup. Do not download several quants at once.

Atomic Engine's original-GGUF reading avoids an extracted weight copy and is attractive if its long-context path is completed. An external SSD is a capacity alternative only after measuring its actual read pattern through the rig; it is not assumed faster than the internal SSD.

## 4 What determines tokens per second

For batch-one decode, the useful cost model is:

```text
time per committed token ≈ serial routing/dispatch work
                        + critical path of GPU work and expert-read waits

SSD traffic per token ≈ sum of missed expert bytes across logical visits
                     + unused prefetch and read amplification

tokens/s ≤ effective SSD bytes/s ÷ physical SSD bytes per committed token
```

The last expression is an I/O ceiling, not a throughput prediction. Computation, attention, dequantization, copies, and dependencies lower the attainable rate. Some I/O can overlap computation; total I/O service time is not the same as exposed wait time.

Local tensor offsets give approximately 15.13 decimal GB of Gemma routed-expert storage, including negligible padding/scales. With 8 of 128 selected and no cache reuse, that is approximately **0.946 GB of routed weights per generated token**. At an illustrative effective SSD rate of 3 GB/s:

| Fraction of requested expert bytes missing | Approximate read bytes per token | I/O-only ceiling |
|---|---:|---:|
| 100% | 0.946 GB | 3.2 tokens/s |
| 50% | 0.473 GB | 6.3 tokens/s |
| 20% | 0.189 GB | 15.9 tokens/s |
| 10% | 0.095 GB | 31.7 tokens/s |

The 3 GB/s is an assumption for illustration, not a measured SSD result. Real routing, mixed expert sizes, alignment, read-ahead, and unused predictions change the traffic. Measure bytes and waits before deciding which optimization matters.

Inkling's existing IQ1_S has about 5.48 GB of expert storage. Counting all 48 logical visits with no reuse gives about 0.685 GB/token. A higher-quality quant increases the bytes. Its repeated core supplies a specific optimization opportunity: half the physical layers account for three quarters of logical visits. Cache physical expert weights across those visits, but keep per-visit KV and convolution state distinct.

For a 32K coding task, decode is only part of the experience. Track first-token latency, uncached prompt processing, follow-up prompt processing, time to a completed useful answer, and latency between streamed tokens. A configuration with faster decode and minutes of additional prefill is not automatically the better agent configuration.

## 5 Establish working baselines

### Stage A Prepare the experiments

1. Pin runtime commits/binary hashes, model revision, GGUF metadata, tensor encodings, tokenizer/template, and all flags. Record macOS version, power mode, connected power, background apps, and free storage.
2. Preserve the current working tree. Distinguish installed Magnitude, clean source Magnitude, and any new source candidate in results.
3. Restore the normal evaluation environment. `uv run --project inference/evaluation mrig runs` currently fails in this sandbox on UV cache access; a temporary cache with `--no-sync` then reports no `mrig` entry point. The standard evaluation environment needs setup before any new hardware result can be trusted.
4. Run `mrig runs` after setup and ensure the local device is available. The absent default run-state directory observed during planning is not proof that no external inference process is using the GPU.
5. Prepare one small compatibility prompt and one representative long code/document fixture. Do not start a broad benchmark suite.

**Exit:** each candidate has a reproducible identity and launch recipe, and the rig can record a local run.

### Stage B Qualify the existing files

For Gemma, run one bounded text-only request through Magnitude with the current checkpoint. Confirm the architecture, expert streaming, attention state, and whether the downloaded MTP companion is actually used. Having a companion file on disk does not prove speculation is active. Establish an explicit non-speculative control before measuring MTP.

For Inkling, first use the existing IQ1_S file for a small load/generate check in a compatible runtime. It is cheap enough to test resident execution and exact SSD streaming without downloading a new model. Check the unusual mixed tensor formats, the loop layout, and chat-template/stop behavior. Failure is a compatibility diagnosis; inspect it before another run.

MoeShift preparation commands are known from the installed tools. Execute extraction and verification inside a recorded rig script when this phase is implemented:

```sh
MOE_PREFIX="$(brew --prefix moeshift)"
INKLING_GGUF="/absolute/path/to/the/exact/Inkling-file.gguf"
INKLING_EXPERTS="/absolute/path/to/a/new/experts-directory"

"$MOE_PREFIX/libexec/bin/python" \
  "$MOE_PREFIX/share/moeshift/tools/moe/experts.py" \
  extract --model "$INKLING_GGUF" --out-dir "$INKLING_EXPERTS"

"$MOE_PREFIX/libexec/bin/python" \
  "$MOE_PREFIX/share/moeshift/tools/moe/experts.py" \
  verify --model "$INKLING_GGUF" --experts "$INKLING_EXPERTS"
```

Keep both original and prepared files. A manifest must match the exact GGUF; never reuse another quant's expert directory. These commands are preparation templates, not actions completed by this plan. [MoeShift manual configuration](https://github.com/davide-troiani/moeshift/blob/v0.1.0-alpha.6/configs/README.md)

**Exit:** plausible text is generated, format/template checks pass, and streaming changes do not alter the model's selected experts or weight bytes.

### Stage C Establish the quality Inkling path

After resolving disk capacity, obtain one pinned four-bit candidate. A possible AtomicChat download is:

```sh
hf download AtomicChat/Inkling-2-Mini-GGUF \
  Inkling-2-Mini-AD-Q4_K.gguf \
  --revision 309509848b3151419aa26d069eb03db57fd29a00
```

Confirm that revision's file metadata and compatibility before downloading. Use the returned cache path directly. If MoeShift rejects its mixed expert representation, first characterize the missing type/layout; compare the effort with using its tested Unsloth quant. Never substitute tensor types or rename an architecture to make loading succeed.

Start from the installed preset:

```text
/opt/homebrew/Cellar/moeshift/0.1.0-alpha.6/share/moeshift/inkling-2-mini.json
```

Create a separate experimental configuration with absolute `model` and `moe-experts` paths. Preserve unrelated user profiles. Initial proposed settings:

| Setting | First quality baseline | Reason |
|---|---|---|
| Context capacity | 32,768 total tokens | Establish the requested operating range before 64K. |
| Input/output | About 30,720 input tokens plus 512 generated tokens | Leave space for output and verify the post-template token count. |
| KV types | FP16 K and V | Numerical reference; Q8 is a later candidate. |
| GPU placement | Preset Metal placement, `n-gpu-layers: 99` | Expert compute should use the supported Metal path. |
| `prefill-max-tokens` | 1,024 initially | Conservative starting point below the shipped 3,072 on an 18 GB system. Reduce only if peak memory requires it. |
| `moe-prefill-bank` | 128 | Start with the model preset's shared-bank capacity. |
| `moe-decode-bank` | 0 | Preserve the intended separate decode-window design. |
| Decode window/protected slots | 12 and 6 respectively | Preserve the qualified pairing before tuning cache capacity. |
| Prediction depth/read count | 3 and 2 respectively | Do not independently change coupled policy settings. |
| CPU threads | 8 from the preset; later compare 6 if relevant | Thread count is not equal to available GPU compute. |
| Diagnostics | Enabled for attribution, disabled for timing | Avoid counting log/tracing overhead as inference time. |
| Requests in parallel | 1 | Matches the actual single-user objective and server execution model. |

First do a short request with these capacities; then one occupied 32K-context request. Merely starting with `ctx-size: 32768` does not exercise long-context behavior. For a full 32,768-token input, increase capacity enough for template and generated output, for example to 40,960 if supported, and record that distinct profile.

**Exit:** the quality checkpoint completes a real long request without OOM, persistent swap growth, hidden truncation, or changed loop count.

## 6 Optimize in the order the evidence supports

### First reclaim unnecessary memory

Keep one model active, disable unused vision components, avoid multiple server slots, and verify actual KV allocation. Preserve the full requested context. Reduce scratch or prefill microbatch peaks before reducing expert cache, because a smaller expert cache can increase every subsequent decode step's I/O.

For each engine, explain the allocation ledger at the beginning and end of prefill. Resolve unexpected double residency, full-context allocation for local attention, excessive checkpoint storage, and conversion copies before micro-optimizing kernels. Do not confuse reserved virtual addresses with resident physical memory.

### Then reduce expert misses

Capture one representative route trace containing token, phase, logical visit, physical tensor identity, expert IDs, stored/resident byte sizes, and reuse distance. Replay cache policies offline before running the model again. Replay the existing policy exactly first, including pinned and in-flight entries; a simulator that cannot reproduce the baseline cannot establish a better policy.

Compare only one promising candidate with the current policy. Useful questions are whether more capacity should go to repeated core layers, whether long-prefill traffic evicts the useful decode set, and whether a small cache increase meaningfully reduces bytes per token. Weight hit rates by bytes and exposed stall cost, not just expert counts.

MoeShift already combines repeated Inkling visits' allocations into shared physical-block windows and uses a decayed demand score with protected entries. Magnitude already has a reuse ranking, residency directory, asynchronous source reads, and dynamic bank sizing. These mechanisms should be measured and extended at their current ownership points, rather than reimplemented as another cache. [MoeShift cache and overlap design](https://github.com/davide-troiani/moeshift/blob/v0.1.0-alpha.6/docs/architecture.md#6-expert-caches-and-phase-transitions)

For an 18 GB machine, compare an initial approximately 10 GiB total engine budget with one nearby admissible budget, such as 9 or 11 GiB, selected by the trace's predicted marginal value. MoeShift slot counts are not a total-process cap: translate any extra slots into bytes and include transient allocations. Avoid a grid over every cache size.

### Overlap only the work that can hide a measured stall

If exposed expert-read waits dominate, run one controlled prefetch/overlap comparison. Distinguish required demand reads from predictions. Count useful predictions, unused bytes, late predictions, evictions caused by predictions, and GPU stalls saved.

Demand takes priority. Limit pending bytes so prediction cannot consume the last compute-ready slots. Reuse stable read workers and buffers. Coalesce adjacent reads only when the extra bytes cost less than the requests saved. On Apple Silicon, shared-memory reads and GPU execution also contend for memory bandwidth; overlap is not free.

Do not assume more read threads or deeper lookahead will improve performance. Increase one of them only if queue occupancy and read latency indicate unused storage parallelism. Require cancellation, partial-read failure, and GPU-reader lifetime tests for any new asynchronous path.

### Optimize long prefill independently

A 32K prompt reaches many more experts than one decode token. Its optimal bank, batch shape, and scheduling can differ substantially from decode. Group actual selected rows by expert and amortize reads within a bounded chunk. Retain exact attention dependencies and model math.

Start with the admitted microbatch, then test one larger value if scratch headroom and profiler data support it. Measure total prefill time, expert bytes read, conversion time, GPU submission count, and peak footprint together. Large batches that reduce SSD reads can still lose through graph memory or kernel inefficiency.

For coding agents, qualify prefix reuse separately: first a fresh long prompt, then the same prefix plus a short tool result, then a changed tool result. A reused prefix must include compatible model, tokenizer/template, precision, positions, loop state, and convolution state. Test cache invalidation after budget/context changes. Cached follow-up performance must not be presented as fresh 32K prefill speed.

### Introduce KV quantization only after the reference works

If FP16 KV materially limits useful expert capacity, compare Q8 KV against FP16 at the same context and same weight quantization. Include retrieval near the beginning, middle, and end, a code dependency task, and the relevant loop/window boundaries. Use the additional memory for expert cache only in a separately labeled end-to-end comparison.

An equal-expert-cache comparison isolates attention precision and kernel cost. An equal-total-memory comparison answers which configuration best serves this laptop. Both are useful, but they answer different questions. Defer Q4/TurboQuant KV unless Q8 cannot meet the objective and the quality trade-off is explicitly acceptable.

### Speculative decoding comes after the non-speculative baseline

Evaluate Gemma's downloaded assistant/MTP through a runtime that actually supports its pairing. For Inkling, Atomic Engine's Metal DSpark path is relevant after its long-context qualification; MoeShift's documented server rejects speculative-decoding options. [MoeShift configuration boundaries](https://github.com/davide-troiani/moeshift/blob/v0.1.0-alpha.6/configs/README.md#configuration-reference)

Keep target weights, sampling policy, template, and total memory budget fixed. Record proposed tokens, accepted tokens, committed tokens, draft time, verification time, rollback work, extra expert reads, and peak state. Start with a short draft length and only increase it when acceptance and memory justify it.

```text
speculative speed = committed target tokens
                  / (draft + verify + rollback + exposed I/O time)
```

A smaller draft model can evict valuable target experts. Multi-token verification may activate a wider expert union. Rejected drafts can pollute both expert and state caches. Therefore high acceptance alone does not prove a speedup. Confirm target verification and rollback correctness, and compare stochastic distributions when stochastic sampling is in scope; equal random seeds alone are insufficient proof.

### Tune GPU keepalive and CPU workers last

MoeShift's keepalive behavior was motivated by observations on M4 Pro. Test it on this M3 Pro only if short I/O gaps and low GPU clocks appear to matter. Measure sustained performance, power, and thermals; a quick speed increase followed by thermal throttling is not a strict upgrade. Compare the preset's 8 workers with 6 only if CPU execution or read conversion is a limiting term. [Keepalive design and scope](https://github.com/davide-troiani/moeshift/blob/v0.1.0-alpha.6/docs/architecture.md#8-gpu-keepalive)

## 7 Testing mechanisms and decision gates

All hardware checks and measurements run through `mrig`, in the background, with `--purpose` and `--estimate`. Check existing runs/results first. Use the cheapest instrument that answers the next unresolved question. No broad suites or Cartesian parameter sweeps are needed.

### Correctness checks

| Check | Mechanism | Acceptance condition |
|---|---|---|
| Artifact compatibility | Header/tensor inspection plus one bounded load | Correct architecture, shapes, mixed encodings, revision, and template; no silent fallback or conversion. |
| Tokenization | Compare token IDs and rendered prompts against a pinned compatible reference | Match representative code, Unicode, whitespace, tool messages, BOS/EOS, and stop semantics. |
| Quantized expert execution | Focused CPU/Metal checks for the actual gate/up/down formats and shapes | Agreement with the reference within a justified numeric tolerance; mixed formats exercised. |
| Streaming parity | Teacher-force the same tokens with resident and streamed IQ1_S, or a safe fixed reference path | Correct logits/routes/state; investigate every deterministic token divergence. |
| Quality-quant parity | Fixed Q4 bytes with streaming-policy changes and representative model/layer checks | No unexplained numerical or semantic regression. IQ1_S success does not qualify Q4 kernels. |
| Loop/state integrity | Compare state at repeated-core transitions and later prompt continuation | Physical weights reused; logical KV/controller/convolution state kept independent. |
| Long context | Actual occupied 32K, followed by 64K only after success | No truncation, wrong positions, cache wrap errors, or lost early-context information. |
| Lifetime and failure | Focused tests for cancellation, eviction, short read, and context growth | No use before publication, eviction of active buffers, stale mappings, leaked pins, or continued generation after failed reads. |

For scheduling changes that leave arithmetic order unchanged, seek exact logit/state equality. If kernels reorder floating-point work, establish tolerances from the reference's own repeatability and inspect errors; do not use arbitrary permissive tolerances. Greedy text alone can hide incorrect logits and is not a complete correctness test.

A fully resident four-bit reference may not fit this laptop with the normal applications open. Do not force a large swap-heavy reference run. Use bounded layer/operator comparisons and a safe reference configuration with the same weights and token history. Record whether that reference is itself streamed. Large full-vocabulary logit dumps at every long-context position are unnecessary: retain selected positions, top-logits/full rows where needed, and compact state digests.

### Quality checks

Keep engine correctness separate from model usefulness. For a weight or KV precision choice, use a small frozen set tied to the intended tasks: a cross-file code change with executable checks, structured tool-call arguments, a document question requiring evidence from multiple locations, and retrieval with distractors at relevant context positions. Begin with one case per affected behavior and expand only when there is a failure or unresolved difference.

Score code with tests, JSON/tool calls with parsing and schema checks, and document answers against cited source facts. For retrieval, use exact expected facts/identifiers. Keep reasoning effort and output limits fixed. Record both the target quant's baseline score and the candidate's score; offloading does not excuse quality regression. Publisher top-1 agreement is not a substitute for these task checks.

### Performance measurements

The core matrix is staged, not all run at once:

| Cell | Input and generated output | What it decides |
|---|---|---|
| Small smoke | Roughly 256 input and 32 output tokens | Can this exact artifact/path execute correctly? No speed claim. |
| Short diagnostic | Roughly 2K input and 256 output tokens | Is the basic bottleneck compute, dispatch, or expert I/O? Run only when needed for attribution. |
| Primary 32K context | Roughly 30,720 post-template input and 512 output tokens | Fresh long-prompt performance and sustained decode within a 32,768 capacity. |
| Agent follow-up | Same long prefix plus a 512–1,024-token tool result and 256 output tokens | Prefix reuse, phase transitions, and useful agent latency. |
| 64K context | Roughly 63,488 post-template input and 512 output tokens | Capacity growth and long-context correctness after the 32K winner is established. |
| Sustained session | Repeated realistic turns for about 10 minutes, only for finalists | Thermal behavior, cache drift, memory growth, and tail latency. |

The exact fixture token counts should be checked after applying each model's template. Across different models use the same task/text and report actual tokens; tokens/s across different tokenizers is not a pure hardware comparison. Across engines running the same checkpoint, require identical tokenized prompts.

For a candidate that passed correctness, use alternating same-host rounds, initially three per side. Report the median and the measured resolution. A result below the combined noise is unresolved, not a speedup. Retain the incumbent unless another measured axis improves without regression. A practical engineering target might be at least 10% faster or a useful memory saving, but acceptance also requires the difference to exceed observed noise.

Measure these fields:

| Group | Required observables |
|---|---|
| User latency | Model-load time, first generated token, first visible answer token if reasoning is separate, prefill time, decode duration, full response time |
| Generation | Committed target tokens, decode tokens/s, median and p95 inter-token gap, finish reason and any early EOS |
| Memory | Peak physical footprint for the entire process tree, phase-specific engine allocations, KV bytes, expert-cache bytes, scratch/staging, compressor and swap deltas |
| Storage | Required and speculative logical bytes, physical disk reads when attributable, read size distribution, queue wait, read service time, useful-prefetch ratio |
| Compute | GPU busy/idle intervals, dequantization/conversion cost, synchronization/launch overhead, thermal/power state |
| Provenance | Runtime/binary identity, model hash/revision, config, prompt hash/token count, cache state, power/background-app state, rig record |

Use server-side token accounting where available. SSE messages can contain multiple tokens or only metadata and must not be counted as tokens. Define decode rate consistently; if measuring the interval from the first to last emitted token, use `(N - 1) / interval`. Preserve native engine timing separately. Proposed speculative tokens never count as output throughput.

### Control cache state explicitly

Distinguish four states:

1. Cold process/kernel initialization.
2. Empty engine expert cache, with OS file-cache state reported separately.
3. Warm model/expert cache with fresh prompt state.
4. Reused prompt/KV prefix.

Restarting a server does not make the SSD cold. Reading a model for hashing or verification can warm its pages. Never claim physical SSD bandwidth from logical read bytes alone. Prefer validated uncached reads or storage counters; if cold storage state cannot be established, label the result warm or uncontrolled. Do not use a global cache purge as a routine benchmark step.

Magnitude's source already has an uncached I/O owner using `F_NOCACHE` on macOS. Reuse that instrumentation. A storage microbenchmark is warranted only when the runtime trace leaves a genuine bandwidth/latency question; use representative expert-sized reads and concurrency, not a large sequential-file score.

### Failure and stop rules

Stop and inspect on unsupported architecture/type, incorrect output, failed state comparison, OOM, persistent swap growth, severe memory pressure, or an estimate overrun. A prior nonzero swap allocation is not by itself failure; the relevant signal is new sustained paging and impaired latency during the run.

If a run exceeds its estimate, inspect phase and logs before extending it with a reason. If storage throughput is idle while the GPU is waiting, inspect synchronization and read scheduling before adding workers. If SSD traffic is already near the calibrated ceiling and miss rate cannot fall within the safe budget, further kernel tuning cannot remove that I/O limit.

## 8 Integrate with the existing rig

Use existing instruments rather than another benchmark framework:

- `mrig check` for required Clippy/build checks after source changes.
- `mrig test` for focused relevant unit/operator tests.
- `mrig kernel` only when an identified kernel is limiting.
- `mrig step` for Magnitude model-step attribution, expert traffic, and a forced weight cap.
- `mrig bench` for supported engines' HTTP serving measurements.
- `mrig script --hold timed` for a first external-runtime comparison that has no existing instrument.
- `mrig ab` for evidence comparing two builds on the same host in alternating rounds.

The following are command shapes for implementation, not already executable test assets. The external script and any new adapter still need to be written; model/template selectors must exist in the rig. Resolve selectors before launching so no unrelated catalog model is downloaded.

```sh
uv run --project inference/evaluation mrig --help
uv run --project inference/evaluation mrig runs
uv run --project inference/evaluation mrig script --help

# A future script starts the exact external runtime, sends the fixed fixture,
# records timing/memory/I/O, and shuts down its worker in a finally/trap path.
uv run --project inference/evaluation mrig script \
  /absolute/path/to/external-moe-comparison.sh \
  --backend metal --host local --hold timed \
  --purpose "Compare fixed-quality Inkling configurations at occupied 32K context" \
  --estimate 20m > /private/tmp/moe-comparison.log 2>&1 &
```

Keep working while the job runs. The 20-minute estimate is an initial placeholder to replace using the preceding short run's measured prefill/decode rate. Inspect the log and rig phase at the estimate instead of automatically repeating.

For a Magnitude change:

```sh
uv run --project inference/evaluation mrig ab step \
  --base <baseline-git-ref> --candidate worktree \
  --backend metal --host local \
  --model <registered-model:variant> --template <registered-fixture:32k> \
  --weight-cap <weight-bytes> --streamed-only \
  --purpose "Does the candidate reduce Gemma expert stalls at the same weight cap?" \
  --estimate <duration> > /private/tmp/gemma-ab.log 2>&1 &
```

`--weight-cap` is a weight cap, not a process or total-engine budget. Use `--streamed-only` when a resident cell would exceed this laptop's safe memory. `mrig step --cold` controls its documented kernel/user/JIT caches; it does not by itself prove an empty OS model-file cache.

`mrig ab` compares repository sources; it does not automatically switch arbitrary external engines. A first external script can orchestrate alternating pinned binaries/configs under one timed hold and save both identities. If this comparison recurs, add external adapters or a comparable instrument with explicit config/model identity rather than accumulating opaque scripts.

A complete result should include `rig.json`, command/config snapshots, tokenized prompt identity, a machine-readable report, phase memory samples, aggregate expert-read counters, and relevant correctness evidence. Keep detailed route tracing separate from final timing runs because tracing changes the measured workload.

## 9 Development work only after baseline decisions

### Atomic Engine long context

This is the preferred first substantial Inkling engine project if MoeShift's measured result leaves a material gap. Atomic Engine already handles the model and direct GGUF streaming, so the scope is narrower than implementing the architecture in Magnitude. Its public short-context Metal example and benchmark do not establish M3 Pro or 32K performance. [Atomic Engine implementation and setup](https://github.com/AtomicBot-ai/atomic-engine/tree/fb69fff02101b048a0a165a3b502712dd2ca978f)

1. **Audit the context restriction.** Identify which allocation, serialization, scheduler, attention, and rollback assumptions rely on 4,096. Treat the rejection as an admission contract, not an arbitrary constant to delete.
2. **Generalize memory admission.** Size KV, loop state, graph buffers, and transient growth from geometry and requested context. Share physical weights and charge independent states. Reject budgets below the working floor before partially loading.
3. **Qualify non-speculative long context first.** One new boundary context and then occupied 32K; compare teacher-forced outputs, selected long positions, and state continuity. Use conservative microbatches and FP16 KV initially.
4. **Validate interrupted and repeated requests.** Include prompt continuation, cancellation, budget changes, window wrap, and any supported restore path. Ensure old buffers remain alive until all GPU readers finish.
5. **Measure at the same total budget as the incumbent.** Direct-GGUF space savings are already useful, but a speed claim needs real same-workload timing and physical-read attribution.
6. **Add DSpark later.** Audit target/draft context injection, saved loop visits, acceptance, rollback, and maximum state. Account draft allocations before target cache fills. Qualify at 32K separately from short-context results.

Exit when a pinned Atomic build passes the 32K correctness and memory gates and wins a meaningful measured axis without another regression. If generalizing context reveals large unsupported state machinery, retain MoeShift while reassessing scope.

### Magnitude improvements for Gemma

Start with the measured limiting owner:

| Evidence | Existing area to inspect | Candidate intervention |
|---|---|---|
| Excess physical reads | `inference/engine/executor/src/experts/reuse.rs`, `directory/`, `watcher/` | Improve the existing ranking/calibration or cache allocation based on route traces. |
| Unexplained low expert capacity | `experts/bank/derived.rs`, executor memory ledger and assessment | Fix accounting or duplicate residency before changing replacement policy. |
| Read latency/queue waste | `inference/seismic/io/`, `experts/source.rs`, `staging.rs` | Correct request ordering, buffering, or measured coalescing policy. |
| Prefill rereads/conversion overhead | `experts/conveyor.rs`, `landing.rs`, `compute.rs` | Amortize known prompt unions within bounded memory. |
| CPU/GPU imbalance | `experts/site.rs` and executor assessment | Correct placement/cost assumptions rather than blindly adding CPU threads. |
| Slow attention at long context | State layout and the measured attention kernel | Correct compact window handling or optimize the actual limiting geometry. |

Make the change generic to geometry/capabilities. Run the required Clippy check and appropriate targeted tests. OS-facing changes also require the repository's Windows cross-compilation check; platform-specific memory/I/O behavior keeps one owner.

### Magnitude Inkling support

Only choose this larger project if a unified Magnitude implementation is independently valuable or the existing engines cannot meet the required quality/context/performance. The necessary order is:

1. Pin the exact merged-loop architecture and reference implementation. Revalidate existing metadata/inventory/tokenizer work against it.
2. Represent physical weight identity separately from logical visit/state identity. A loop schedule alone is not executable architecture support.
3. Add the missing mathematical contracts: depth-wise attention residuals, learned loop controllers and cross-pass messages, relative-position attention and clipping, input-dependent convolution state, and the exact feed-forward activation/scales.
4. Implement bounded operator and block references. Qualify mixed expert formats on CPU and Metal with real model geometry.
5. Establish resident IQ1_S parity at short context, then state continuation across loops and local-window boundaries. Compare logits and intermediate states, not only prose.
6. Reuse the existing expert streamer with shared physical-bank identity across visits; avoid three duplicate caches for repeated weights. Keep all logical histories independent.
7. Qualify the quality quant and occupied 32K context under a safe memory budget. Only then register the family for normal serving and model acquisition.
8. Measure against the best external incumbent before optimizing kernels or adding speculative decoding.

This is multiple layers of model and executor work. The existing uncommitted crate should not be presented as a nearly complete feature merely because it recognizes the file.

## 10 Deliverables and execution order

| Order | Deliverable | Decision it unlocks |
|---|---|---|
| 1 | Pinned inventory, preserved source state, working local rig, storage plan | Reproducible experiments without accidental downloads or unrelated changes. |
| 2 | Gemma baseline plus Inkling compatibility evidence | Choose runnable paths and diagnose unsupported formats. |
| 3 | One quality Inkling quant and a successful occupied 32K request | Establish the real target before tuning. |
| 4 | Baseline report with latency, memory, SSD bytes, and correctness | Identify the largest avoidable cost. |
| 5 | One evidence-selected cache, prefill, or overlap improvement | Decide whether configuration is enough or source work is justified. |
| 6 | Qualified 32K daily profile, then 64K profile | Deploy useful long-context behavior with known limits. |
| 7 | Optional speculative decoding and sustained-session qualification | Decide whether accepted-token gains survive cache pressure and heat. |
| 8 | Atomic long-context work or Magnitude architecture work only if justified | Invest engineering effort where the baseline demonstrates a gap. |

Rough effort after dependencies/storage are available: inventory/configuration and smoke checks are hours; baseline collection and one tuning decision may take a day or two with thermal-gated runs; an Atomic long-context extension is a multi-day engineering investigation; a complete Magnitude family with new operators is potentially a multi-week project. These are scope estimates, not promised completion times.

The daily profile is ready when it runs the selected quality checkpoint at occupied 32K, preserves required task behavior, keeps the normal laptop workload responsive, avoids sustained new swap traffic, and has repeatable decode and follow-up latency. Promote 64K only after its own capacity/state check. If two configurations are within measurement noise, keep the simpler one.

## 11 Additional repositories worth using selectively

| Source | Useful contribution | Boundary |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Pinned architecture, GGUF, tokenizer, and numerical reference | CPU expert placement alone does not establish deliberate bounded SSD streaming. |
| [MoeShift](https://github.com/davide-troiani/moeshift) | Prepared expert stores, phase-specific banks, protected demand caching, and resident/missing overlap | Validate the exact architecture, quant, context, and preset. |
| [Atomic Engine](https://github.com/AtomicBot-ai/atomic-engine) | Original-GGUF reads, engine budget accounting, Inkling core reuse, and Metal DSpark work | Current budgeted Inkling context restriction is a prerequisite project. |
| [Jakevin moe-offload](https://github.com/Jakevin/moe-offload) | Route-trace analysis and offline cache/prefetch evaluation ideas | Its documented model paths are research implementations, not evidence of compatibility with these two checkpoints. |
| [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) | Background for the lookahead scheduling foundations used by MoeShift | Use the current integrated runtime first; another fork adds a separate maintenance and compatibility burden. |

Do not start by installing several more runtimes or converting both models into another format. Additional engines become useful only if they answer a specific gap: a required architecture, better long-context state handling, lower physical reads, or an exact speculative path.

The first actionable sequence is therefore: repair the rig environment, baseline Gemma in Magnitude, qualify the existing Inkling in MoeShift, make storage room for one four-bit Inkling, establish occupied 32K correctness, and optimize whichever measured component limits that workload. Keep Atomic's long-context extension as the most focused next development option.

