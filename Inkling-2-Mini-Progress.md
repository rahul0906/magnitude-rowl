# Inkling 2 Mini implementation progress

Updated: 2026-11-26

## Current status

Artifact inspection, logical loop scheduling, and tokenizer recognition are implemented as a safe foundation. An independent bounded audit matched the checkpoint’s complete header and tensor directory to the strict profile. The downloaded model was tested through mrig and local llama.cpp, but that runtime rejects the checkpoint with `unknown model architecture: inkling2_loop` before a generation can start. The Rust real-artifact test is added but unrun because Cargo cannot resolve its dependencies offline. Magnitude still does not register the model for serving because the shared decoder and executor lack several architecture-specific operations.

## Checkpoint pinned for this work

- Repository/revision: `AtomicChat/Inkling-2-Mini-GGUF@309509848b3151419aa26d069eb03db57fd29a00`
- File: `Inkling-2-Mini-AD-IQ1_S.gguf`
- Local path: `/Users/rahul/.cache/huggingface/hub/models--AtomicChat--Inkling-2-Mini-GGUF/snapshots/309509848b3151419aa26d069eb03db57fd29a00/Inkling-2-Mini-AD-IQ1_S.gguf`
- Size: 6,881,463,904 bytes; GGUF v3; 704 tensors; 62 metadata entries.
- Header-only audit read 11,977,671 bytes and matched every tensor name and shape without reading any weight payload.
- Metadata describes 24 physical layers, loop core `[6, 18)`, three passes, and 48 logical visits. Encoding counts: F32 334, Q8_0 320, IQ1_S 24, Q2_0 24, Q4_K 1, Q6_K 1.
- SHA-256: `4d3cab7e88016548136e13f7ec465db7fc49ab8cf9559a60f3daad1f3f15c9d5`.

## Implemented

- Added `inference/engine/families/inkling2`, a workspace crate with a checked, allocation-free visit schedule and strict metadata interpretation.
- Added a physical tensor inventory that validates all tensors, shapes, local/global attention variants, attention-residual queries, global convolution projections, and the 13 controller tensors on each loop-core layer. Unknown or missing tensor names are rejected.
- Added synthetic metadata and tensor-inventory tests, plus an ignored integration test that can inspect a downloaded GGUF using `INKLING2_GGUF=...` without loading its weights.
- Added the `gpt2` + `inkling2` tokenizer profile, matching the reference GPT-4o byte-BPE split, ignored merges, and no implicit BOS by default. Added a profile regression test.
- Extended the routed CPU expert parity test to cover Inkling’s mixed IQ1_S gate/up and Q2_0 down weights.
- The loop schedule dummy test checks the 48 logical visits and per-visit history layout; physical weights remain named by their stored layer.

## Remaining runtime gates

- No `ModelFamily` is registered yet. The current shared `ModelDefinition`/executor contracts do not express Inkling’s depth-wise residual mixing, cross-pass messages and learned loop controllers, learned relative-position bias and clipping, or its input-dependent global-attention convolution taps. Serving it through existing operators would misrepresent its math.
- The checkpoint’s expert encodings are recognized by the GGML/Seismic format registry. The focused routed IQ1_S/Q2_0 parity case is added but remains unrun, and per-backend support for this model’s complete mixed-quantized layout is not qualified.
- Tokenizer piece IDs still need comparison to a pinned reference implementation. Chat-template and stop-token behavior also need golden tests against the checkpoint.
- Expert RAM/SSD streaming, memory accounting, image input, and performance qualification remain future stages after resident numerical parity.

## Validation

- Passed: bounded local GGUF header/tensor audit; exact 704 names and shapes, exact encoding counts; no weight payloads read.
- Passed: all 36 `inkling2_loop.*` metadata keys in the downloaded header are explicitly accounted for by the strict metadata reader.
- Passed: standalone `rustc --test` schedule checks: 2 passed.
- Passed earlier: `cargo metadata --no-deps` recognizes the new workspace crate.
- Passed: stable `rustfmt --check` on the new crate and artifact test, parser checks on modified existing Rust files, and `git diff --check`.
- Added but not run: metadata/inventory Rust tests, tokenizer-profile test, real-artifact Rust test, and the mixed IQ1_S/Q2_0 routed-site parity test.
- Blocked: `cargo test --offline -p magnitude-family-inkling2` fails before compiling because the workspace’s `magnitude-engine` requires uncached `opentelemetry`. A temporary narrow test scope got past that workspace dependency but then stopped at the uncached `image v0.25.8` crate. DNS resolution to crates.io is unavailable, so neither missing crate could be fetched.
- Ran (inference not passed): mrig setup and a real local `mrig script` model load using the installed llama.cpp runtime. The sandboxed attempt first failed because llama.cpp could not bind its internal ephemeral HTTP port; with that port available, the runtime reached model loading and reported `unknown model architecture: inkling2_loop`. It produced no tokens. The tested Homebrew build was llama.cpp 0.6.0 (build 11429, commit `d81235049`). The run record is `inference/evaluation/results/20261126T074753Z-script-ff57df/`.
- Checked: the other local llama.cpp executable is 0.7.0-dev (build 12178, commit `03ec60072`); its binary contains no `inkling2_loop` architecture identifier, so it is not a viable alternate runtime for this checkpoint.
- Environment note: `uv run --project inference/evaluation mrig` still cannot sync the pinned llama.cpp Python source because GitHub DNS is unavailable. I ran the mrig source CLI from cached Python packages with temporary host/state paths under `/private/tmp`; no repository code was changed for this workaround.

## Reference source

- ggml-org’s [Inkling 2 GGUF loader and graph](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/inkling2.cpp)
- ggml-org’s [fixed loop schedule and shared physical weights](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/inkling2-loop.cpp)
- ggml-org’s [tokenizer pre-tokenizer mapping](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-vocab.cpp)
