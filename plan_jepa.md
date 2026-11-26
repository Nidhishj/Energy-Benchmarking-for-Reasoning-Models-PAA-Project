# Plan: energy benchmarking of JEPA-type architectures

Status: proposal and feasibility check (2026-11-27). Nothing has been run.
Assumption: "jev type architecture" = **JEPA** (Joint Embedding Predictive Architecture, LeCun et al.). Correct this file if something else was meant.

## 1. What JEPA is

- A generative LLM is trained to **reconstruct the input space**: predict the next token over the whole vocabulary.
- A JEPA is trained to **predict the representation (embedding) of a target from the representation of a context**, in latent space. It never reconstructs pixels or tokens.
- Its parts are:
  - a context encoder;
  - a target encoder (usually an EMA copy, to avoid collapse);
  - a predictor that maps context embeddings to predicted target embeddings.

| Model | Domain | What happens at inference |
|---|---|---|
| I-JEPA, V-JEPA, V-JEPA 2 | Images / video | Encoder produces embeddings. No generation at all, so used as a feature extractor or world model. |
| **VL-JEPA** ([2512.10942](https://arxiv.org/pdf/2512.10942)) | Vision-language | Predicts **text embeddings** instead of generating tokens. A small "y-decoder" turns an embedding into text **only when text is needed**. "Selective decoding" decodes only when the predicted embedding stream changes enough. The authors report lower inference latency than generative VLMs trained on the same data. |
| **LLM-JEPA** ([2509.14252](https://www.arxiv.org/abs/2509.14252), ICLR 2026) | Text LLMs | Adds a JEPA loss (distance between the predicted embedding of one view and the embedding of the other view, e.g. NL ↔ SQL) to the normal next-token loss, during fine-tuning or pretraining. Reports better accuracy on NL-RX, GSM8K, Spider and RottenTomatoes across Llama3, OpenELM, Gemma2 and Olmo. **At inference the model is still a standard autoregressive transformer.** The JEPA predictor is a training-time objective. |

## 2. Has JEPA energy been benchmarked?

From the literature check on 2026-11-27, **no inference-energy measurement of JEPA models was found.**
- VL-JEPA reports latency and the number of decoding operations saved by selective decoding, not joules.
- LLM-JEPA reports accuracy; there is no efficiency claim at inference, since the architecture is unchanged.
- Other hits (TI-JEPA, ACT-JEPA, Distributed JEPA) are other domains and don't measure energy.

So the question is open, but the two cases are very different:

| | LLM-JEPA | VL-JEPA |
|---|---|---|
| Does the inference compute differ from the baseline? | **No**: same transformer, same per-token cost | **Yes**: embedding prediction + optional small decoder vs full autoregressive generation |
| Where an energy difference could come from | Only from output **length** and **accuracy** (energy per correct answer) | Fewer generated tokens (selective decoding), smaller decoder, no vocabulary softmax per step |
| Fits our method (CLEAR component replay, NVML) | Yes, identical tooling to Qwen3 | Yes, but new components (vision encoder, predictor, y-decoder) |
| Fits the project's topic (reasoning energy) | Partly: GSM8K reasoning | Weakly: video/vision-language, not math reasoning |
| Inference-only (as in `CLAUDE.md`) | Only if JEPA-tuned **checkpoints are released**; otherwise we'd have to fine-tune (training) | Only if checkpoints are released |

General inference-energy benchmarks also don't cover JEPA:
- *Where Do the Joules Go?* ([2601.22076](https://arxiv.org/html/2601.22076)), with 46 models on H100/B200;
- the *ML.ENERGY Benchmark* ([2505.06371](https://arxiv.org/html/2505.06371v2));
- *MLPerf Power* ([2410.12032](https://arxiv.org/pdf/2410.12032)).

They cover LLMs and diffusion models. **So JEPA inference energy, and certainly its component split, is unmeasured.**

### 2b. Released JEPA models from Meta (Facebook)

| Model | Where | Notes |
|---|---|---|
| **V-JEPA 2** (ViT-L / ViT-H / ViT-g, 64 frames, 256 px) | Hugging Face: `facebook/vjepa2-vitl-fpc64-256`, `facebook/vjepa2-vith-fpc64-256`, `facebook/vjepa2-vitg-fpc64-256` | Native in HF Transformers since 2025-06-11 (`AutoModel`, `AutoVideoProcessor`). Video/image encoder for classification, retrieval, or as a VLM's vision encoder. |
| V-JEPA 2-AC | Meta's `facebookresearch/vjepa2` repo | Action-conditioned predictor (world model, robot planning) |
| V-JEPA 2.1 (ViT-B/L/g/G) | Community-hosted on HF (e.g. `apiantonio/vjepa2.1-*`). Official hosting was being requested (GitHub issue) | Check the current official status |
| I-JEPA | Meta repo; HF support not verified | Image encoder |
| VL-JEPA | Weights not verified | Check |
| LLM-JEPA | Code: `galilai-group/llm-jepa` / `rbalestr-lab/llm-jepa`; fine-tuned weights not verified | Check |

**V-JEPA 2 is the easiest target.** The weights are official, the model is native in HF, and at inference it's a plain **ViT encoder**:
- bidirectional, one forward pass, no KV cache, no autoregressive decoding;
- that's exactly CLEAR's original setting (CLEAR's BERT example is an encoder too).

So the CLEAR method applies with the least change:
- **components:** patch embedding (3D tubelets), attention, MLP, norms in each block, and the predictor when used;
- **workload knob:** input tokens = frames × patches per frame. 64 frames at 256 px with 2×16×16 tubelets gives ~8k tokens. Attention cost grows with tokens², the MLP linearly, so the attention/MLP crossover vs video length is the encoder-side counterpart of our Exp A.

### 2c. Can we use the CLEAR code for JEPA?
- **V-JEPA 2 / I-JEPA (ViT encoders):** CLEAR's *method* applies directly. The authors' code (`clear_repo/`) is BERT-specific (hook names, `AutoModelForMaskedLM`), so the hook targets must change. Easier is to reuse our `energy.clear_measure` with forward pre-hooks on each ViT block's attention and MLP. No decode-cache logic needed.
- **LLM-JEPA:** a standard transformer at inference, so our existing Exp A runner works unchanged.
- **VL-JEPA:** an encoder (as above), plus a predictor and a small y-decoder. The decoder is autoregressive but small, so reuse the decode-step runner if needed.

## 3. Phase 0: feasibility (do this before any GPU work, ~½ day)

1. **Checkpoints.** Are LLM-JEPA fine-tuned weights public? Check the code repos (`github.com/galilai-group/llm-jepa`, `github.com/rbalestr-lab/llm-jepa`) and Hugging Face. Same for VL-JEPA weights and V-JEPA 2.
2. **Licence and size:** must fit one 48 GB GPU in BF16.
3. **Matching baselines:**
   - for LLM-JEPA, the same base model fine-tuned with the normal loss only (the paper's baseline);
   - for VL-JEPA, the generative VLM trained on the same data, as the paper compares.
4. **Decision rule:**
   - If no checkpoints exist, LLM-JEPA would need fine-tuning. That's outside "inference only", so discuss with the professor first. A small LoRA fine-tune on GSM8K with the paper's code is a possible exception.
   - VL-JEPA without released weights is not feasible.

## 4. Experiments (only if Phase 0 passes)

### J1. LLM-JEPA vs standard fine-tuning: does the training objective change inference energy?
- **Hypothesis:**
  - **per-token energy identical**, since the architecture is the same (control check, like our Exp B);
  - **per-answer energy differs only through answer length**;
  - **J per correct answer** differs through accuracy.
- **Setup:**
  - GSM8K (the paper's reasoning task, 200-problem fixed subset), plus MATH-200 if the base model can do it;
  - both models at batch 1, same decoding settings, same seeds;
  - NVML + Zeus, as in our B2b runner.
- **Measure:**
  - accuracy (95% CI), output tokens, J/query, J/token, J/correct;
  - E(k) checkpoints;
  - an Exp A component curve on one trace each, to confirm identical component costs.
- **Interesting outcome:** if the JEPA objective makes answers shorter or more accurate, it lowers energy per correct answer **without any architectural change**. A training objective then acts as an energy lever.
- **Cost:** ~2–4 h of GPU for a 1–3B model on 200 problems per arm.

### J2. VL-JEPA: energy of predicting embeddings vs generating text
- **Hypothesis:** for streaming tasks (video captioning or retrieval over a stream), VL-JEPA with selective decoding uses much less energy per query than a generative VLM, mostly by skipping autoregressive decoding steps.
- **Measure:**
  - J per query and J per video-second;
  - quality metric from the paper (caption or retrieval score);
  - **component split** with CLEAR replay: vision encoder, predictor, y-decoder vs the generative VLM's vision encoder, LLM decoder layers, LM head.
- **Key sweep:** the selective-decoding threshold, from always decode to rarely decode, giving an **energy vs quality curve**. This mirrors our thinking-budget curve.
- **Cost:** depends on model size and dataset. Estimate after Phase 0.

### J0 (first, feasible now). V-JEPA 2 component energy vs input size
- **Models:** `facebook/vjepa2-vitl-fpc64-256` (and ViT-H / ViT-g if time), BF16, SDPA, batch 1, inference only.
- **Workload:** video clips at 8, 16, 32 and 64 frames (and 256 vs 384 px), so the encoder sees ~1k to ~8k+ tokens.
- **Measure:**
  - whole-model energy per clip and per frame;
  - CLEAR component split per block (attention vs MLP vs norms vs patch embedding), with layer-check blocks;
  - %Capture ≥ 90%, T = 20, ~2 s trials.
- **Questions:**
  - where does an encoder's energy go as the video gets longer?
  - where is the attention/MLP crossover in tokens? (Compare with our decode crossover at ~12k.)
  - prefill-like, compute-bound behaviour vs our memory-bound decode: does the card still sit at its 300 W cap?
- **Baseline for "JEPA vs non-JEPA":** a same-size supervised or MAE-pretrained ViT video encoder (e.g. VideoMAE). The architecture is the same at inference, so the per-token energy is expected to be identical, and any difference comes from resolution, tubelet size or token count. Report that honestly: JEPA is a *training* paradigm, not an inference-time architecture change, for V-JEPA.
- **Cost:** a few GPU-hours. Only a small sample of clips is needed (any public video dataset, e.g. a few hundred clips from Kinetics or SSv2).

### J3 (optional). V-JEPA 2 encoder energy per frame vs a generative video model's encoder
Only relevant if we position the work as "predictive vs generative architectures" in general. Lower priority.

## 5. Recommendation

- **Priority: SSM/hybrid plan (`plan_ssm_hybrid.md`) first, JEPA second.** The SSM plan directly tests our main mechanism (KV-cache-driven attention growth during long reasoning) and has ready reasoning models with thinking toggles. JEPA is either the same architecture at inference (LLM-JEPA) or a different domain (VL-JEPA).
- **Within JEPA, J0 (V-JEPA 2) first:** official weights, native HF, CLEAR's own setting (an encoder). Nobody has published its energy.
- **J1 (LLM-JEPA) next** if fine-tuned checkpoints exist:
  - it uses our existing tooling unchanged;
  - it's a cheap, clean question: does a training objective change inference energy through length and accuracy alone?
- Do J2 only if VL-JEPA weights are public and the team wants a vision-language extension.
- State this framing clearly: V-JEPA 2 and LLM-JEPA are standard transformers at inference, so the energy question is about workload (tokens, decoding avoided), not a new kind of layer. VL-JEPA's selective decoding is the only true inference-time change.

## 6. Risks
- No released checkpoints, so we'd need training, which conflicts with the inference-only scope.
- LLM-JEPA's reported gains are on small, fine-tuned task models. The effect on answer length may be negligible, and then J1 gives a null result. That's fine as a control, but not a headline.
- VL-JEPA tasks and metrics differ from math reasoning, so comparisons with our Qwen3 results would be qualitative only.
