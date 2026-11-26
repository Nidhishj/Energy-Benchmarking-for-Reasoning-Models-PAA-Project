# Plan: systolic arrays, i.e. the same workloads on a systolic-array accelerator (simulation)

Status: proposal (2026-11-27). Nothing has been run.
Companion files: `remaining_work.md`, `plan_ssm_hybrid.md`, `plan_jepa.md`.

## 1. Why, and what we can and cannot measure

- A **systolic array** is a grid of multiply-accumulate units that pass operands to their neighbours each cycle. It's the core of the TPU's MXU and of most NPU and LLM accelerators.
- It reaches peak efficiency only on large matrix-matrix products (GEMMs), because each weight loaded into the grid is reused across many inputs.
- NVIDIA tensor cores are matrix units, not a classic weight-stationary systolic array.
- **All our measurements are on GPUs.** That fits the course (*Programming AI Accelerators*), but leaves open whether our findings hold on a systolic accelerator:
  1. batch-1 decode is memory-bound, about 1 FLOP per byte;
  2. attention grows with the KV cache while the MLP is flat;
  3. thinking costs about 6× because answers are longer.
- **We cannot measure real systolic-array energy.**
  - We have no TPU or NPU hardware.
  - Cloud TPUs expose no per-chip energy counter equivalent to NVML that we know of; verify before ruling this out.
- **So: simulate**, with our exact workload shapes as input.

**Why it's interesting.** At batch-1 decode, every weight GEMM becomes a GEMV (M = 1).
- A weight-stationary N×N array then uses about 1/N of its MACs: for example 1/128 for a 128×128 MXU.
- Energy is dominated by SRAM/DRAM traffic, and by loading weights into the array, not by MACs.
- Attention at decode is also two GEMVs against the KV cache (qKᵀ and pV), streamed from memory.
- **Mamba's selective scan is a recurrence, not a GEMM**, so it maps poorly onto a GEMM systolic array. Mamba-X builds a dedicated systolic scan array for this reason.

This is exactly where our GPU results and accelerator design meet.

## 2. Existing work (literature check, 2026-11-27)

| Work | What | Relevance |
|---|---|---|
| **SCALE-Sim v3** ([2504.15377](https://arxiv.org/pdf/2504.15377)) | Cycle-accurate systolic-array simulator with modular memory and data layout; notes that Timeloop+Accelergy give energy but not cycle accuracy | The tool: cycles, utilisation, SRAM/DRAM accesses |
| **Timeloop + Accelergy** | Mapping-space search plus component energy estimation (energy per access, per MAC) | The tool: energy |
| **LLMCompass** (ISCA 2024) | LLM inference hardware evaluation, SCALE-Sim backend | More compute helps prefill a lot but **barely helps decode**; big arrays are hard to keep busy |
| **SPAD** ([2510.08544](https://arxiv.org/pdf/2510.08544)) | Separate prefill and decode chips | Smaller arrays (16×16) plus vector units are more efficient for decode |
| **SISA** ([2603.29913](https://arxiv.org/pdf/2603.29913)) | Scale-in systolic array for GEMM including decode | Reports up to 93% energy-delay-product reduction vs a monolithic array (self-reported) |
| *LLM inference energy-performance tradeoffs* ([2501.08219](https://arxiv.org/pdf/2501.08219)) | GPU characterisation | Decode dominated by weight and KV-cache access |
| Mamba-X ([2508.02977](https://arxiv.org/html/2508.02977v1)), FastMamba, eMamba | SSM accelerators | Systolic scan arrays for Mamba |

**Gap.** None of these evaluates **long reasoning traces**: per-token energy vs context up to 32k, thinking on vs off, or the attention vs MLP split as context grows, on a systolic array. None checks a simulator against **measured** component energy the way we have it on GPUs.

## 3. Experiments

### Y1. Qwen3-8B decode step on a simulated systolic accelerator, vs context L
- **Workload:** the exact shapes of one decode step at L = 256 … 32k, batch 1:
  - q/k/v/o projections: 4096×4096 and 4096×1024;
  - MLP: 4096×12288 ×2 and 12288×4096;
  - attention qKᵀ over L cached keys (32 query heads, 8 KV heads, head_dim 128) and pV;
  - LM head: 4096×151,936.
- **Hardware configs:**
  - a TPU-like weight-stationary 128×128 array with large on-chip SRAM and HBM-class bandwidth;
  - SPAD-like small 16×16 arrays plus a vector unit.
- **Tools:** SCALE-Sim v3 for cycles, utilisation and memory accesses; Accelergy (or SCALE-Sim's energy hooks) for energy per access and per MAC.
- **Outputs:**
  - energy per token vs L, split into attention vs MLP vs LM head, and MAC vs SRAM vs DRAM;
  - array utilisation;
  - **the attention/MLP crossover L on the accelerator vs our measured ~12k on GPUs.**
- **Validation:** compare simulated component *shares* vs L with our measured GPU shares (attention 25% → 67%, MLP 57% → 26% on the PRO 5000). Absolute joules aren't comparable; shares and trends are.

### Y2. Batch and prefill (where systolic arrays shine)
- Same shapes with batch B = 1, 8, 32, 128 (M = B), and prefill of 512–8k tokens.
- **Question:** at what batch size does decode stop being memory-bound on the array, and how does the attention share change? (The KV cache grows with B but isn't shared, while weights are reused.)

### Y3. Mamba-2 / hybrid decode step on the same configs (links to `plan_ssm_hybrid.md`)
- Map the Mamba mixer: in_proj and out_proj are GEMVs, conv1d and the selective scan go on the vector unit or a scan array.
- **Question:** does the SSM's flat per-token cost hold on a GEMM systolic array, or does the scan (poor fit) offset the saved KV-cache reads? Compare with Y1 at L = 32k.

### Y4 (optional). Whole reasoning answers
- Price saved answers (AIME on/off, correct/wrong) with Y1's simulated c(L), as `compare_on_off.py` does with the GPU curve.
- **Question:** is "thinking costs ~6×, wrong answers cost the most" the same on an accelerator?

## 4. Effort and order
- Install SCALE-Sim v3 (Python, CPU only) and Accelergy/Timeloop (Docker image available): ~½–1 day.
- Write a topology generator from our model configs (Qwen3-8B, Nemotron-Nano-9B-v2): ~½ day.
- Y1 + Y2: CPU simulations of minutes to hours per config. **No GPU needed**, so it can run in parallel with GPU work.
- Y3 needs a scan-mapping decision; follow Mamba-X or treat the scan as vector-unit work.

## 5. Risks and limits
- These are simulated, not measured. Energy-per-access constants come from Accelergy's technology tables, so absolute joules are approximate. Report shares, trends and crossovers, and validate trends against our GPU measurements.
- SCALE-Sim models GEMMs well. Attention's softmax, norms and the SSM scan need vector-unit modelling, which is rougher.
- Results depend strongly on the assumed SRAM size (can the KV cache or weights stay on chip?). Sweep it.
