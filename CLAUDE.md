# CLAUDE.md — Component-Level Energy Benchmarking of Reasoning LLMs

Course project, *Programming AI Accelerators* (Team 3: Aaditya Bhatia, Nidhish Jain, Virat Garg).
Full proposal: `proposal_2pager-1.pdf`. Reference papers: `papers/`.

## Goal (one paragraph)
Apply CLEAR's component-level energy attribution (Jain et al., arXiv:2510.02810: cache a module's input →
replay it N times → read NVML energy → validate) to **long, growing reasoning traces**, to find where the energy of
"thinking" goes (Attention vs MLP vs rest) as context grows. Inference only, no training.

- **Baseline:** whole-model energy + accuracy, reasoning vs non-reasoning (reproduce Jin et al. arXiv:2505.14733
  Base-vs-RT trend at small scale). Calibrates the meter before any component number is trusted.
- **Exp A:** per-token Attention vs MLP energy at context checkpoints (e.g. 256, 1k, 2k, 4k, 8k, 16k): find the crossover.
- **Exp B:** correct vs incorrect traces — is the extra energy of wrong answers Attention-concentrated?
- **Exp C (main):** forced early stop (inject `</think>` at budgets) → accuracy vs energy, and which component's energy drops.
- Priority if short on time: **C > A > B**; Qwen3 cross-check and per-layer heatmap are cut first.

## Models (Hugging Face, fp16)
| Role | Model | Notes |
|---|---|---|
| Main reasoning model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | fits one 11 GB GPU with long KV cache |
| Non-reasoning baseline | `Qwen/Qwen2.5-Math-1.5B-Instruct` | same architecture family as the distill's base |
| Native think on/off | `Qwen/Qwen3-1.7B` (stretch: `Qwen/Qwen3-4B`) | toggle via `enable_thinking` in `apply_chat_template` |
| Stretch only | `DeepSeek-R1-Distill-Qwen-7B` | ~15 GB fp16, **does not fit one 2080 Ti** |

R1-Distill recommended sampling: temperature 0.6, top-p 0.95, no system prompt, ask for `\boxed{}` answer.
Qwen3's `thinking_budget` is a feature of Qwen's API/serving stacks, not of plain HF `generate`. Under HF, the budget
is enforced with the same custom stopping logic as for R1-Distill.

## Datasets
MATH-500 (`HuggingFaceH4/MATH-500`), GSM8K (`openai/gsm8k`), AIME24 (small, hard, long traces), HumanEval for code (optional).
Grade math with `math-verify`. Start with a fixed 50–100 problem subset of MATH-500, with seeds recorded.

## Hardware: ADA cluster (read before writing any job script)
- Node **gnode074**, partition `u22`, account `cvit`: 3× **RTX 2080 Ti, 11 GB, Turing**.
  - **No bf16** → run fp16. Watch for inf/NaN logits in Qwen fp16 and check outputs for garbage.
  - **No FlashAttention-2** → use `attn_implementation="sdpa"` (or `"eager"` when hooks need attention internals).
  - NVML total-energy counter (`nvmlDeviceGetTotalEnergyConsumption`, mJ) works on Turing. The power sensor
    updates at about 100 ms, which is why CLEAR replays a module many times instead of timing it once.
- `/ssd_scratch` is node-local to gnode074, so every job needs `--nodelist=gnode074`. Keep sbatch files and logs in `~/` (shared).
- u22: `MaxMemPerCPU=3000` MB. Use 1 GPU per job with `--cpus-per-task=10 --mem=24G`.
- Set `HF_HOME=/ssd_scratch/nidhishj/hf_cache` so model weights don't fill `$HOME`.
- Interactive GPU shell:
  `srun -A cvit -p u22 --nodelist=gnode074 --gres=gpu:1 --cpus-per-task=10 --mem=24G --time=06:00:00 --pty bash`

## Energy measurement rules (non-negotiable for any reported number)
1. Energy = difference of the NVML energy counter around the region, **minus idle power × time**. Report both raw and idle-subtracted values.
2. Warm up before measuring (CUDA init, first kernels). Call `torch.cuda.synchronize()` before every counter read.
3. Record per run: GPU index, temperature, SM clock, power limit, driver, torch/transformers versions, seed, and the full config.
   Throttling at high temperature changes the numbers.
4. Component energy (CLEAR): hook `layer.self_attn` / `layer.mlp` / norms. Cache the input **and the KV cache at the
   target context length**, then replay the module enough times that the region runs for ≥1–2 s. Divide by the number of replays.
5. Validate as CLEAR does: the sum of the component energies should capture ≥90% of the whole forward-pass energy, with
   std. dev. <9.5% across repeats. Report the achieved numbers even when they miss these thresholds.
6. Other jobs may share the node. Use only the GPU you were allocated (`CUDA_VISIBLE_DEVICES`) and note node load in the logs.

## Planned layout (create as needed)
```
src/energy/    NVML meter (context manager), idle calibration, Zeus cross-check
src/gen/       generation with forced-</think> budgets, answer extraction, grading
src/clear/     hooks, activation + KV caching, replay, validation
configs/       YAML per experiment
scripts/       entry points (baseline.py, exp_a.py, exp_c.py, exp_b.py)
slurm/         sbatch templates (copy to ~/ to submit from the login node)
results/       raw JSONL per run (git-ignored when large), plus figures/
```

## Conventions
- Hugging Face `transformers` + forward hooks. **Do not use vLLM/SGLang** for component measurement, since there is no module access.
- Every run writes one JSONL row per query: tokens in/out, time, energy (raw and idle-subtracted), correctness, config hash.
- Results and plots must be reproducible from saved JSONL. Don't hand-edit numbers.
