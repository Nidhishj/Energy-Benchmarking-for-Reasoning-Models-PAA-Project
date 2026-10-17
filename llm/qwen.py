"""Qwen3 + MATH500 helpers shared by B2b and B3 (baselines.md sections 1.4, 3, 4).

Thinking on/off uses Qwen3's chat-template switch (`enable_thinking`), so both modes run the same weights.
"""

import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
SUBSET = REPO / "configs" / "math500_subset.json"

DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}

# Qwen3 model-card sampling settings for each mode (greedy decoding is discouraged in thinking mode).
DECODING = {
    "on": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "off": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
}
MAX_NEW_TOKENS = {"on": 8192, "off": 2048}

# Qwen's recommended instruction for math problems.
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def load_model(path: str, dtype: str, attn: str, device: str):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPES[dtype], attn_implementation=attn)
    model = model.to(device).eval()
    return tok, model


def model_info(model, model_id: str, path: str, dtype: str) -> dict:
    return {"id": model_id, "local_path": path, "revision": getattr(model.config, "_commit_hash", None),
            "class": type(model).__name__, "dtype": dtype, "attn_impl": model.config._attn_implementation,
            "n_layers": model.config.num_hidden_layers}


def load_subset(path: Path = SUBSET) -> dict:
    return json.loads(Path(path).read_text())


def prompt_ids(tok, problem: str, mode: str, device: str) -> torch.Tensor:
    messages = [{"role": "user", "content": f"{problem}\n{MATH_INSTRUCTION}"}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=(mode == "on"))
    return tok(text, return_tensors="pt").input_ids.to(device)


def generation_kwargs(tok, mode: str, max_new_tokens: int) -> dict:
    return {"do_sample": True, **DECODING[mode], "max_new_tokens": max_new_tokens,
            "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id}


def seed_for(problem_id: str, mode: str, trial: int) -> int:
    """Same seed for the same (problem, mode, trial) on every GPU, independent of run order."""
    return int(hashlib.sha1(f"{problem_id}|{mode}|{trial}".encode()).hexdigest()[:8], 16)


def split_output(tok, out_ids: list[int], mode: str, max_new_tokens: int) -> dict:
    """Thinking/answer split of one generation.

    think_end_pos: index of `</think>` in the generated tokens (None if absent).
    natural_len:   thinking tokens before `</think>`, counted from the first generated token (so it includes the
                   opening `<think>` tag). None when thinking was cut off by the token cap: the true length is
                   unknown, so Experiment C's percentage budgets are undefined for that run. 0 in thinking-off mode.
    """
    think_end_id = tok.convert_tokens_to_ids("</think>")
    eos_ids = set(tok.eos_token_id if isinstance(tok.eos_token_id, list) else [tok.eos_token_id])
    ended_with_eos = bool(out_ids) and out_ids[-1] in eos_ids
    hit_cap = len(out_ids) >= max_new_tokens and not ended_with_eos
    pos = None
    if think_end_id in out_ids:
        pos = len(out_ids) - 1 - out_ids[::-1].index(think_end_id)
    if mode == "off":
        natural_len, answer_ids = 0, out_ids
    elif pos is None:
        natural_len, answer_ids = None, []
    else:
        natural_len, answer_ids = pos, out_ids[pos + 1:]
    return {
        "output_len": len(out_ids),
        "think_end_pos": pos,
        "natural_len": natural_len,
        "answer_tokens": len(answer_ids),
        "thinking_tokens": (pos + 1 if pos is not None else len(out_ids)) if mode == "on" else 0,
        "hit_token_cap": hit_cap,
        "answer_text": tok.decode(answer_ids, skip_special_tokens=True),
    }


def grade(answer_text: str, gold: str) -> dict:
    """MATH500 answer equivalence with Hugging Face math-verify (the model is asked for a \\boxed{} answer)."""
    from math_verify import parse, verify

    if not answer_text.strip():
        return {"correct": False, "pred": None, "parse_ok": False}
    gold_parsed = parse("\\boxed{" + gold + "}")
    pred = parse(answer_text)
    return {"correct": bool(pred) and bool(verify(gold_parsed, pred)), "pred": str(pred[-1]) if pred else None,
            "parse_ok": bool(pred)}
