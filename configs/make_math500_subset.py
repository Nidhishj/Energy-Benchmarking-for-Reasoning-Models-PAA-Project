"""Write configs/math500_subset.json: the fixed MATH500 problems used by B2b and B3 (baselines.md sections 3, 4).

- 6 problems per difficulty level (30 total), sampled with seed 0 from HuggingFaceH4/MATH-500 (test split).
- The same problems are used with thinking on and off, and on every GPU.
- b3_short: the shortest, middle and longest prompt (in tokens, thinking-on chat template) among the 30. They stand
  in for CLEAR Fig. 1's three input-length curves.

The problems and gold answers are stored in the file, so GPU sessions don't need to download the dataset.

Usage: python configs/make_math500_subset.py --tokenizer Qwen/Qwen3-8B
"""

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.qwen import SUBSET, prompt_ids  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    p.add_argument("--per-level", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rng = random.Random(a.seed)
    problems = []
    for level in range(1, 6):
        pool = sorted((r for r in ds if int(r["level"]) == level), key=lambda r: r["unique_id"])
        problems += rng.sample(pool, a.per_level)

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = []
    for r in problems:
        out.append({"problem_id": r["unique_id"], "level": int(r["level"]), "subject": r["subject"],
                    "problem": r["problem"], "answer": r["answer"],
                    "prompt_len_on": int(prompt_ids(tok, r["problem"], "on", "cpu").shape[1]),
                    "prompt_len_off": int(prompt_ids(tok, r["problem"], "off", "cpu").shape[1])})

    by_len = sorted(out, key=lambda r: (r["prompt_len_on"], r["problem_id"]))
    b3 = [by_len[0], by_len[len(by_len) // 2], by_len[-1]]
    doc = {
        "source": "HuggingFaceH4/MATH-500 (test split)",
        "selection": f"{a.per_level} per level (1-5), random.Random({a.seed}).sample over problems sorted by unique_id",
        "tokenizer": a.tokenizer,
        "b3_short": [{"problem_id": r["problem_id"], "role": role, "prompt_len_on": r["prompt_len_on"]}
                     for r, role in zip(b3, ["shortest", "middle", "longest"])],
        "problems": out,
    }
    SUBSET.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} problems to {SUBSET}")
    for r in doc["b3_short"]:
        print(f"  B3-short {r['role']:8s}: {r['problem_id']} ({r['prompt_len_on']} prompt tokens)")


if __name__ == "__main__":
    main()
