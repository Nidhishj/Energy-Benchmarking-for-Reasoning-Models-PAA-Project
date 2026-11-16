"""Write configs/aime_subset.json: AIME 2024 + AIME 2025 (60 problems), in the same format as math500_subset.json.

- AIME 2024: HuggingFaceH4/aime_2024 (30 problems, AIME I and II). problem_id "aime2024/I-1" ... from the AoPS URL.
- AIME 2025: MathArena/aime_2025 (30 problems; problem_idx 1-15 = AIME I, 16-30 = AIME II). problem_id "aime2025/I-1" ...
- "level" holds the year (2024/2025): AIME has no MATH-style difficulty level, and the analysis scripts group by it.
- Answers are integers 0-999, graded with math-verify like MATH-500.

Usage: python configs/make_aime_subset.py --tokenizer models/Qwen--Qwen3-8B
"""

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.qwen import prompt_ids  # noqa: E402

OUT = Path(__file__).resolve().parent / "aime_subset.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    rows = []
    for r in load_dataset("HuggingFaceH4/aime_2024", split="train"):
        m = re.search(r"AIME_(I+)_Problems/Problem_(\d+)", r["url"])
        rows.append({"problem_id": f"aime2024/{m.group(1)}-{int(m.group(2))}", "level": 2024, "subject": "AIME",
                     "problem": r["problem"], "answer": str(r["answer"]).strip(), "source": "HuggingFaceH4/aime_2024"})
    for r in load_dataset("MathArena/aime_2025", split="train"):
        i = int(r["problem_idx"])
        part, num = ("I", i) if i <= 15 else ("II", i - 15)
        subj = re.findall(r"'([^']+)'", str(r.get("problem_type", ""))) or ["AIME"]
        rows.append({"problem_id": f"aime2025/{part}-{num}", "level": 2025, "subject": "; ".join(subj),
                     "problem": r["problem"], "answer": str(r["answer"]).strip(), "source": "MathArena/aime_2025"})
    rows.sort(key=lambda r: (r["level"], r["problem_id"].split("/")[1].split("-")[0],
                             int(r["problem_id"].split("-")[-1])))
    for r in rows:  # AIME answers are integers 0-999; "025" -> "25" so the grader compares numbers
        if r["answer"].isdigit():
            r["answer"] = str(int(r["answer"]))
    assert len(rows) == 60 and len({r["problem_id"] for r in rows}) == 60
    for r in rows:
        r["prompt_len_on"] = int(prompt_ids(tok, r["problem"], "on", "cpu").shape[1])
        r["prompt_len_off"] = int(prompt_ids(tok, r["problem"], "off", "cpu").shape[1])
    doc = {"source": "AIME 2024 (HuggingFaceH4/aime_2024) + AIME 2025 (MathArena/aime_2025)",
           "selection": "all 60 problems; level = year", "tokenizer": a.tokenizer, "problems": rows}
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} problems to {OUT}")


if __name__ == "__main__":
    main()
