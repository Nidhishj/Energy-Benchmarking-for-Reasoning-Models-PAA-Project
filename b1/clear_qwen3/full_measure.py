# CLEAR full_measure.py adapted to Qwen3: the whole forward pass, model(**inputs), as CLEAR calls it (default
# use_cache, so each call also builds a KV cache, as in CLEAR's code). Usage: python full_measure.py <seq_len> <out.json>
import json
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM

from common import model_name, dtype, num_trials, repeat_model, measure_energy, tokenize

seq_len, output_file = int(sys.argv[1]), sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation="sdpa").cuda()
model.eval()
inputs = tokenize(tokenizer, seq_len)

results = {"full_model": measure_energy(lambda: model(**inputs), repeat_model, num_trials, "FullModel")}

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)
print(f"\nFull-model energy results saved to {output_file}")
