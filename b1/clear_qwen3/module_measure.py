# CLEAR module_measure.py adapted to Qwen3: the parts of CLEAR's decoder tables (Table 1, Tables 14-16).
#   Attention Block = self_attn, MLP = mlp, Norm (All) = input_layernorm + post_attention_layernorm (each measured
#   alone, then added), Embedding Layer = embed_tokens (random ids, as CLEAR does for BERT), LM Head, Final Layer Norm.
# Unlike BERT, Qwen3's self_attn and mlp contain no norm, so Norm (All) is not double counted.
# Usage: python module_measure.py <activations.pt> <out.json>
import json
import sys

import torch
from transformers import AutoModelForCausalLM

from common import model_name, dtype, num_trials, repeat_parts, measure_energy

activations_file, output_file = sys.argv[1], sys.argv[2]
data = torch.load(activations_file)
layer_idx = int(data["layer_idx"])
li = data["layer_in"]
layer_in = li["hidden"].cuda()
pos = (li["cos"].cuda(), li["sin"].cuda())
mask = None if li["attention_mask"] is None else li["attention_mask"].cuda()
attn_in, post_ln_in, mlp_in = data["attn_in"].cuda(), data["post_ln_in"].cuda(), data["mlp_in"].cuda()
final_norm_in, lm_head_in = data["final_norm_in"].cuda(), data["lm_head_in"].cuda()

model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation="sdpa").cuda()
model.eval()
chosen_layer = model.model.layers[layer_idx]
ids = torch.randint(0, model.config.vocab_size, layer_in.shape[:2], device="cuda")

parts = {
    "embeddings": lambda: model.model.embed_tokens(ids),
    "attention": lambda: chosen_layer.self_attn(hidden_states=attn_in, attention_mask=mask, position_embeddings=pos),
    "mlp": lambda: chosen_layer.mlp(mlp_in),
    "input_layernorm": lambda: chosen_layer.input_layernorm(layer_in),
    "post_attention_layernorm": lambda: chosen_layer.post_attention_layernorm(post_ln_in),
    "final_norm": lambda: model.model.norm(final_norm_in),
    "lm_head": lambda: model.lm_head(lm_head_in),
}
results = {name: measure_energy(fn, repeat_parts, num_trials, name) for name, fn in parts.items()}

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)
print(f"\nResults saved to {output_file}")
