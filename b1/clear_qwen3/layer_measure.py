# CLEAR layer_measure.py adapted to Qwen3: one whole decoder layer replayed on its stored input.
# Usage: python layer_measure.py <activations.pt> <out.json>
import json
import sys

import torch
from transformers import AutoModelForCausalLM

from common import model_name, dtype, num_trials, repeat_parts, measure_energy

activations_file, output_file = sys.argv[1], sys.argv[2]
data = torch.load(activations_file)
layer_idx = int(data["layer_idx"])
li = data["layer_in"]
hidden = li["hidden"].cuda()
pos = (li["cos"].cuda(), li["sin"].cuda())
mask = None if li["attention_mask"] is None else li["attention_mask"].cuda()

model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation="sdpa").cuda()
model.eval()
chosen_layer = model.model.layers[layer_idx]

results = {f"layer_{layer_idx}": measure_energy(
    lambda: chosen_layer(hidden, attention_mask=mask, position_embeddings=pos),
    repeat_parts, num_trials, f"Layer {layer_idx}")}

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)
print(f"\nLayer energy results saved to {output_file}")
