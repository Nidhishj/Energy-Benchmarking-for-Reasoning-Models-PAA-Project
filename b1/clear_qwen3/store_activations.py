# CLEAR store_activations.py adapted to Qwen3 (decoder). Usage: python store_activations.py <seq_len> <out.pt>
# Like CLEAR's BERT version, hooks capture the inputs of the parts in one real forward pass. A decoder layer also
# needs the rotary position embeddings (cos, sin) and the attention mask it was called with, so those are kept too.
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from common import model_name, dtype, layer_to_capture, tokenize

seq_len, output_file = int(sys.argv[1]), sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation="sdpa").cuda()
model.eval()

activations = {"layer_idx": layer_to_capture, "seq_len": seq_len}
layer = model.model.layers[layer_to_capture]


def keep(name, take):
    def hook(module, args, kwargs):
        if name not in activations:
            activations[name] = take(args, kwargs)
    return hook


def hidden(args, kwargs):
    return (args[0] if args else kwargs["hidden_states"]).detach().cpu()


def layer_inputs(args, kwargs):
    mask = kwargs.get("attention_mask")
    cos, sin = kwargs["position_embeddings"]
    return {"hidden": hidden(args, kwargs), "cos": cos.detach().cpu(), "sin": sin.detach().cpu(),
            "attention_mask": None if mask is None else mask.detach().cpu()}


layer.register_forward_pre_hook(keep("layer_in", layer_inputs), with_kwargs=True)
layer.self_attn.register_forward_pre_hook(keep("attn_in", hidden), with_kwargs=True)
layer.post_attention_layernorm.register_forward_pre_hook(keep("post_ln_in", hidden), with_kwargs=True)
layer.mlp.register_forward_pre_hook(keep("mlp_in", hidden), with_kwargs=True)
model.model.norm.register_forward_pre_hook(keep("final_norm_in", hidden), with_kwargs=True)
model.lm_head.register_forward_pre_hook(keep("lm_head_in", hidden), with_kwargs=True)

inputs = tokenize(tokenizer, seq_len)
with torch.no_grad():
    out = model(**inputs)

assert torch.isfinite(out.logits).all(), "non-finite logits in FP16"
print("input:", repr(tokenizer.decode(inputs["input_ids"][0])))
print("attention mask passed to the layer:", activations["layer_in"]["attention_mask"] is not None)
torch.save(activations, output_file)
print(f"Saved activations to {output_file}")
