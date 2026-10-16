# save_activations.py
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os

# Settings
model_name = "bert-base-uncased"   # simpler alias also works
cache_dir = "./.hfCache"
output_file = "activations2.pt"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16).cuda()
model.eval()
print(model)
exit(0)
# Pick a random encoder layer index
num_layers = len(model.bert.encoder.layer)
layer_to_capture = 5   # for example
print(f"Capturing activations for layer {layer_to_capture}")

# Dict to hold activations
activations = {
    "layer_idx": layer_to_capture,
    "embeddings": [],
    "attn_in": [],
    "mlp_in": [],
    "mlm_head_in": [],
}

embeddings_hook_counter = 0
attn_hook_counter = 0
mlp_hook_counter = 0
mlm_head_hook_counter = 0

# ---------- Hooks ----------
def embeddings_hook(module, input, output):
    global embeddings_hook_counter
    embeddings_hook_counter += 1
    if embeddings_hook_counter == 1:
        activations["embeddings"].append(output.detach().cpu()) 

def attn_hook(module, input, output):
    global attn_hook_counter
    attn_hook_counter += 1
    if attn_hook_counter == 1:
        # input[0] = hidden states into attention
        activations["attn_in"].append(input[0].detach().cpu())

def mlp_hook(module, input, output):
    global mlp_hook_counter
    mlp_hook_counter += 1
    if mlp_hook_counter == 1:
        # input[0] = hidden states into the feedforward (intermediate)
        activations["mlp_in"].append(input[0].detach().cpu())

def mlm_head_hook(module, input, output):
    global mlm_head_hook_counter
    mlm_head_hook_counter += 1
    if mlm_head_hook_counter == 1:
        # input[0] = hidden states before MLM classification
        activations["mlm_head_in"].append(input[0].detach().cpu())

# Register hooks
model.bert.embeddings.register_forward_hook(embeddings_hook)

chosen_layer = model.bert.encoder.layer[layer_to_capture]
chosen_layer.attention.self.register_forward_hook(attn_hook)       # attention input
chosen_layer.intermediate.dense.register_forward_hook(mlp_hook)    # MLP input
model.cls.predictions.transform.dense.register_forward_hook(mlm_head_hook)

# Input prompt
prompt = "The capital of France is"*10 
seq_len = 8
tokens = tokenizer.tokenize(prompt)
stud = tokens[:seq_len]
prompt = tokenizer.convert_tokens_to_string(stud)
inputs = tokenizer(prompt, return_tensors="pt", max_length=seq_len, padding="max_length", truncation=True).to(model.device)

print(inputs)

# Run forward
with torch.no_grad():
    _ = model(**inputs)

# Save activations
torch.save(activations, output_file)
print(f"Saved activations to {output_file}")
