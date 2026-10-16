# read_activations.py
import torch

# Define the file where activations are saved
input_file = "activations.pt"

# Load the activations from the file
# We use map_location='cpu' to load the tensor to the CPU,
# in case the file was saved from a CUDA device.
try:
    activations = torch.load(input_file, map_location='cpu')
    print("Successfully loaded activations.")
except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
    exit()

# The activations for both attn_in and mlp_in are stored as a list of tensors.
# Since we only ran one forward pass, the list will contain just one tensor.
# We can inspect the first (and only) tensor in either list.
if activations["mlp_in"]:
    # Get the activation tensor for the MLP input
    mlp_input_tensor = activations["mlp_in"][0]

    # The shape of the tensor is (batch_size, sequence_length, hidden_dim)
    # The number of tokens is the sequence_length.
    tensor_shape = mlp_input_tensor.shape
    num_tokens = tensor_shape[1]

    print(f"\nCaptured activations for layer: {activations['layer_idx']}")
    print(f"Shape of the MLP input activation tensor: {tensor_shape}")
    print(f"Number of tokens in the input sequence: {num_tokens}")
else:
    print("The 'mlp_in' list in the activations file is empty. Cannot determine token count.")