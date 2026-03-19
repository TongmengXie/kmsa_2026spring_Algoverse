import numpy as np
import torch
from utils.generation import NEUTRAL_SYSTEM, LIE_SYSTEM


def extract_activations(
    question: str,
    response: str,
    label: str,
    model,
    tokenizer,
    device: str,
) -> np.ndarray:
    """
    Run a single forward pass on (prompt + response) and extract the hidden state
    at the last token position for every transformer layer.

    Returns
    -------
    np.ndarray of shape (n_layers, hidden_dim)
        Index 0 = layer 1, ..., index n_layers-1 = last layer.
        (Embedding layer is excluded.)
    """
    system_prompt = LIE_SYSTEM if label == "lie" else NEUTRAL_SYSTEM

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    full_text = prompt_text + response
    input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # outputs.hidden_states: tuple of (n_layers + 1) tensors, shape (1, seq_len, hidden_dim)
    # Index 0 is the embedding layer — skip it, keep layers 1..n_layers
    hidden_states = outputs.hidden_states[1:]  # (n_layers,) each: (1, seq_len, hidden_dim)

    # Take the last token position from each layer → (n_layers, hidden_dim)
    activations = torch.stack([hs[0, -1, :] for hs in hidden_states])
    return activations.cpu().float().numpy()
