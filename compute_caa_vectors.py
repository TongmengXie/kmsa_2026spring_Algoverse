"""
Compute CAA (Contrastive Activation Addition) steering vectors.

For each layer, computes mean activation difference between classes:
  - v_deception:      mean(deception) - mean(truth)
  - v_mistake:        mean(honest_mistake) - mean(truth)
  - v_dec_vs_mistake: mean(deception) - mean(honest_mistake)

Prerequisites:
  - outputs/activations.npy (11708, 28, 3584)
  - outputs/labels.npy (11708,)

Usage:
  python compute_caa_vectors.py
"""

import numpy as np
import pickle
from pathlib import Path

LABEL_MAP = {0: "truth", 1: "honest_mistake", 2: "deception"}

print("Loading activations...")
activations = np.load("outputs/activations.npy")
labels = np.load("outputs/labels.npy")
labels_str = np.array([LABEL_MAP[i] for i in labels])
print(f"Shape: {activations.shape}")
print(f"Labels: { {k: int((labels_str == k).sum()) for k in LABEL_MAP.values()} }")

out_dir = Path("steering_vectors")
out_dir.mkdir(exist_ok=True)

print(f"\n{'Layer':>5}  {'v_deception':>12}  {'v_mistake':>10}  {'v_dec_vs_mis':>13}")
for layer_idx in range(10, 27):
    layer_acts = activations[:, layer_idx, :]

    mean_truth = layer_acts[labels_str == "truth"].mean(axis=0)
    mean_deception = layer_acts[labels_str == "deception"].mean(axis=0)
    mean_mistake = layer_acts[labels_str == "honest_mistake"].mean(axis=0)

    v_deception = mean_deception - mean_truth
    v_mistake = mean_mistake - mean_truth
    v_dec_vs_mistake = mean_deception - mean_mistake

    print(f"{layer_idx:>5}  "
          f"{np.linalg.norm(v_deception):>12.2f}  "
          f"{np.linalg.norm(v_mistake):>10.2f}  "
          f"{np.linalg.norm(v_dec_vs_mistake):>13.2f}")

    result = {
        "layer": layer_idx,
        "v_deception": v_deception,
        "v_mistake": v_mistake,
        "v_dec_vs_mistake": v_dec_vs_mistake,
        "mean_truth": mean_truth,
        "mean_deception": mean_deception,
        "mean_mistake": mean_mistake,
    }

    with open(out_dir / f"caa_vectors_layer{layer_idx}.pkl", "wb") as f:
        pickle.dump(result, f)

print(f"\nSaved CAA vectors for layers 10-26 to {out_dir}/")
print("Done!")
