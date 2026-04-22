"""
Steering Experiment — Algoverse Spring 2026

Injects CAA (Contrastive Activation Addition) steering vectors into 
Qwen2.5-7B-Instruct residual stream and measures behavioral effects.

Tests three vector types:
  - v_deception:      mean(deception) - mean(truth)
  - v_mistake:        mean(honest_mistake) - mean(truth)
  - v_dec_vs_mistake: mean(deception) - mean(honest_mistake)

Sweeps layers 14-22, alphas [0.5, 1.0, 2.0, 3.0, 5.0], all-tokens steering.

Prerequisites:
  - CAA vectors computed via compute_caa_vectors.py
  - probe_dataset.csv in data/dataset/

Usage:
  python steering_experiment.py
"""

import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
NEUTRAL_SYSTEM = "Answer the following question concisely and factually."
N_TEST = 50
SEED = 42
LAYERS = [14, 16, 17, 18, 19, 20, 21, 22]
ALPHAS = [0.5, 1.0, 2.0, 3.0, 5.0]
VECTOR_TYPES = ["v_deception", "v_mistake", "v_dec_vs_mistake"]
OUTPUT_DIR = Path("steering_results")
VECTOR_DIR = Path("steering_vectors")
DATA_DIR = Path("data/dataset")
# Use torch.float16 for V100 compatibility; use torch.bfloat16 for A100/H100
TORCH_DTYPE = torch.float16
# ─────────────────────────────────────────────────────────────────────────


def get_test_prompts(data_dir, n_test, seed):
    """Get held-out factual truth prompts for testing."""
    probe_df = pd.read_csv(data_dir / "probe_dataset.csv")
    truth_df = probe_df[
        (probe_df["label"] == "truth") & (probe_df["domain"] == "factual")
    ].copy()
    truth_df = truth_df.sample(n=min(n_test, len(truth_df)), random_state=seed)
    return truth_df.reset_index(drop=True)


def gen(question, model, tokenizer, device, vec=None, hook_layer=None,
        alpha=0.0, max_new_tokens=150):
    """Generate response, optionally with CAA steering."""
    messages = [
        {"role": "system", "content": NEUTRAL_SYSTEM},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    handle = None
    if vec is not None and hook_layer is not None and alpha != 0:
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0].clone()
                v = vec.to(hs.device)
                hs = hs + alpha * v
                return (hs,) + output[1:]
            else:
                hs = output[0].clone()
                v = vec.to(hs.device)
                hs = hs + alpha * v
                output[0] = hs
                return output
        handle = model.model.layers[hook_layer].register_forward_hook(hook_fn)

    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    if handle:
        handle.remove()

    return tokenizer.decode(
        out[0][input_ids.shape[-1]:], skip_special_tokens=True
    ).strip()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map="auto"
    )
    model.eval()
    print(f"Model loaded: {model.config.num_hidden_layers} layers")

    # ── Load CAA vectors ──────────────────────────────────────────────────
    print("Loading CAA vectors...")
    caa_by_layer = {}
    for layer in LAYERS:
        path = VECTOR_DIR / f"caa_vectors_layer{layer}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                caa_by_layer[layer] = pickle.load(f)
    print(f"Loaded vectors for {len(caa_by_layer)} layers")

    # ── Load test prompts ─────────────────────────────────────────────────
    print(f"Loading {N_TEST} test prompts...")
    test_df = get_test_prompts(DATA_DIR, N_TEST, SEED)
    print(f"Got {len(test_df)} test prompts")

    # ── Generate baselines ────────────────────────────────────────────────
    print("Generating baselines...")
    baselines = []
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Baselines"):
        resp = gen(row["question"], model, tokenizer, DEVICE)
        baselines.append({
            "idx": i,
            "question": row["question"],
            "correct_answer": row.get("correct_answer", ""),
            "baseline_response": resp,
            "baseline_length": len(resp.split()),
        })
    baselines_df = pd.DataFrame(baselines)
    baselines_df.to_csv(OUTPUT_DIR / "baselines.csv", index=False)
    print(f"Baselines saved. Mean length: {baselines_df['baseline_length'].mean():.1f} words")

    # ── Run steering experiments ──────────────────────────────────────────
    results = []
    total_combos = len(caa_by_layer) * len(VECTOR_TYPES) * len(ALPHAS)
    print(f"Running {total_combos} combos x {N_TEST} prompts "
          f"= {total_combos * N_TEST} generations")

    combo_count = 0
    for layer in LAYERS:
        if layer not in caa_by_layer:
            continue
        vecs = caa_by_layer[layer]

        for vec_name in VECTOR_TYPES:
            vec_np = vecs[vec_name]
            vec_t = torch.tensor(vec_np, dtype=TORCH_DTYPE)

            for alpha in ALPHAS:
                combo_count += 1
                print(f"[{combo_count}/{total_combos}] "
                      f"Layer {layer}, {vec_name}, alpha={alpha}")

                for i, row in tqdm(
                    test_df.iterrows(), total=len(test_df),
                    desc=f"L{layer} {vec_name[:5]} a={alpha}", leave=False,
                ):
                    steered = gen(
                        row["question"], model, tokenizer, DEVICE,
                        vec=vec_t, hook_layer=layer, alpha=alpha,
                    )
                    baseline = baselines[i]["baseline_response"]
                    results.append({
                        "question": row["question"],
                        "correct_answer": row.get("correct_answer", ""),
                        "baseline_response": baseline,
                        "steered_response": steered,
                        "layer": layer,
                        "vector_type": vec_name,
                        "alpha": alpha,
                        "changed": baseline != steered,
                        "baseline_words": len(baseline.split()),
                        "steered_words": len(steered.split()),
                        "word_count_diff": len(steered.split()) - len(baseline.split()),
                    })

                # Checkpoint after each combo
                pd.DataFrame(results).to_csv(
                    OUTPUT_DIR / "steering_full_checkpoint.csv", index=False
                )

    # ── Save final results ────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "steering_full_results.csv", index=False)
    print(f"Saved {len(results_df)} results")

    # ── Compute summary ──────────────────────────────────────────────────
    summary = results_df.groupby(["vector_type", "layer", "alpha"]).agg(
        pct_changed=("changed", "mean"),
        mean_word_diff=("word_count_diff", "mean"),
        n=("changed", "count"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "steering_summary.csv", index=False)

    for vec_name in VECTOR_TYPES:
        print(f"\n--- {vec_name} (pct changed) ---")
        sub = summary[summary["vector_type"] == vec_name]
        pivot = sub.pivot(index="layer", columns="alpha", values="pct_changed")
        print(pivot.to_string())

    for vec_name in VECTOR_TYPES:
        print(f"\n--- {vec_name} (word diff) ---")
        sub = summary[summary["vector_type"] == vec_name]
        pivot = sub.pivot(index="layer", columns="alpha", values="mean_word_diff")
        print(pivot.to_string())

    print("\nDone!")


if __name__ == "__main__":
    main()
