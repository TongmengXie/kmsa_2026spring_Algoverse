"""
Train binary probes on full-dimensional activations (3584-dim).
Saves clf.coef_ for potential use as steering vectors.

Binary comparisons:
  - truth vs deception
  - truth vs honest_mistake
  - honest_mistake vs deception

Prerequisites:
  - outputs/activations.npy (11708, 28, 3584)
  - outputs/labels.npy (11708,)

Usage:
  python train_binary_probes.py --activations_dir outputs
"""

import argparse
import numpy as np
import pickle
import time
import csv
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

LABEL_MAP = {0: "truth", 1: "honest_mistake", 2: "deception"}
LAYER_RANGE = range(10, 27)


def train_binary_probe(
    activations, labels, class_a, class_b,
    n_splits=5, max_iter=200, random_state=42,
):
    """Train a binary logistic regression probe and return coef_ + metrics."""
    mask = np.isin(labels, [class_a, class_b])
    X = activations[mask]
    y = labels[mask]

    print(f"    {class_a}: {(y == class_a).sum()}, "
          f"{class_b}: {(y == class_b).sum()}, total: {len(y)}")

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    fold_f1s = {class_a: [], class_b: []}

    # Train on ALL data for the final coef_ (steering vector)
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)
    clf_full = LogisticRegression(
        solver="saga", max_iter=max_iter, random_state=random_state, C=1.0
    )
    clf_full.fit(X_scaled_full, y)

    # Cross-validation for reporting F1
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        clf = LogisticRegression(
            solver="saga", max_iter=max_iter, random_state=random_state, C=1.0
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        f1s = f1_score(y_val, y_pred, labels=[class_a, class_b], average=None)
        fold_f1s[class_a].append(f1s[0])
        fold_f1s[class_b].append(f1s[1])

    return {
        "class_a": class_a,
        "class_b": class_b,
        "coef_": clf_full.coef_[0],
        "intercept_": clf_full.intercept_[0],
        "scaler_mean_": scaler_full.mean_,
        "scaler_scale_": scaler_full.scale_,
        "f1_a": float(np.mean(fold_f1s[class_a])),
        "f1_b": float(np.mean(fold_f1s[class_b])),
        "f1_macro": float(np.mean([
            np.mean(fold_f1s[class_a]),
            np.mean(fold_f1s[class_b]),
        ])),
        "classes_": list(clf_full.classes_),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="steering_vectors")
    args = parser.parse_args()

    act_dir = Path(args.activations_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    print("Loading activations...")
    activations = np.load(act_dir / "activations.npy")
    labels_int = np.load(act_dir / "labels.npy")
    labels_str = np.array([LABEL_MAP[i] for i in labels_int])

    print(f"Activations shape: {activations.shape}")
    print(f"Labels: { {k: int((labels_str == k).sum()) for k in LABEL_MAP.values()} }")

    comparisons = [
        ("truth", "deception"),
        ("truth", "honest_mistake"),
        ("honest_mistake", "deception"),
    ]

    all_results = {}
    for class_a, class_b in comparisons:
        comp_name = f"{class_a}_vs_{class_b}"
        print(f"\n{'='*60}")
        print(f"Binary probe: {class_a} vs {class_b}")
        print("=" * 60)

        layer_results = []
        for layer_idx in LAYER_RANGE:
            t0 = time.time()
            print(f"  Layer {layer_idx}...", end=" ")
            layer_acts = activations[:, layer_idx, :]
            result = train_binary_probe(layer_acts, labels_str, class_a, class_b)
            result["layer"] = layer_idx
            elapsed = time.time() - t0
            print(f"F1 macro={result['f1_macro']:.3f} "
                  f"({class_a}={result['f1_a']:.3f}, "
                  f"{class_b}={result['f1_b']:.3f}) [{elapsed:.1f}s]")
            layer_results.append(result)

        all_results[comp_name] = layer_results
        with open(out_dir / f"binary_probe_{comp_name}.pkl", "wb") as f:
            pickle.dump(layer_results, f)
        print(f"Saved: {out_dir / f'binary_probe_{comp_name}.pkl'}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    for comp_name, results in all_results.items():
        print(f"\n{comp_name}:")
        print(f"  {'Layer':>5}  {'F1 macro':>8}  {'F1 A':>6}  {'F1 B':>6}")
        for r in results:
            print(f"  {r['layer']:>5}  {r['f1_macro']:>8.3f}  "
                  f"{r['f1_a']:>6.3f}  {r['f1_b']:>6.3f}")

    csv_path = out_dir / "binary_probe_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "comparison", "layer", "f1_macro", "f1_class_a", "f1_class_b"
        ])
        for comp_name, results in all_results.items():
            for r in results:
                writer.writerow([
                    comp_name, r["layer"], r["f1_macro"], r["f1_a"], r["f1_b"]
                ])
    print(f"\nSaved summary: {csv_path}")


if __name__ == "__main__":
    main()
