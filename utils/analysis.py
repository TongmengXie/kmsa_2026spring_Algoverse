import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

from utils.probe import train_linear_probe


def reduce_activations_pca(
    activations: np.ndarray,
    n_components: int,
) -> dict:
    """
    Apply PCA independently per layer to reduce hidden dimension.

    Parameters
    ----------
    activations  : (n_samples, n_layers, hidden_dim)
    n_components : target number of dimensions after PCA

    Returns
    -------
    dict with keys:
        activations   : np.ndarray (n_samples, n_layers, n_components), float32
        components    : np.ndarray (n_layers, n_components, hidden_dim) — eigenvectors per layer
        explained_var : np.ndarray (n_layers,) — total explained variance ratio per layer
    """
    n_samples, n_layers, hidden_dim = activations.shape
    reduced    = np.zeros((n_samples, n_layers, n_components), dtype=np.float32)
    components = np.zeros((n_layers, n_components, hidden_dim), dtype=np.float32)
    explained_var = np.zeros(n_layers)

    for layer_idx in range(n_layers):
        pca = PCA(n_components=n_components, random_state=42)
        reduced[:, layer_idx, :]    = pca.fit_transform(activations[:, layer_idx, :]).astype(np.float32)
        components[layer_idx]       = pca.components_.astype(np.float32)
        explained_var[layer_idx]    = pca.explained_variance_ratio_.sum()

    return {
        "activations":   reduced,
        "components":    components,
        "explained_var": explained_var,
    }


def select_pca_k(
    activations: np.ndarray,
    labels: np.ndarray,
    k_values: list[int],
    output_path: str | Path = "outputs/pca_reduction_k_selection_results.csv",
) -> pd.DataFrame:
    """
    Select optimal PCA components k by scanning k_values across 3 representative layers
    (25%, 50%, 75% of total depth). For each (layer, k) combination, fits PCA once at
    max(k_values) and slices, then trains a linear probe to record 4 metrics.

    Parameters
    ----------
    activations : (n_samples, n_layers, hidden_dim)
    labels      : (n_samples,) — string labels
    k_values    : list of ints, e.g. [16, 32, 64, 128, 256, 512]
    output_path : path to save results CSV

    Returns
    -------
    pd.DataFrame with columns:
        layer_idx, layer_pct, k, variance_explained,
        f1_macro_val, f1_macro_train, f1_gap, train_time_sec
    """
    n_samples, n_layers, hidden_dim = activations.shape
    max_k = max(k_values)

    fracs = [0.25, 0.50, 0.75]
    layer_indices = [max(0, min(n_layers - 1, round(n_layers * f) - 1)) for f in fracs]
    layer_pct_labels = ["25%", "50%", "75%"]

    print(f"n_layers={n_layers}, representative layers: "
          + ", ".join(f"{pct}→layer {idx}" for pct, idx in zip(layer_pct_labels, layer_indices)))
    print(f"k values to scan: {k_values}")
    print(f"Fitting PCA with max_k={max_k} per layer, then slicing for each k.\n")

    rows = []

    for layer_idx, layer_pct in zip(layer_indices, layer_pct_labels):
        layer_acts = activations[:, layer_idx, :]

        # Fit PCA once at max_k for this layer
        pca = PCA(n_components=max_k, random_state=42)
        acts_max = pca.fit_transform(layer_acts)           # (n_samples, max_k)
        cumvar = np.cumsum(pca.explained_variance_ratio_)  # cumulative variance per component

        print(f"Layer {layer_idx} ({layer_pct}):")

        for k in k_values:
            acts_k = acts_max[:, :k]                       # slice first k components
            var_explained = float(cumvar[k - 1])

            t0 = time.time()
            result = train_linear_probe(acts_k, labels)
            train_time = time.time() - t0

            f1_val   = result["f1_macro_val"]
            f1_train = result["f1_macro_train"]
            f1_gap   = f1_train - f1_val

            print(f"  k={k:4d} | var={var_explained:.3f} | "
                  f"val_F1={f1_val:.3f} | train_F1={f1_train:.3f} | "
                  f"gap={f1_gap:.3f} | time={train_time:.1f}s")

            rows.append({
                "layer_idx":        layer_idx,
                "layer_pct":        layer_pct,
                "k":                k,
                "variance_explained": var_explained,
                "f1_macro_val":     f1_val,
                "f1_macro_train":   f1_train,
                "f1_gap":           f1_gap,
                "train_time_sec":   round(train_time, 2),
            })

        print()

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return df


def save_results_csv(results: list[dict], path: str | Path) -> pd.DataFrame:
    """
    Convert a list of per-layer probe result dicts to a DataFrame and save as CSV.

    Confusion matrix entries are flattened into columns named:
        cm_{true_class}_{pred_class}       (counts)
        cm_norm_{true_class}_{pred_class}  (row-normalized)

    All other scalar/dict fields are included as columns.

    Parameters
    ----------
    results : list of dicts from probe_all_layers or probe_all_layers_cascaded
    path    : output CSV path

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for r in results:
        row = {"layer": r["layer"], "f1_macro": r["f1_macro"]}

        # Per-class F1
        for cls, val in r["f1_per_class"].items():
            row[f"f1_{cls}"] = val

        # Confusion matrix (counts)
        classes = r["classes"]
        cm = r["confusion_matrix"]
        for i, true_cls in enumerate(classes):
            for j, pred_cls in enumerate(classes):
                row[f"cm_{true_cls}_{pred_cls}"] = int(cm[i, j])

        # Confusion matrix (normalized)
        cm_norm = r["confusion_matrix_norm"]
        for i, true_cls in enumerate(classes):
            for j, pred_cls in enumerate(classes):
                row[f"cm_norm_{true_cls}_{pred_cls}"] = float(cm_norm[i, j])

        # Cascaded-only fields
        if "stage1_f1" in r:
            for k, v in r["stage1_f1"].items():
                row[f"stage1_f1_{k}"] = v
        if "stage2_f1" in r:
            for k, v in r["stage2_f1"].items():
                row[f"stage2_f1_{k}"] = v
        if "stage2_auroc" in r:
            row["stage2_auroc"] = r["stage2_auroc"]

        rows.append(row)

    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
