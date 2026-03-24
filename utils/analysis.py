import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA


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
