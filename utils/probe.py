import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report


def train_linear_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    max_iter: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Train a logistic regression probe on activations from a single layer
    using stratified k-fold cross-validation.

    Parameters
    ----------
    activations : (n_samples, hidden_dim)
    labels      : (n_samples,) — string labels e.g. "truth"/"lie"/"mistake"

    Returns
    -------
    dict with keys:
        f1_per_class : dict {class_name: mean_f1_across_folds}
        f1_macro     : float
        report       : str  (classification report from last fold)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classes = sorted(set(labels))

    fold_f1s = {c: [] for c in classes}

    for train_idx, val_idx in skf.split(activations, labels):
        X_train, X_val = activations[train_idx], activations[val_idx]
        y_train, y_val = labels[train_idx],      labels[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        clf = LogisticRegression(max_iter=max_iter, random_state=random_state, C=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        f1s = f1_score(y_val, y_pred, labels=classes, average=None)
        for cls, score in zip(classes, f1s):
            fold_f1s[cls].append(score)

    f1_per_class = {cls: float(np.mean(scores)) for cls, scores in fold_f1s.items()}
    f1_macro = float(np.mean(list(f1_per_class.values())))

    return {
        "f1_per_class": f1_per_class,
        "f1_macro":     f1_macro,
    }


def probe_all_layers(
    activations: np.ndarray,
    labels: np.ndarray,
    **kwargs,
) -> list[dict]:
    """
    Run train_linear_probe for every layer.

    Parameters
    ----------
    activations : (n_samples, n_layers, hidden_dim)

    Returns
    -------
    list of dicts, one per layer (length = n_layers)
    """
    n_layers = activations.shape[1]
    results = []
    for layer_idx in range(n_layers):
        layer_acts = activations[:, layer_idx, :]   # (n_samples, hidden_dim)
        result = train_linear_probe(layer_acts, labels, **kwargs)
        result["layer"] = layer_idx
        results.append(result)
    return results
