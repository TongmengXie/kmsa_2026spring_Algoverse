import json

cells = []

# Cell 1: imports
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# -- imports & paths\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from pathlib import Path\n"
        "\n"
        'SUMMARY_DIR = Path("summary")\n'
        'FIG_DIR = SUMMARY_DIR / "figures"\n'
        "FIG_DIR.mkdir(exist_ok=True)"
    )
})

# Cell 2: coalesce
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# -- coalesce all CSVs\n"
        "# filename format: {model_family}_{classification}_{classifier}_{prompt}.csv\n"
        "records = []\n"
        'for fpath in sorted(SUMMARY_DIR.glob("*.csv")):\n'
        '    parts = fpath.stem.split("_")\n'
        "    model_family   = parts[0]\n"
        "    classification = parts[1]\n"
        "    classifier     = parts[2]\n"
        '    prompt         = "debate" if parts[3] == "dabate" else parts[3]\n'
        "\n"
        "    df = pd.read_csv(fpath)\n"
        '    df["rel_layer"]      = df["layer"] / (len(df) - 1)\n'
        '    df["model_family"]   = model_family\n'
        '    df["classification"] = classification\n'
        '    df["classifier"]     = classifier\n'
        '    df["prompt"]         = prompt\n'
        '    if "stage2_auroc" not in df.columns:\n'
        '        df["stage2_auroc"] = np.nan\n'
        "    records.append(df)\n"
        "\n"
        "master = pd.concat(records, ignore_index=True)\n"
        "\n"
        "configs = (\n"
        '    master[["model_family", "classification", "classifier", "prompt"]]\n'
        "    .drop_duplicates()\n"
        '    .sort_values(["model_family", "classification", "classifier", "prompt"])\n'
        "    .reset_index(drop=True)\n"
        ")\n"
        'print(f"{len(master)} total rows  {len(configs)} configs")\n'
        "configs"
    )
})

# Cell 3: helpers
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# -- plotting helpers\n"
        'COMBOS  = [("3way", "lr"), ("3way", "mlp"), ("cascaded", "lr"), ("cascaded", "mlp")]\n'
        'CLABELS = ["3-way LR",    "3-way MLP",    "Cascaded LR",      "Cascaded MLP"]\n'
        'C1, C2  = "steelblue", "tomato"\n'
        "\n"
        'CLASS_COLS   = ["f1_deception", "f1_honest_mistake", "f1_truth"]\n'
        'CLASS_LABELS = ["Deception",    "Honest Mistake",    "Truth"]\n'
        "\n"
        "\n"
        "def _draw_lines(ax, df, cond_col, val1, val2, label1, label2, ycol):\n"
        '    d1 = df[df[cond_col] == val1].sort_values("rel_layer")\n'
        '    d2 = df[df[cond_col] == val2].sort_values("rel_layer")\n'
        "    l1, = ax.plot(d1[\"rel_layer\"], d1[ycol], color=C1, lw=2,        label=label1)\n"
        "    l2, = ax.plot(d2[\"rel_layer\"], d2[ycol], color=C2, lw=2, ls=\"--\", label=label2)\n"
        "    ax.set_ylim(0, 1)\n"
        "    ax.grid(True, alpha=0.3)\n"
        "    return l1, l2\n"
        "\n"
        "\n"
        "def plot_macro_f1(df, cond_col, val1, val2, label1, label2, title, save_path):\n"
        '    """2x2 grid: one subplot per classifier combo, 2 lines per subplot."""\n'
        "    fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n"
        "    for ax, (cls, clf), clabel in zip(axes.flatten(), COMBOS, CLABELS):\n"
        '        sub = df[(df["classification"] == cls) & (df["classifier"] == clf)]\n'
        '        _draw_lines(ax, sub, cond_col, val1, val2, label1, label2, "f1_macro")\n'
        "        ax.set_title(clabel, fontweight=\"bold\")\n"
        '        ax.set_xlabel("Relative layer depth")\n'
        '        ax.set_ylabel("Macro F1")\n'
        "        ax.legend(fontsize=9)\n"
        "    fig.suptitle(title, fontsize=13, fontweight=\"bold\")\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(save_path, dpi=150, bbox_inches=\"tight\")\n"
        "    plt.show()\n"
        '    print(f"Saved to {save_path}")\n'
        "\n"
        "\n"
        "def plot_perclass_f1(df, cond_col, val1, val2, label1, label2, title, save_path):\n"
        '    """3x4 grid: rows=class (deception/honest_mistake/truth), cols=classifier combo."""\n'
        "    fig, axes = plt.subplots(3, 4, figsize=(20, 11), sharey=True)\n"
        "    for row, (ccol, clname) in enumerate(zip(CLASS_COLS, CLASS_LABELS)):\n"
        "        for col, ((cls, clf), clabel) in enumerate(zip(COMBOS, CLABELS)):\n"
        "            ax = axes[row, col]\n"
        '            sub = df[(df["classification"] == cls) & (df["classifier"] == clf)]\n'
        "            _draw_lines(ax, sub, cond_col, val1, val2, label1, label2, ccol)\n"
        "            if row == 0:\n"
        "                ax.set_title(clabel, fontweight=\"bold\")\n"
        "            if col == 0:\n"
        '                ax.set_ylabel(f"{clname} F1", fontsize=10)\n'
        "            if row == 2:\n"
        '                ax.set_xlabel("Relative layer depth")\n'
        "            if row == 0 and col == 0:\n"
        "                ax.legend(fontsize=9)\n"
        "    fig.suptitle(title, fontsize=13, fontweight=\"bold\")\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(save_path, dpi=150, bbox_inches=\"tight\")\n"
        "    plt.show()\n"
        '    print(f"Saved to {save_path}")'
    )
})

# Cell 4: Set A
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# -- Set A: prompt effect (qwen2.5 concise vs debate)\n"
        'df_A = master[master["model_family"] == "qwen2.5"]\n'
        "\n"
        "plot_macro_f1(\n"
        '    df_A, "prompt", "concise", "debate", "Concise", "Debate",\n'
        '    "Set A - Prompt Effect (Qwen2.5-7B): Macro F1",\n'
        '    FIG_DIR / "setA_macro_f1.png",\n'
        ")\n"
        "plot_perclass_f1(\n"
        '    df_A, "prompt", "concise", "debate", "Concise", "Debate",\n'
        '    "Set A - Prompt Effect (Qwen2.5-7B): Per-class F1",\n'
        '    FIG_DIR / "setA_perclass_f1.png",\n'
        ")"
    )
})

# Cell 5: Set B
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# -- Set B: thinking mode (qwen2.5 concise vs qwen3 concise)\n"
        "# qwen2.5 = no thinking; qwen3 = thinking on\n"
        "# x-axis is relative layer depth so the different layer counts (28 vs 36) align\n"
        'df_concise = master[master["prompt"] == "concise"]\n'
        "\n"
        "plot_macro_f1(\n"
        '    df_concise, "model_family", "qwen2.5", "qwen3",\n'
        '    "Qwen2.5 (no thinking)", "Qwen3 (thinking on)",\n'
        '    "Set B - Thinking Mode Effect: Macro F1",\n'
        '    FIG_DIR / "setB_macro_f1.png",\n'
        ")\n"
        "plot_perclass_f1(\n"
        '    df_concise, "model_family", "qwen2.5", "qwen3",\n'
        '    "Qwen2.5 (no thinking)", "Qwen3 (thinking on)",\n'
        '    "Set B - Thinking Mode Effect: Per-class F1",\n'
        '    FIG_DIR / "setB_perclass_f1.png",\n'
        ")"
    )
})

# Cell 6: Set C
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# -- Set C: model family (qwen2.5 concise vs gemma4 concise)\n"
        "# both without thinking mode; x-axis normalized so 28 vs 42 layers align\n"
        "\n"
        "plot_macro_f1(\n"
        '    df_concise, "model_family", "qwen2.5", "gemma4",\n'
        '    "Qwen2.5-7B", "Gemma4-E4B",\n'
        '    "Set C - Model Family Effect: Macro F1",\n'
        '    FIG_DIR / "setC_macro_f1.png",\n'
        ")\n"
        "plot_perclass_f1(\n"
        '    df_concise, "model_family", "qwen2.5", "gemma4",\n'
        '    "Qwen2.5-7B", "Gemma4-E4B",\n'
        '    "Set C - Model Family Effect: Per-class F1",\n'
        '    FIG_DIR / "setC_perclass_f1.png",\n'
        ")"
    )
})

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(r"d:\code\kmsa_2026spring_Algoverse\summary.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("done")
