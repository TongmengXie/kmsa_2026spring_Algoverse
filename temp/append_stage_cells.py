import json

with open(r"d:\code\kmsa_2026spring_Algoverse\summary.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

new_cells = [
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": (
            "# -- cascaded MLP stage breakdown helper\n"
            "# Layout (2x3):\n"
            "#   Row 0 (Stage 1): f1_truth | f1_non_truth | f1_macro (reference)\n"
            "#   Row 1 (Stage 2): f1_deception | f1_honest_mistake | stage2_auroc\n"
            "\n"
            "STAGE_LAYOUT = [\n"
            '    ("stage1_f1_truth",         "Stage 1 - F1 Truth"),\n'
            '    ("stage1_f1_non_truth",      "Stage 1 - F1 Non-truth"),\n'
            '    ("f1_macro",                 "Overall Macro F1 (reference)"),\n'
            '    ("stage2_f1_deception",      "Stage 2 - F1 Deception"),\n'
            '    ("stage2_f1_honest_mistake", "Stage 2 - F1 Honest Mistake"),\n'
            '    ("stage2_auroc",             "Stage 2 - AUROC"),\n'
            "]\n"
            "ROW_LABELS = [\"Stage 1\", \"Stage 2\"]\n"
            "\n"
            "\n"
            "def plot_cascaded_stages(df, cond_col, val1, val2, label1, label2, title, save_path):\n"
            '    """2x3 grid: cascaded MLP stage 1 and stage 2 breakdown."""\n'
            '    sub = df[(df["classification"] == "cascaded") & (df["classifier"] == "mlp")]\n'
            "    fig, axes = plt.subplots(2, 3, figsize=(15, 8))\n"
            "    for i, (col_name, subplot_title) in enumerate(STAGE_LAYOUT):\n"
            "        row, col = divmod(i, 3)\n"
            "        ax = axes[row, col]\n"
            "        _draw_lines(ax, sub, cond_col, val1, val2, label1, label2, col_name)\n"
            "        ax.set_title(subplot_title, fontweight=\"bold\")\n"
            '        ax.set_xlabel("Relative layer depth")\n'
            "        if col == 0:\n"
            "            ax.set_ylabel(ROW_LABELS[row], fontsize=11, fontweight=\"bold\")\n"
            "        if row == 0 and col == 0:\n"
            "            ax.legend(fontsize=9)\n"
            "    fig.suptitle(title, fontsize=13, fontweight=\"bold\")\n"
            "    plt.tight_layout()\n"
            "    plt.savefig(save_path, dpi=150, bbox_inches=\"tight\")\n"
            "    plt.show()\n"
            '    print(f"Saved to {save_path}")'
        )
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": (
            "# -- Set A: cascaded MLP stage breakdown (qwen2.5 concise vs debate)\n"
            "plot_cascaded_stages(\n"
            '    master[master["model_family"] == "qwen2.5"],\n'
            '    "prompt", "concise", "debate", "Concise", "Debate",\n'
            '    "Set A - Prompt Effect: Cascaded MLP Stage Breakdown",\n'
            '    FIG_DIR / "setA_cascaded_stages.png",\n'
            ")"
        )
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": (
            "# -- Set B: cascaded MLP stage breakdown (qwen2.5 vs qwen3 concise)\n"
            "_concise = master[master[\"prompt\"] == \"concise\"]\n"
            "\n"
            "plot_cascaded_stages(\n"
            '    _concise, "model_family", "qwen2.5", "qwen3",\n'
            '    "Qwen2.5 (no thinking)", "Qwen3 (thinking on)",\n'
            '    "Set B - Thinking Mode: Cascaded MLP Stage Breakdown",\n'
            '    FIG_DIR / "setB_cascaded_stages.png",\n'
            ")"
        )
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": (
            "# -- Set C: cascaded MLP stage breakdown (qwen2.5 vs gemma4 concise)\n"
            "plot_cascaded_stages(\n"
            '    _concise, "model_family", "qwen2.5", "gemma4",\n'
            '    "Qwen2.5-7B", "Gemma4-E4B",\n'
            '    "Set C - Model Family: Cascaded MLP Stage Breakdown",\n'
            '    FIG_DIR / "setC_cascaded_stages.png",\n'
            ")"
        )
    },
]

nb["cells"].extend(new_cells)

with open(r"d:\code\kmsa_2026spring_Algoverse\summary.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("done")
