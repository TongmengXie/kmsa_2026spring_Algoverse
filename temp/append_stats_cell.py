import json

with open(r"d:\code\kmsa_2026spring_Algoverse\summary.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

cell_src = (
    "# -- LR vs MLP comparison tables with Wilcoxon significance\n"
    "# n = number of layers per model (qwen2.5=28, qwen3=36, gemma4=42)\n"
    "# Wilcoxon signed-rank test on paired (MLP_f1 - LR_f1) per layer\n"
    "# Note: adjacent layers are correlated, so p-values are anti-conservative\n"
    "from scipy import stats\n"
    "\n"
    "THINKING = {'qwen3': 'on', 'qwen2.5': 'off', 'gemma4': 'off'}\n"
    'master["thinking"] = master["model_family"].map(THINKING)\n'
    "\n"
    "\n"
    "def make_comparison_table(metric):\n"
    "    rows = []\n"
    "    for (model, cls_type, prompt), grp in master.groupby(\n"
    "            ['model_family', 'classification', 'prompt'], sort=True):\n"
    '        lr  = grp[grp["classifier"] == "lr" ][metric].values\n'
    '        mlp = grp[grp["classifier"] == "mlp"][metric].values\n'
    "        if len(lr) == 0 or len(mlp) == 0:\n"
    "            continue\n"
    "        n = len(lr)\n"
    "        diff = mlp - lr\n"
    "        try:\n"
    "            _, p = stats.wilcoxon(diff)\n"
    "        except ValueError:\n"
    "            p = 1.0\n"
    "        rows.append({\n"
    '            "model":          model,\n'
    '            "classification": cls_type,\n'
    '            "prompt":         prompt,\n'
    '            "thinking":       THINKING[model],\n'
    '            "lr_min":         round(lr.min(),  4),\n'
    '            "lr_mean":        round(lr.mean(), 4),\n'
    '            "lr_max":         round(lr.max(),  4),\n'
    '            "mlp_min":        round(mlp.min(),  4),\n'
    '            "mlp_mean":       round(mlp.mean(), 4),\n'
    '            "mlp_max":        round(mlp.max(),  4),\n'
    '            "mlp-lr":         round(diff.mean(), 4),\n'
    '            "n":              n,\n'
    '            "p_wilcoxon":     round(p, 4),\n'
    "        })\n"
    "    return pd.DataFrame(rows)\n"
    "\n"
    "\n"
    "METRICS = [\n"
    '    ("f1_macro",          "macro_f1"),\n'
    '    ("f1_deception",      "perclass_deception"),\n'
    '    ("f1_honest_mistake", "perclass_honest_mistake"),\n'
    '    ("f1_truth",          "perclass_truth"),\n'
    "]\n"
    "\n"
    "tables = {}\n"
    "for metric, name in METRICS:\n"
    "    t = make_comparison_table(metric)\n"
    "    tables[name] = t\n"
    "    out = FIG_DIR.parent / f'lr_vs_mlp_{name}.csv'\n"
    "    t.to_csv(out, index=False)\n"
    '    print(f"Saved: {out}")\n'
    "    display(t)\n"
    "    print()"
)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cell_src
}

nb["cells"].append(new_cell)

with open(r"d:\code\kmsa_2026spring_Algoverse\summary.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("done")
