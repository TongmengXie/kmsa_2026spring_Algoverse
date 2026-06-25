# Figure Redraw Spec

All 9 figures for the paper are being redrawn (terminology change + higher resolution).
This file is self-contained: it lists every figure, what it plots, and the exact in-figure
label conventions to use. Hand this to whoever writes the plotting code.

> **Context in one line:** the paper reframes the old "deception" class as
> **falsehood compliance** (the model gives a wrong answer to a question it knows,
> because a *falsehood-inducing* system prompt told it to). Every figure must use the
> new terms below.

---

## Global conventions (apply to ALL figures)

**Class names / legend labels** (use these exact strings everywhere):
- `Truth`
- `Honest mistake`
- `Falsehood compliance`  ← replaces the old `Deception` (use `Falsehood comp.` if space is tight)
- `Macro F1` (for the aggregate line)

**Steering-vector names** (used in the flip-rate figure):
- `v_fc`            ← replaces old `v_deception`  (falsehood-compliance direction)
- `v_mistake`        ← unchanged (honest-mistake direction)
- `v_fc_vs_mistake`  ← replaces old `v_dec_vs_mistake`

**Fixed color mapping** (keep identical across every figure):
- Macro F1 = blue
- Truth = orange
- Honest mistake = green
- Falsehood compliance = red

**Format / quality:**
- Export **vector PDF** (preferred for LaTeX) or **>= 300 DPI PNG**.
- Legend + axis fonts must stay legible at two-column width (~3.3 in / 8.4 cm) for the
  two main-text figures.
- No "deception" string may appear in any rendered figure (legends, axis labels, panel
  titles, subtitles).

---

## Main-text figures (2)

### 1. `fig1_qwen25_probe_performance`
- **Role:** Figure 1 (main text), single column.
- **Plots:** Three-class probe performance vs. layer for **Qwen2.5-7B-Instruct**
  (debate-prompt activations), under all four probes.
- **Axes:** x = layer (or relative depth 0–1); y = F1.
- **Lines:** Macro F1 + per-class F1 for Truth, Honest mistake, Falsehood compliance.
- **Expected shape (sanity check):** rise-then-plateau; peak macro F1 ~0.82 around
  relative depth 0.67–0.70; Falsehood-compliance line highest (~0.98), Truth lowest (~0.69).
- **Label changes:** red line legend `deception` -> **`Falsehood compliance`**.

### 2. `fig_flip_rate_vs_alpha`
- **Role:** Figure 2 (main text), single column.
- **Plots:** Correctness **flip rate vs. steering strength alpha**, three CAA vectors,
  layers 14–22, on Qwen2.5-7B-Instruct (debate-prompt vectors).
- **Axes:** x = alpha (0.5, 1, 2, 3, 5); y = flip rate (0–1 or %).
- **Lines/series:** `v_fc`, `v_mistake`, `v_fc_vs_mistake` (one series per vector; layers
  shown as separate lines or shaded band per vector).
- **Expected (sanity check):** `v_fc` reaches 84–94% at alpha=5; `v_mistake` stays <= 30%
  across all settings; `v_fc_vs_mistake` tracks `v_fc`.
- **Label changes:** legend `v_deception` -> **`v_fc`**, `v_dec_vs_mistake` -> **`v_fc_vs_mistake`**
  (`v_mistake` unchanged).

---

## Appendix figures (7)

### 3. `pair_a_perclass_f1`
- **Plots:** Robustness to **falsehood-inducing prompt format** — per-class F1 across layers
  for all four probe architectures (3-Way LR, 3-Way MLP, Cascaded LR, Cascaded MLP) on
  Qwen2.5-7B-Instruct, comparing **concise vs. debate** falsehood-inducing system prompts.
- **Expected:** both prompts converge by mid-depth (concise=0.823 vs. debate=0.819 at layer 19).
- **Label changes:** title/subtitle `deception prompt` -> **`falsehood-inducing prompt`**;
  class legend `deception` -> **`Falsehood compliance`**.

### 4. `fig3_qwen3_probe_performance`
- **Plots:** Qwen3-4B **thinking vs. non-thinking** — per-class F1 across layers, all four probes.
- **Expected:** essentially no thinking-mode effect (non-thinking 0.815 vs. thinking 0.812).
- **Label changes:** class legend `deception` -> **`Falsehood compliance`**.

### 5. `fig4_gemma4_probe_performance`
- **Plots:** Gemma-4-E4B-IT **thinking vs. non-thinking** — per-class F1 across layers, all four probes.
- **Expected:** thinking weaker (non-thinking 0.848 vs. thinking 0.829), largest decline on Truth.
- **Label changes:** class legend `deception` -> **`Falsehood compliance`**.

### 6. `fig5_qwen25_vs_qwen3nt`
- **Plots:** Qwen2.5 vs. Qwen3 non-thinking (**within-family**) — per-class F1 across layers.
- **Label changes:** class legend `deception` -> **`Falsehood compliance`**.

### 7. `fig6_qwen25_vs_gemma4nt`
- **Plots:** Qwen2.5 vs. Gemma-4 non-thinking (**across families**) — per-class F1 across layers.
- **Label changes:** class legend `deception` -> **`Falsehood compliance`**.

### 8. `cascaded_lr_all_configs`
- **Plots:** Cascaded **LR**, all configs. **Six-panel** decomposition:
  - Top row: Macro F1 | Stage 1 Truth F1 | Stage 1 Non-truth F1
  - Bottom row: Stage 2 AUROC | Stage 2 Honest-Mistake F1 | **Stage 2 Falsehood-Compliance F1**
- **Configs (series):** Qwen2.5-7B (concise), Qwen2.5-7B (debate), Qwen3-4B (thinking),
  Gemma4-E4B (no thinking), Gemma4-E4B (thinking).
- **Expected:** Stage 1 Truth F1 is the only meaningful bottleneck; Stage 2 AUROC near 1.0.
- **Label changes:** bottom-right panel title `Stage 2 Deception F1`
  -> **`Stage 2 Falsehood-Compliance F1`**.
  (Note: Stage 2 separates Honest mistake vs. Falsehood compliance.)

### 9. `cascaded_mlp_all_configs`
- **Plots:** Same six-panel layout as #8 but with **MLP** classifiers at both stages.
- **Expected:** small but consistent improvement over LR; same bottleneck (Stage 1 Truth F1).
- **Label changes:** same panel-title change as #8.

---

## Quick checklist before exporting

- [ ] No figure contains the word "deception".
- [ ] Class legend = Truth / Honest mistake / Falsehood compliance (+ Macro F1), same colors everywhere.
- [ ] Steering legend = v_fc / v_mistake / v_fc_vs_mistake.
- [ ] Cascaded figures' Stage-2 third panel reads "Falsehood-Compliance F1".
- [ ] Exported as vector PDF or >=300 DPI; fonts legible at column width.
