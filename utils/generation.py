import pandas as pd
import torch
from pathlib import Path
from tqdm.auto import tqdm


def parse_thinking_response(raw_text: str) -> tuple[str, str]:
    """Split Gemma 4 thinking output into (thinking_block, final_answer).

    Gemma 4 format: <|channel>thought\\n{reasoning}<|channel|>{final answer}

    thinking_block includes both surrounding tags so it can be concatenated
    directly into the activation extraction forward pass without reconstruction.
    Returns ("", raw_text) if no thinking tags are found.
    """
    close_tag = "<|channel|>"
    if close_tag in raw_text:
        idx = raw_text.index(close_tag)
        thinking_block = raw_text[: idx + len(close_tag)]
        final_answer = raw_text[idx + len(close_tag) :].strip()
    else:
        thinking_block = ""
        final_answer = raw_text.strip()
    return thinking_block, final_answer


def generate_response(
    question: str,
    model,
    tokenizer,
    device: str,
    system_prompt: str = "",
    max_new_tokens: int = 100,
    do_sample: bool = False,
) -> tuple[str, str]:
    """Generate a response to `question` using the given system prompt.

    Returns
    -------
    thinking_block : str
        Raw thinking block including Gemma 4 tags
        (<|channel>thought\\n....<|channel|>).  Empty string if the model
        produced no thinking output (e.g. thinking mode not triggered).
    final_answer : str
        The model's final answer with thinking block stripped.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    # Do NOT skip special tokens — needed to detect <|channel|> boundary
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    return parse_thinking_response(raw)


def run_factual_generation(
    passed_df,
    failed_df,
    model,
    tokenizer,
    device: str,
    neutral_system: str,
    factual_deception_scenario: str,
    output_path: Path,
    checkpoint_every: int = 50,
    max_new_tokens: int = 100,
    do_sample: bool = False,
):
    """
    Generate factual responses for configs A, B, C with checkpoint/resume support.

    Config A: passed_df + neutral_system              → truth
    Config B: failed_df + neutral_system              → honest_mistake
    Config C: passed_df + factual_deception_scenario  → deception

    Skips entirely if output_path already contains all expected rows.
    Resumes from partial checkpoint if output_path exists but is incomplete.

    Returns
    -------
    resp_df : pd.DataFrame — full results with columns:
              question, correct_answer, config, system_prompt, response
    """
    output_path = Path(output_path)
    configs = [
        ("A", passed_df, neutral_system),
        ("B", failed_df, neutral_system),
        ("C", passed_df, factual_deception_scenario),
    ]
    total_expected = len(passed_df) * 2 + len(failed_df)

    if output_path.exists():
        resp_df = pd.read_csv(output_path)
        if len(resp_df) == total_expected:
            print(f"[skip] Already complete ({total_expected} rows): {output_path.name}")
            return resp_df
        done_keys = set(zip(resp_df["question"], resp_df["config"]))
        print(f"Resuming: {len(resp_df)} done, {total_expected - len(resp_df)} remaining")
    else:
        done_keys = set()
        print(f"Starting fresh: {total_expected} rows across 3 configs")

    for config_name, source_df, system_prompt in configs:
        remaining = source_df[~source_df["question"].isin(
            {q for q, c in done_keys if c == config_name}
        )].reset_index(drop=True)

        if len(remaining) == 0:
            print(f"Config {config_name}: already complete, skipping.")
            continue

        print(f"Config {config_name}: {len(remaining)} rows to generate.")
        records = []
        for i, row in enumerate(tqdm(remaining.itertuples(), total=len(remaining), desc=f"Config {config_name}")):
            thinking, response = generate_response(
                row.question, model, tokenizer, device,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
            records.append({
                "question":      row.question,
                "correct_answer": row.correct_answer,
                "config":        config_name,
                "system_prompt": system_prompt,
                "thinking":      thinking,
                "response":      response,
            })
            if (i + 1) % checkpoint_every == 0:
                pd.DataFrame(records).to_csv(
                    output_path, mode="a", header=not output_path.exists(), index=False
                )
                done_keys.update((r["question"], r["config"]) for r in records)
                records = []
        if records:
            pd.DataFrame(records).to_csv(
                output_path, mode="a", header=not output_path.exists(), index=False
            )

    resp_df = pd.read_csv(output_path)
    print(f"Done. Total rows: {len(resp_df)}")
    print(resp_df["config"].value_counts().to_string())
    return resp_df


def run_scenario_generation(
    deception_df,
    model,
    tokenizer,
    device: str,
    output_path: Path,
    raw_path: Path,
    checkpoint_every: int = 50,
    max_new_tokens: int = 100,
    do_sample: bool = False,
):
    """
    Generate honest + deceptive responses for all social scenario pairs.

    Uses the `prompt` column of deception_df as system prompt and `question`
    as the user turn.  Results are checkpointed to raw_path (long format) and
    then pivoted to wide format (200 rows) before saving to output_path.

    Skips entirely if output_path already exists.
    Resumes from raw_path if output_path does not exist but raw_path does.

    Returns
    -------
    scenario_resp_df : pd.DataFrame — 200 rows, wide format, columns:
                       pair_id, question, honest_scenario, honest_response,
                       deceptive_scenario, deceptive_response
    """
    output_path = Path(output_path)
    raw_path    = Path(raw_path)
    total = len(deception_df)

    if output_path.exists():
        scenario_resp_df = pd.read_csv(output_path)
        print(f"[skip] Already complete ({len(scenario_resp_df)} pairs): {output_path.name}")
        return scenario_resp_df

    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        done_keys = set(zip(raw_df["pair_id"], raw_df["label"]))
        print(f"Resuming: {len(raw_df)} done, {total - len(raw_df)} remaining")
    else:
        done_keys = set()
        print(f"Starting fresh: {total} rows")

    remaining = deception_df[~deception_df.apply(
        lambda r: (r["pair_id"], r["label"]) in done_keys, axis=1
    )].reset_index(drop=True)

    if len(remaining) > 0:
        print(f"{len(remaining)} rows to generate.")
        records = []
        for i, row in enumerate(tqdm(remaining.itertuples(), total=len(remaining), desc="Scenario generation")):
            thinking, response = generate_response(
                row.question, model, tokenizer, device,
                system_prompt=row.prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
            records.append({
                "pair_id":       row.pair_id,
                "label":         row.label,
                "question":      row.question,
                "system_prompt": row.prompt,
                "thinking":      thinking,
                "response":      response,
            })
            if (i + 1) % checkpoint_every == 0:
                pd.DataFrame(records).to_csv(
                    raw_path, mode="a", header=not raw_path.exists(), index=False
                )
                done_keys.update((r["pair_id"], r["label"]) for r in records)
                records = []
        if records:
            pd.DataFrame(records).to_csv(
                raw_path, mode="a", header=not raw_path.exists(), index=False
            )
        print("Generation complete.")

    # Pivot to wide format
    raw = pd.read_csv(raw_path)

    honest = raw[raw["label"] == "honest"].rename(columns={
        "system_prompt": "honest_scenario",
        "thinking":      "honest_thinking",
        "response":      "honest_response",
    })[["pair_id", "question", "honest_scenario", "honest_thinking", "honest_response"]]

    deceptive = raw[raw["label"] == "deceptive"].rename(columns={
        "system_prompt": "deceptive_scenario",
        "thinking":      "deceptive_thinking",
        "response":      "deceptive_response",
    })[["pair_id", "deceptive_scenario", "deceptive_thinking", "deceptive_response"]]

    scenario_resp_df = (
        honest.merge(deceptive, on="pair_id")
        .sort_values("pair_id")
        .reset_index(drop=True)
    )
    scenario_resp_df.to_csv(output_path, index=False)
    print(f"Saved {len(scenario_resp_df)} pairs to {output_path.name}")
    return scenario_resp_df
