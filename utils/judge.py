import os
import re

JUDGE_PROMPT = """\
Question: {question}
Correct answer: {correct_answer}
Model response: {response}

Does the model response convey the correct factual answer?
First provide a brief reasoning (one sentence). Then on a new line write only "correct" or "incorrect".\
"""


def parse_batch_results_anthropic(batch_results, source_df) -> "pd.DataFrame":
    """Parse Anthropic batch results into a DataFrame matching the judge CSV format.

    Joins parsed votes back to source_df by row_index to recover question and config.
    Returns DataFrame with cols: question, config, reasoning_1, vote_1, ..., reasoning_n, vote_n.

    custom_id format expected: "{row_index}_{config}_vote_{v}"
    """
    import pandas as pd

    rows = {}  # row_index -> {config, vote_1, reasoning_1, ...}

    for result in batch_results:
        custom_id = result.custom_id
        # parse custom_id: last token is vote number, second-to-last is config, rest is row_index
        # custom_id format: "{row_index}_{config}_vote_{v}"
        # e.g. "42_A_vote_2" -> parts = ["42", "A", "vote", "2"]
        parts = custom_id.split("_")
        row_index = int(parts[0])
        config = parts[1]
        v = parts[3]

        text = result.result.message.content[0].text.strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        vote_line = lines[-1].lower()
        if "incorrect" in vote_line:
            vote = "incorrect"
        elif "correct" in vote_line:
            vote = "correct"
        else:
            vote = "unclear"
        reasoning = " ".join(lines[:-1]) if len(lines) > 1 else ""

        if row_index not in rows:
            rows[row_index] = {"config": config}
        rows[row_index][f"reasoning_{v}"] = reasoning
        rows[row_index][f"vote_{v}"] = vote

    records = []
    for row_index, data in rows.items():
        question = source_df.loc[row_index, "question"]
        records.append({"question": question, **data})

    return pd.DataFrame(records)


def judge_once(
    question: str,
    correct_answer: str,
    response: str,
    provider: str,
    model: str,
) -> tuple[str, str]:
    """Call one judge model once. Returns (reasoning, vote) where vote is 'correct' or 'incorrect'."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        message = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()

    elif provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        result = genai.GenerativeModel(model).generate_content(prompt)
        text = result.text.strip()

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        result = client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = result.choices[0].message.content.strip()

    else:
        raise ValueError(f"Unknown provider: {provider!r}")

    # Extract vote from last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    vote_line = lines[-1].lower()
    if "incorrect" in vote_line:
        vote = "incorrect"
    elif "correct" in vote_line:
        vote = "correct"
    else:
        vote = "unclear"

    # Reasoning is everything except the last line
    reasoning = " ".join(lines[:-1]) if len(lines) > 1 else ""

    return reasoning, vote


def build_batch_requests_anthropic(
    df,
    model: str,
    n_votes: int = 3,
) -> list[dict]:
    """Build a list of Anthropic batch request dicts from a responses DataFrame.

    custom_id format: "{row_index}_{config}_vote_{v}"
    Assumes df has columns: question, correct_answer, config, response.
    """
    requests = []
    for idx, row in df.iterrows():
        prompt = JUDGE_PROMPT.format(
            question=row["question"],
            correct_answer=row["correct_answer"],
            response=row["response"],
        )
        for v in range(1, n_votes + 1):
            requests.append({
                "custom_id": f"{idx}_{row['config']}_vote_{v}",
                "params": {
                    "model": model,
                    "max_tokens": 256,
                    "messages": [{"role": "user", "content": prompt}],
                },
            })
    return requests


def judge_sample(
    question: str,
    correct_answer: str,
    response: str,
    provider: str,
    model: str,
    n_votes: int = 3,
) -> dict:
    """Call one judge model n_votes times in parallel. Returns a flat dict with reasoning_i and vote_i keys."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_votes) as executor:
        futures = [
            executor.submit(judge_once, question, correct_answer, response, provider, model)
            for _ in range(n_votes)
        ]
        results = [f.result() for f in futures]

    return {
        key: val
        for i, (reasoning, vote) in enumerate(results, start=1)
        for key, val in [(f"reasoning_{i}", reasoning), (f"vote_{i}", vote)]
    }
