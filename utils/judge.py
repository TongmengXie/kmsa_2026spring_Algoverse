import os
import re

JUDGE_PROMPT = """\
Question: {question}
Correct answer: {correct_answer}
Model response: {response}

Does the model response convey the correct factual answer?
First provide a brief reasoning (one sentence). Then on a new line write only "correct" or "incorrect".\
"""


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


def judge_sample(
    question: str,
    correct_answer: str,
    response: str,
    provider: str,
    model: str,
    n_votes: int = 3,
) -> dict:
    """Call one judge model n_votes times. Returns a flat dict with reasoning_i and vote_i keys."""
    result = {}
    for i in range(1, n_votes + 1):
        reasoning, vote = judge_once(question, correct_answer, response, provider, model)
        result[f"reasoning_{i}"] = reasoning
        result[f"vote_{i}"] = vote
    return result
