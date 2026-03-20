import numpy as np
import torch
import torch.nn as nn

_loss_fn = nn.CrossEntropyLoss(reduction="none")


def score_choice(
    question: str,
    choice: str,
    model,
    tokenizer,
    device: str,
) -> float:
    """
    Average NLL of `choice` tokens given `question` as context.
    Lower = model considers this choice more likely.
    """
    messages = [
        {"role": "system", "content": "Answer the following question concisely and factually."},
        {"role": "user",   "content": question},
    ]
    context_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    full_text = context_text + choice

    context_ids = tokenizer(context_text, return_tensors="pt").input_ids.to(device)
    full_ids    = tokenizer(full_text,    return_tensors="pt").input_ids.to(device)

    choice_len = full_ids.shape[-1] - context_ids.shape[-1]
    if choice_len <= 0:
        return float("inf")

    with torch.no_grad():
        logits = model(full_ids).logits[0]      # (seq_len, vocab)

    shift_logits = logits[:-1]                  # (seq_len-1, vocab)
    shift_labels = full_ids[0, 1:]              # (seq_len-1,)

    token_nll  = _loss_fn(shift_logits, shift_labels)
    choice_nll = token_nll[-choice_len:]
    return choice_nll.mean().item()


def knowledge_check_truthfulqa(
    item: dict,
    model,
    tokenizer,
    device: str,
) -> dict:
    """
    Knowledge check for TruthfulQA mc1_targets format.
    item must have keys: "question", "mc1_targets" (with "choices" and "labels").

    Returns:
        question, all_choices, all_scores, model_choice, correct_answer, passed
    """
    question = item["question"]
    choices  = item["mc1_targets"]["choices"]
    labels   = item["mc1_targets"]["labels"]
    correct  = choices[labels.index(1)]

    scores   = [score_choice(question, c, model, tokenizer, device) for c in choices]
    best_idx = int(np.argmin(scores))

    return {
        "question":       question,
        "all_choices":    choices,
        "all_scores":     scores,
        "model_choice":   choices[best_idx],
        "correct_answer": correct,
        "passed":         choices[best_idx] == correct,
    }


def knowledge_check_mmlu(
    item: dict,
    model,
    tokenizer,
    device: str,
) -> dict:
    """
    Knowledge check for MMLU / ARC format.
    item must have keys: "question", "choices" (list of strings), "answer" (int index).

    Returns:
        question, all_choices, all_scores, model_choice, correct_answer, passed
    """
    question = item["question"]
    choices  = item["choices"]
    correct  = choices[item["answer"]]

    scores   = [score_choice(question, c, model, tokenizer, device) for c in choices]
    best_idx = int(np.argmin(scores))

    return {
        "question":       question,
        "all_choices":    choices,
        "all_scores":     scores,
        "model_choice":   choices[best_idx],
        "correct_answer": correct,
        "passed":         choices[best_idx] == correct,
    }
