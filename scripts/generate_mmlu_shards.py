#!/usr/bin/env python3
"""
Run MMLU factual response generation for one GPU shard.

Usage — one command per terminal (tmux recommended):
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_mmlu_shards.py --gpu_id 0
    CUDA_VISIBLE_DEVICES=1 python scripts/generate_mmlu_shards.py --gpu_id 1
    CUDA_VISIBLE_DEVICES=2 python scripts/generate_mmlu_shards.py --gpu_id 2

After all three finish, run merge_factual_generation_shards() in the notebook.
CUDA_VISIBLE_DEVICES remaps the physical GPU to cuda:0 inside each process.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.settings import (
    DO_SAMPLE,
    FACTUAL_DECEPTION_SCENARIO,
    MMLU_KC_PATH,
    MMLU_RESPONSES_PATH,
    MODEL_ID,
    NEUTRAL_SYSTEM,
)
from utils.generation import run_factual_generation_shard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id",       type=int, required=True, help="Shard index (0, 1, or 2)")
    parser.add_argument("--total_shards", type=int, default=3,     help="Total number of shards")
    args = parser.parse_args()

    device = "cuda:0"  # CUDA_VISIBLE_DEVICES maps the physical GPU to index 0

    test_df = pd.read_csv(MMLU_KC_PATH)
    passed_df = test_df[test_df["passed"] == True].reset_index(drop=True)
    failed_df = test_df[test_df["passed"] == False].reset_index(drop=True)
    print(f"Loaded knowledge test: {len(passed_df)} passed, {len(failed_df)} failed")

    print(f"Loading {MODEL_ID} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    run_factual_generation_shard(
        passed_df, failed_df, model, tokenizer, device,
        NEUTRAL_SYSTEM, FACTUAL_DECEPTION_SCENARIO,
        MMLU_RESPONSES_PATH,
        shard_idx=args.gpu_id,
        total_shards=args.total_shards,
        checkpoint_every=50,
        do_sample=DO_SAMPLE,
    )


if __name__ == "__main__":
    main()
