"""
Fine-tune BART or LED on the football match summarization dataset.

Usage:
    python scripts/run_finetuning.py --model bart --output_dir checkpoints/bart
    python scripts/run_finetuning.py --model led  --output_dir checkpoints/led
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset_loader import load_match
from src.data.splits import load_splits


def load_transcripts_and_summaries(match_ids, truncate_tokens=None, tokenizer=None):
    transcripts, summaries = [], []
    for mid in match_ids:
        m = load_match(mid)
        transcript = m.transcript
        if truncate_tokens and tokenizer:
            ids = tokenizer.encode(transcript, add_special_tokens=False)
            transcript = tokenizer.decode(ids[:truncate_tokens], skip_special_tokens=True)
        transcripts.append(transcript)
        summaries.append(m.summary or "")
    return transcripts, summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bart", "led"], required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    splits = load_splits()
    train_ids = splits["train"]
    val_ids = splits["val"]

    if args.output_dir is None:
        args.output_dir = os.path.join("checkpoints", args.model)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model == "bart":
        from transformers import BartTokenizer
        from src.models.finetuning.bart_finetuner import finetune_bart, MODEL_NAME, MAX_INPUT_TOKENS

        tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
        print(f"Loading {len(train_ids)} train / {len(val_ids)} val matches...")
        train_t, train_s = load_transcripts_and_summaries(
            train_ids, truncate_tokens=MAX_INPUT_TOKENS, tokenizer=tokenizer
        )
        val_t, val_s = load_transcripts_and_summaries(
            val_ids, truncate_tokens=MAX_INPUT_TOKENS, tokenizer=tokenizer
        )
        finetune_bart(train_t, train_s, val_t, val_s, output_dir=args.output_dir)

    elif args.model == "led":
        from transformers import AutoTokenizer
        from src.models.finetuning.led_finetuner import finetune_led, MODEL_NAME, MAX_INPUT_TOKENS

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Loading {len(train_ids)} train / {len(val_ids)} val matches...")
        train_t, train_s = load_transcripts_and_summaries(
            train_ids, truncate_tokens=MAX_INPUT_TOKENS, tokenizer=tokenizer
        )
        val_t, val_s = load_transcripts_and_summaries(
            val_ids, truncate_tokens=MAX_INPUT_TOKENS, tokenizer=tokenizer
        )
        finetune_led(train_t, train_s, val_t, val_s, output_dir=args.output_dir)

    print(f"\nDone. Checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
