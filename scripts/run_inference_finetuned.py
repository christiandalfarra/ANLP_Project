"""
Run inference with fine-tuned BART or LED checkpoints and save predictions.

This script generates summaries using fine-tuned models (not prompting).
For BART: uses the chunk+aggregate pipeline.
For LED:  uses the long-context pipeline.

Usage:
    python scripts/run_inference_finetuned.py --model bart --checkpoint checkpoints/bart
    python scripts/run_inference_finetuned.py --model led  --checkpoint checkpoints/led
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.data.dataset_loader import load_match
from src.data.splits import load_splits
from src.pipelines.chunk_aggregate import run_chunk_aggregate
from src.pipelines.longcontext import run_longcontext

PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "predictions")


def make_generate_fn(model, tokenizer, device, max_input=1024, max_new_tokens=256):
    def fn(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input,
        ).to(device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4)
        return tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    return fn


def make_led_generate_fn(model, tokenizer, device, max_input=16384, max_new_tokens=256, num_beams=2):
    """LED inference. num_beams=2 keeps VRAM in budget for 16k input on a single T4."""
    def fn(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input,
        ).to(device)
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[:, 0] = 1
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                global_attention_mask=global_attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
            )
        return tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bart", "led"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    condition = f"finetuned_{args.model}"
    out_path = os.path.join(PREDICTIONS_DIR, f"{condition}.json")
    if os.path.exists(out_path):
        print(f"Predictions already exist at {out_path}. Delete to re-run.")
        return

    splits = load_splits()
    test_ids = splits["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint from {args.checkpoint} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    predictions = {}

    for mid in test_ids:
        print(f"\nMatch: {mid}")
        match = load_match(mid)

        if args.model == "bart":
            # Chunk + aggregate
            from src.prompts.zero_shot import build_merge_prompt

            class _Prompter:
                def __init__(self, fn): self._fn = fn
                def generate(self, prompt, max_new_tokens=256): return self._fn(prompt)

            gen_base = make_generate_fn(model, tokenizer, device, max_input=1024)
            prompter = _Prompter(gen_base)

            def gen_fn(text, start, end):
                from src.prompts.zero_shot import build_chunk_prompt
                return gen_base(build_chunk_prompt(text, start, end))

            def merge_fn(summaries):
                return gen_base(build_merge_prompt(summaries))

            summary = run_chunk_aggregate(match, tokenizer, gen_fn, merge_fn, max_tokens_per_chunk=900)

        elif args.model == "led":
            gen_fn = make_led_generate_fn(model, tokenizer, device)
            summary = run_longcontext(match, gen_fn)

        predictions[mid] = summary
        print(f"  Generated: {summary[:120]}...")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\nPredictions saved to {out_path}")


if __name__ == "__main__":
    main()
