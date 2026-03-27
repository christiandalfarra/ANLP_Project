"""
Run prompting experiments and save predictions.

Usage examples:
    # Zero-shot, FLAN-T5, chunk+aggregate
    python scripts/run_prompting.py --model flan --strategy chunk --prompt zero

    # Few-shot, FLAN-T5, chunk+aggregate
    python scripts/run_prompting.py --model flan --strategy chunk --prompt few

    # CoT, FLAN-T5, chunk+aggregate
    python scripts/run_prompting.py --model flan --strategy chunk --prompt cot

    # Zero-shot, LED, long-context
    python scripts/run_prompting.py --model led --strategy long --prompt zero

    # Run all 6 prompting conditions
    python scripts/run_prompting.py --all

    # Use fine-tuned checkpoint instead of pretrained
    python scripts/run_prompting.py --model bart --strategy chunk --prompt zero --checkpoint checkpoints/bart

Output:
    outputs/predictions/{condition_name}.json
    e.g. outputs/predictions/flan_chunk_zero.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset_loader import load_match
from src.data.splits import load_splits
from src.pipelines.chunk_aggregate import run_chunk_aggregate
from src.pipelines.longcontext import run_longcontext

PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "predictions")


def get_flan_chunk_generate_fn(prompter, prompt_type, few_shot_examples=None):
    """Return a (text, start, end) -> summary function for FLAN chunk-based prompting."""
    if prompt_type == "zero":
        from src.prompts.zero_shot import build_chunk_prompt
        def fn(text, start, end):
            return prompter.generate(build_chunk_prompt(text, start, end))
    elif prompt_type == "few":
        from src.prompts.few_shot import build_chunk_prompt as fs_build
        def fn(text, start, end):
            return prompter.generate(fs_build(text, start, end, few_shot_examples))
    elif prompt_type == "cot":
        from src.prompts.chain_of_thought import build_pass1_prompt, build_pass2_prompt
        def fn(text, start, end):
            events = prompter.generate(build_pass1_prompt(text, start, end))
            return prompter.generate(build_pass2_prompt(events, start, end))
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return fn


def get_merge_fn(prompter, prompt_type, few_shot_example=None):
    if prompt_type in ("zero", "few"):
        from src.prompts.zero_shot import build_merge_prompt
        def fn(summaries):
            return prompter.generate(build_merge_prompt(summaries), max_new_tokens=256)
    elif prompt_type == "cot":
        from src.prompts.chain_of_thought import build_merge_prompt
        def fn(summaries):
            return prompter.generate(build_merge_prompt(summaries), max_new_tokens=256)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return fn


def run_single_condition(model_name, strategy, prompt_type, checkpoint=None, splits=None):
    condition = f"{model_name}_{strategy}_{prompt_type}"
    out_path = os.path.join(PREDICTIONS_DIR, f"{condition}.json")
    if os.path.exists(out_path):
        print(f"Skipping {condition} (already exists at {out_path})")
        return

    print(f"\n{'='*60}")
    print(f"Running condition: {condition}")
    print(f"{'='*60}")

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    if splits is None:
        splits = load_splits()
    test_ids = splits["test"]

    # Load model
    if checkpoint:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        import torch
        class _SimplePrompter:
            def __init__(self, model, tokenizer, device):
                self.model = model
                self.tokenizer = tokenizer
                self.device = device
            def generate(self, prompt, max_new_tokens=256):
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
                with torch.no_grad():
                    ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4)
                return self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        prompter = _SimplePrompter(model, tokenizer, device)
    elif model_name == "flan":
        from src.models.prompting.flan_prompter import FlanPrompter
        prompter = FlanPrompter()
        tokenizer = prompter.tokenizer
    elif model_name == "led":
        from src.models.prompting.led_prompter import LEDPrompter
        prompter = LEDPrompter()
        tokenizer = prompter.tokenizer
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Prepare few-shot examples if needed
    few_shot_examples = None
    if prompt_type == "few":
        from src.prompts.few_shot import select_examples
        train_ids = splits["train"]
        from src.data.dataset_loader import load_match as _lm
        summaries_map = {}
        for mid in train_ids:
            m = _lm(mid)
            summaries_map[mid] = m.summary or ""
        example_ids = select_examples(train_ids, summaries_map, n=2)
        few_shot_examples = []
        for eid in example_ids:
            m = _lm(eid)
            chunk_text_ex = m.transcript[:600]
            few_shot_examples.append((chunk_text_ex, m.summary or ""))

    # Run inference
    predictions = {}
    for mid in test_ids:
        print(f"\nMatch: {mid}")
        match = load_match(mid)

        if strategy == "chunk":
            max_tokens = 450 if model_name == "flan" else 900
            gen_fn = get_flan_chunk_generate_fn(prompter, prompt_type, few_shot_examples)
            merge_fn = get_merge_fn(prompter, prompt_type)
            summary = run_chunk_aggregate(match, tokenizer, gen_fn, merge_fn, max_tokens)
        elif strategy == "long":
            if prompt_type == "zero":
                from src.prompts.zero_shot import build_longcontext_prompt
                summary = run_longcontext(match, lambda t: prompter.generate(build_longcontext_prompt(t)))
            elif prompt_type == "cot":
                from src.prompts.chain_of_thought import build_longcontext_pass1_prompt, build_longcontext_pass2_prompt
                def _cot_long(text):
                    events = prompter.generate(build_longcontext_pass1_prompt(text))
                    return prompter.generate(build_longcontext_pass2_prompt(events))
                summary = run_longcontext(match, _cot_long)
            else:
                from src.prompts.zero_shot import build_longcontext_prompt
                summary = run_longcontext(match, lambda t: prompter.generate(build_longcontext_prompt(t)))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        predictions[mid] = summary
        print(f"  Generated: {summary[:120]}...")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\nPredictions saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["flan", "led", "bart"], default="flan")
    parser.add_argument("--strategy", choices=["chunk", "long"], default="chunk")
    parser.add_argument("--prompt", choices=["zero", "few", "cot"], default="zero")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned checkpoint directory")
    parser.add_argument("--all", action="store_true",
                        help="Run all 6 standard prompting conditions")
    args = parser.parse_args()

    splits = load_splits()

    if args.all:
        conditions = [
            ("flan", "chunk", "zero"),
            ("flan", "chunk", "few"),
            ("flan", "chunk", "cot"),
            ("led",  "long",  "zero"),
            ("led",  "long",  "few"),
            ("led",  "long",  "cot"),
        ]
        for model, strategy, prompt in conditions:
            run_single_condition(model, strategy, prompt, splits=splits)
    else:
        run_single_condition(args.model, args.strategy, args.prompt, args.checkpoint, splits)


if __name__ == "__main__":
    main()
