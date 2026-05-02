"""
BART-Large-CNN fine-tuning on football match summarization.

Strategy: rather than training BART on (truncated transcript, full summary) —
which only exposes BART to the first ~5 minutes of a 2.5h match — we train
BART as the *merge step* of the chunk+aggregate inference pipeline.

For each training match we pre-generate per-chunk summaries with pretrained
BART (zero-shot), concatenate them, and train BART to map that concatenated
intermediate summary to the gold reference summary. The intermediate summary
spans the entire match in compressed form, so the input the model trains on
matches the input it sees at merge-time during inference.

Two-phase training: encoder frozen for the first 3 epochs, then unfrozen.
Pre-generated per-chunk summaries are cached on disk so subsequent runs skip
the (~10–30 min one-time) generation step.
"""

import json
import os
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer

from src.data.chunker import chunk_text
from src.data.dataset_loader import load_match
from src.models.finetuning.trainer import train_model

MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_TOKENS = 1024
MAX_TARGET_TOKENS = 256
CHUNK_TOKENS = 900           # max tokens per chunk fed to BART during pre-summarization
CHUNK_SUMMARY_TOKENS = 32    # 30 chunks * 32 = ~960 tokens, fits in BART's 1024 context with no truncation
CACHE_DIR = os.path.join("checkpoints", "_chunk_summary_cache")


class MatchSummarizationDataset(Dataset):
    def __init__(
        self,
        encodings: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def _generate_chunk_summaries_for_match(
    match_id: str,
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer,
    device: str,
) -> str:
    """Run pretrained BART on each chunk of `match_id`, return the concatenated chunk-summaries."""
    cache_path = os.path.join(CACHE_DIR, f"{match_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)["intermediate_summary"]

    try:
        match = load_match(match_id)
        chunks = chunk_text(match.segments, tokenizer, max_tokens=CHUNK_TOKENS)
        partial_summaries = []
        model.eval()
        with torch.no_grad():
            for text, _, _ in chunks:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=CHUNK_TOKENS,
                ).to(device)
                ids = model.generate(
                    **inputs,
                    max_new_tokens=CHUNK_SUMMARY_TOKENS,
                    num_beams=2,
                    no_repeat_ngram_size=3,
                )
                partial_summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True).strip())

        intermediate = " ".join(partial_summaries)
    except Exception as e:
        # Don't kill the whole run for one bad match — fall back to truncated raw transcript
        print(f"  [WARN] chunk-summary generation failed for {match_id}: {e}. Falling back to raw transcript.", flush=True)
        m = load_match(match_id)
        intermediate = m.transcript

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"intermediate_summary": intermediate}, f, ensure_ascii=False)
    return intermediate


def build_intermediate_summaries(
    match_ids: List[str],
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer,
    device: str,
) -> List[str]:
    """For each match id, return BART's concatenated per-chunk summaries (cached)."""
    out = []
    for i, mid in enumerate(match_ids, 1):
        print(f"  [{i}/{len(match_ids)}] {mid}", flush=True)
        out.append(_generate_chunk_summaries_for_match(mid, model, tokenizer, device))
    return out


def prepare_dataset(
    intermediate_summaries: List[str],
    references: List[str],
    tokenizer: BartTokenizer,
    max_input: int = MAX_INPUT_TOKENS,
    max_target: int = MAX_TARGET_TOKENS,
) -> MatchSummarizationDataset:
    model_inputs = tokenizer(
        intermediate_summaries,
        max_length=max_input,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    labels = tokenizer(
        text_target=references,
        max_length=max_target,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    label_ids = labels["input_ids"]
    label_ids[label_ids == tokenizer.pad_token_id] = -100

    return MatchSummarizationDataset(model_inputs, label_ids)


def finetune_bart(
    train_match_ids: List[str],
    train_summaries: List[str],
    val_match_ids: List[str],
    val_summaries: List[str],
    output_dir: str,
    freeze_encoder_epochs: int = 3,
    total_epochs: int = 15,
    device: Optional[str] = None,
):
    """
    Fine-tune BART-Large-CNN as the merge step of chunk+aggregate.

    Phase 1: freeze encoder for `freeze_encoder_epochs`.
    Phase 2: unfreeze encoder and continue training.

    Note: takes match_ids + summaries (not raw transcripts) because the
    intermediate summaries are generated from segments via the chunker.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {MODEL_NAME}...", flush=True)
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    print(f"Pre-generating chunk summaries for {len(train_match_ids)} train matches "
          f"(cached to {CACHE_DIR}, one-time cost)...", flush=True)
    train_inter = build_intermediate_summaries(train_match_ids, model, tokenizer, device)
    print(f"Pre-generating chunk summaries for {len(val_match_ids)} val matches...", flush=True)
    val_inter = build_intermediate_summaries(val_match_ids, model, tokenizer, device)

    print("Tokenizing datasets...", flush=True)
    train_ds = prepare_dataset(train_inter, train_summaries, tokenizer)
    val_ds = prepare_dataset(val_inter, val_summaries, tokenizer)

    # --- Phase 1: Frozen encoder ---
    print(f"Phase 1: Training with frozen encoder for {freeze_encoder_epochs} epochs...", flush=True)
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    phase1_dir = os.path.join(output_dir, "phase1")
    train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=phase1_dir,
        training_args_kwargs={
            "num_train_epochs": freeze_encoder_epochs,
            "load_best_model_at_end": False,
            "fp16": device == "cuda",
        },
        early_stopping_patience=freeze_encoder_epochs + 1,
    )

    # Phase 1 checkpoint is no longer needed — phase 2 continues from the
    # in-memory model. Free the disk before phase 2 starts saving.
    import shutil
    if os.path.isdir(phase1_dir):
        shutil.rmtree(phase1_dir, ignore_errors=True)
        print(f"Removed {phase1_dir} to free disk for phase 2.", flush=True)

    # --- Phase 2: Unfrozen encoder ---
    print(f"Phase 2: Training with unfrozen encoder for up to {total_epochs - freeze_encoder_epochs} more epochs...", flush=True)
    for param in model.model.encoder.parameters():
        param.requires_grad = True

    train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=output_dir,
        training_args_kwargs={
            "num_train_epochs": total_epochs - freeze_encoder_epochs,
            "fp16": device == "cuda",
        },
        early_stopping_patience=3,
    )

    return model, tokenizer
