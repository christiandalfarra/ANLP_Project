"""
BART-Large-CNN fine-tuning on football match summarization.

Strategy: fine-tune on (transcript truncated to 1024 tokens, summary) pairs.
Encoder is frozen for the first `freeze_encoder_epochs` epochs to prevent
catastrophic forgetting of BART's pre-trained language knowledge.
"""

import os
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer

from src.models.finetuning.trainer import train_model

MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_TOKENS = 1024
MAX_TARGET_TOKENS = 256


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


def prepare_dataset(
    transcripts: List[str],
    summaries: List[str],
    tokenizer: BartTokenizer,
    max_input: int = MAX_INPUT_TOKENS,
    max_target: int = MAX_TARGET_TOKENS,
) -> MatchSummarizationDataset:
    model_inputs = tokenizer(
        transcripts,
        max_length=max_input,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            summaries,
            max_length=max_target,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
    # Replace padding token id in labels with -100 (ignored in loss)
    label_ids = labels["input_ids"]
    label_ids[label_ids == tokenizer.pad_token_id] = -100

    return MatchSummarizationDataset(model_inputs, label_ids)


def finetune_bart(
    train_transcripts: List[str],
    train_summaries: List[str],
    val_transcripts: List[str],
    val_summaries: List[str],
    output_dir: str,
    freeze_encoder_epochs: int = 3,
    total_epochs: int = 15,
    device: Optional[str] = None,
):
    """
    Fine-tune BART-Large-CNN.
    Phase 1: freeze encoder for `freeze_encoder_epochs`.
    Phase 2: unfreeze encoder and continue training.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {MODEL_NAME}...")
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    train_ds = prepare_dataset(train_transcripts, train_summaries, tokenizer)
    val_ds = prepare_dataset(val_transcripts, val_summaries, tokenizer)

    # --- Phase 1: Frozen encoder ---
    print(f"Phase 1: Training with frozen encoder for {freeze_encoder_epochs} epochs...")
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
        early_stopping_patience=freeze_encoder_epochs + 1,  # no early stopping in phase 1
    )

    # --- Phase 2: Unfrozen encoder ---
    print(f"Phase 2: Training with unfrozen encoder for up to {total_epochs - freeze_encoder_epochs} more epochs...")
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
