"""
LED-Base-16384 fine-tuning on football match summarization.

Strategy: fine-tune on (full transcript capped at 8192 tokens, summary) pairs.
Key requirements:
  - global_attention_mask must be set for the BOS token
  - gradient_checkpointing=True is essential to fit on T4 (15GB VRAM)
  - fp16=True halves memory usage
"""

from typing import List, Optional, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.models.finetuning.trainer import train_model

MODEL_NAME = "allenai/led-base-16384"
MAX_INPUT_TOKENS = 8192   # cap at 8192 for training speed/memory; model supports 16384
MAX_TARGET_TOKENS = 256


class LEDDataset(Dataset):
    def __init__(self, input_ids, attention_masks, global_attention_masks, label_ids):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.global_attention_masks = global_attention_masks
        self.label_ids = label_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "global_attention_mask": self.global_attention_masks[idx],
            "labels": self.label_ids[idx],
        }


def prepare_led_dataset(
    transcripts: List[str],
    summaries: List[str],
    tokenizer,
    max_input: int = MAX_INPUT_TOKENS,
    max_target: int = MAX_TARGET_TOKENS,
) -> LEDDataset:
    encodings = tokenizer(
        transcripts,
        max_length=max_input,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Global attention on BOS (index 0) for every sample
    global_attention_mask = torch.zeros_like(encodings["input_ids"])
    global_attention_mask[:, 0] = 1

    with tokenizer.as_target_tokenizer():
        label_enc = tokenizer(
            summaries,
            max_length=max_target,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    label_ids = label_enc["input_ids"]
    label_ids[label_ids == tokenizer.pad_token_id] = -100

    return LEDDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        global_attention_mask,
        label_ids,
    )


def finetune_led(
    train_transcripts: List[str],
    train_summaries: List[str],
    val_transcripts: List[str],
    val_summaries: List[str],
    output_dir: str,
    device: Optional[str] = None,
):
    """Fine-tune LED-Base-16384 with gradient checkpointing."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Enable gradient checkpointing before moving to device
    model.gradient_checkpointing_enable()
    model = model.to(device)

    train_ds = prepare_led_dataset(train_transcripts, train_summaries, tokenizer)
    val_ds = prepare_led_dataset(val_transcripts, val_summaries, tokenizer)

    train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=output_dir,
        training_args_kwargs={
            "learning_rate": 5e-5,
            "fp16": device == "cuda",
            "gradient_checkpointing": True,
        },
        early_stopping_patience=3,
    )

    return model, tokenizer
