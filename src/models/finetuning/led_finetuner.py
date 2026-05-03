"""
LED-Base-16384 fine-tuning on football match summarization.

Strategy: fine-tune on (random 8192-token window of transcript, full summary) pairs.
LED can natively handle 16384 tokens but at that length the optimizer state
plus gradients OOM on a single T4. We therefore train at 8192 with random-window
sampling: each epoch the dataset re-samples a different 8k-token slice of every
transcript so over training the model is exposed to roughly the full match.

Key requirements:
  - global_attention_mask must be set for the BOS token
  - gradient_checkpointing=True is essential to fit on T4 (15GB VRAM)
  - fp16=True halves memory usage
"""

import random
from typing import List, Optional, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.models.finetuning.trainer import train_model

MODEL_NAME = "allenai/led-base-16384"
MAX_INPUT_TOKENS = 8192   # window length the model sees per training step
MAX_TARGET_TOKENS = 256


class LEDDataset(Dataset):
    """Pre-tokenises full transcripts; training samples a random window per epoch,
    validation always returns the first `window` tokens so eval is deterministic
    (otherwise early-stopping on noisy val ROUGE is unreliable)."""
    def __init__(self, full_input_ids: List[torch.Tensor], label_ids: torch.Tensor,
                 window: int, pad_token_id: int, train: bool = True):
        self.full_input_ids = full_input_ids
        self.label_ids = label_ids
        self.window = window
        self.pad_token_id = pad_token_id
        self.train = train

    def __len__(self):
        return len(self.full_input_ids)

    def __getitem__(self, idx):
        ids = self.full_input_ids[idx]
        n = ids.shape[0]
        if n > self.window:
            start = random.randint(0, n - self.window) if self.train else 0
            ids = ids[start:start + self.window]
        elif n < self.window:
            pad_len = self.window - n
            ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
        attn = (ids != self.pad_token_id).long()
        global_attn = torch.zeros_like(ids)
        global_attn[0] = 1
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "global_attention_mask": global_attn,
            "labels": self.label_ids[idx],
        }


def prepare_led_dataset(
    transcripts: List[str],
    summaries: List[str],
    tokenizer,
    max_input: int = MAX_INPUT_TOKENS,
    max_target: int = MAX_TARGET_TOKENS,
    train: bool = True,
) -> LEDDataset:
    # Tokenise each transcript at its FULL length (no truncation here — sampler picks a window per epoch).
    full_input_ids = [
        tokenizer(t, return_tensors="pt", truncation=False)["input_ids"][0]
        for t in transcripts
    ]

    label_enc = tokenizer(
        text_target=summaries,
        max_length=max_target,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    label_ids = label_enc["input_ids"]
    label_ids[label_ids == tokenizer.pad_token_id] = -100

    return LEDDataset(full_input_ids, label_ids, window=max_input,
                      pad_token_id=tokenizer.pad_token_id, train=train)


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

    # Force eval generation to emit >= 80 tokens. Seq2SeqTrainingArguments has no
    # generation_min_length kwarg; we set it on the model's generation_config so
    # Trainer.generate() picks it up during predict_with_generate eval.
    model.generation_config.min_length = 80
    model.generation_config.no_repeat_ngram_size = 3

    # Enable gradient checkpointing before moving to device
    model.gradient_checkpointing_enable()
    model = model.to(device)

    train_ds = prepare_led_dataset(train_transcripts, train_summaries, tokenizer, train=True)
    val_ds = prepare_led_dataset(val_transcripts, val_summaries, tokenizer, train=False)

    train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=output_dir,
        training_args_kwargs={
            "learning_rate": 2e-5,
            "label_smoothing_factor": 0.0,
            # Longer warmup: previous runs spiked grad_norm to 200-577 in early
            # epochs. 100 steps over ~150 total = 2/3 warmup but stops the spike.
            "warmup_steps": 100,
            "fp16": device == "cuda",
            "gradient_checkpointing": True,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "per_device_eval_batch_size": 1,
            # Don't set eval_accumulation_steps: with predict_with_generate=True
            # it triggers `OverflowError: out of range integral type conversion`
            # in transformers' CPU accumulation buffers when sequences contain
            # -100 label padding. Default (None = keep on GPU) is fine at bs=1.
            "generation_max_length": 192,
            "generation_num_beams": 2,
        },
        early_stopping_patience=2,
    )

    return model, tokenizer
