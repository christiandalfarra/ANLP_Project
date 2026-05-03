"""
HuggingFace Trainer wrapper with early stopping on validation ROUGE-L.
Used by both BART and LED fine-tuning.
"""

import os
import numpy as np
from typing import Dict, Any, Optional

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
from rouge_score import rouge_scorer


def compute_rouge_metrics(tokenizer):
    """Returns a compute_metrics function for use with Seq2SeqTrainer."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def _compute(eval_preds):
        preds, labels = eval_preds

        # Replace -100 (padding label) with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        r1_list, r2_list, rl_list = [], [], []
        for pred, ref in zip(decoded_preds, decoded_labels):
            scores = scorer.score(ref.strip(), pred.strip())
            r1_list.append(scores["rouge1"].fmeasure)
            r2_list.append(scores["rouge2"].fmeasure)
            rl_list.append(scores["rougeL"].fmeasure)

        return {
            "rouge1": np.mean(r1_list),
            "rouge2": np.mean(r2_list),
            "rougeL": np.mean(rl_list),
        }

    return _compute


def build_training_args(
    output_dir: str,
    learning_rate: float = 3e-5,
    num_train_epochs: int = 15,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    eval_accumulation_steps: int = 1,
    gradient_accumulation_steps: int = 8,
    generation_num_beams: int = 4,
    warmup_steps: int = 50,
    label_smoothing_factor: float = 0.1,
    fp16: bool = True,
    gradient_checkpointing: bool = False,
    predict_with_generate: bool = True,
    generation_max_length: int = 256,
    save_total_limit: int = 1,
    save_only_model: bool = True,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "rougeL",
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
) -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_accumulation_steps=eval_accumulation_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        generation_num_beams=generation_num_beams,
        warmup_steps=warmup_steps,
        label_smoothing_factor=label_smoothing_factor,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        predict_with_generate=predict_with_generate,
        generation_max_length=generation_max_length,
        save_total_limit=save_total_limit,
        save_only_model=save_only_model,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        weight_decay=0.01,
        logging_steps=10,
        report_to="none",  # disable wandb/tensorboard by default
    )


def train_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir: str,
    training_args_kwargs: Optional[Dict[str, Any]] = None,
    early_stopping_patience: int = 3,
):
    """
    Train a seq2seq model with early stopping on validation ROUGE-L.
    Returns the trained model.
    """
    kwargs = training_args_kwargs or {}
    args = build_training_args(output_dir=output_dir, **kwargs)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    compute_metrics = compute_rouge_metrics(tokenizer)

    # transformers >=4.46 renamed `tokenizer=` to `processing_class=`. Try the
    # new name first, fall back for older versions.
    common_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    try:
        trainer = Seq2SeqTrainer(processing_class=tokenizer, **common_kwargs)
    except TypeError:
        trainer = Seq2SeqTrainer(tokenizer=tokenizer, **common_kwargs)

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    return model
