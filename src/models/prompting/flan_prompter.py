"""
FLAN-T5-Large prompter for chunk-based summarization.

Model: google/flan-t5-large (780M params)
Max input: 512 tokens → use 450-token chunks (see chunker.py)
"""

from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.models.prompting.base_prompter import BasePrompter

MODEL_NAME = "google/flan-t5-large"
MAX_INPUT_TOKENS = 450  # safe limit for chunker


class FlanPrompter(BasePrompter):

    def __init__(self, device: Optional[str] = None):
        super().__init__(MODEL_NAME, device)

    def _load_model(self) -> None:
        print(f"Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        print("Model loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
