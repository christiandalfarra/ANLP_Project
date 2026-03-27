"""
LED-Base-16384 prompter for long-context (full-transcript) summarization.

Model: allenai/led-base-16384
Max input: 16384 tokens — processes the full match transcript in one pass.
Global attention is placed on the first token (BOS) so the decoder can
attend to the full input via the encoder's global attention mechanism.
"""

from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.models.prompting.base_prompter import BasePrompter

MODEL_NAME = "allenai/led-base-16384"
MAX_INPUT_TOKENS = 16384


class LEDPrompter(BasePrompter):

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
            max_length=MAX_INPUT_TOKENS,
        ).to(self.device)

        # Global attention on BOS token
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[:, 0] = 1

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                global_attention_mask=global_attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
