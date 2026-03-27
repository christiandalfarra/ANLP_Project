"""
Abstract base class for all prompting-based summarizers.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch


class BasePrompter(ABC):
    """
    Common interface for all prompt-based summarization models.
    Subclasses implement `_load_model` and `generate`.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load tokenizer and model onto self.device."""
        ...

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Run inference for a single prompt string.
        Returns the generated text (decoded, stripped).
        """
        ...

    def batch_generate(self, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
        """Default: sequential. Subclasses may override for batched inference."""
        return [self.generate(p, max_new_tokens) for p in prompts]
