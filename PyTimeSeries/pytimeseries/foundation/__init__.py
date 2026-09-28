"""Time-series foundation models (Part 11): pretrained networks applied
**zero-shot** — no local training. Each is a thin client with lazy imports and
clear errors; running them needs model weights (Hugging Face) or an API key,
declared via the ``[foundation]`` extra."""
from .clients import Chronos, TimesFM, Moirai, TimeGPT

__all__ = ["Chronos", "TimesFM", "Moirai", "TimeGPT"]
