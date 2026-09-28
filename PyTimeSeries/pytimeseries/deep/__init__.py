"""Deep-learning forecasters (Part 9). Requires the ``[dl]`` extra (torch)."""
from .forecasters import (
    MLPForecaster, RNNForecaster, LSTMForecaster, GRUForecaster,
    TCNForecaster, Seq2SeqForecaster,
)

__all__ = [
    "MLPForecaster", "RNNForecaster", "LSTMForecaster", "GRUForecaster",
    "TCNForecaster", "Seq2SeqForecaster",
]
