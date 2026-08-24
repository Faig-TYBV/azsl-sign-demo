"""
GRU baseline classifier for AzSLD word recognition.

Consumes sequences of shape [batch, seq_len=26, input_size=126] and
classifies each sequence into one of ``num_classes`` classes using the
final GRU hidden state followed by dropout and a linear head.
"""

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """Unidirectional GRU sequence classifier (baseline)."""

    def __init__(
        self,
        input_size: int = 126,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 200,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Dropout between GRU layers is only applied when num_layers > 1.
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )

        final_dropout = dropout if num_layers == 1 else dropout
        self.head_dropout = nn.Dropout(final_dropout)
        gru_out = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(gru_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, input_size] -> logits: [batch, num_classes]"""
        _, h_n = self.gru(x)  # h_n: [num_layers * dirs, batch, hidden]
        last = h_n[-1]  # final layer's hidden state: [batch, hidden]
        logits = self.fc(self.head_dropout(last))
        return logits

    def get_config(self) -> dict:
        """Constructor arguments, suitable for checkpoint storage."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
        }
