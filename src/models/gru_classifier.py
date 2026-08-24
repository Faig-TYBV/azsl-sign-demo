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
        pooling: str = "last",
    ):
        super().__init__()
        if pooling not in ("last", "mean_max"):
            raise ValueError(f"Unknown pooling: {pooling!r} (use 'last' or 'mean_max')")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.pooling = pooling

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

        self.head_dropout = nn.Dropout(dropout)
        gru_out = hidden_size * (2 if bidirectional else 1)
        # 'last': final hidden state [B, hidden]
        # 'mean_max': concat of temporal mean & max over all outputs [B, 2*hidden]
        fc_in = gru_out * 2 if pooling == "mean_max" else gru_out
        self.fc = nn.Linear(fc_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, input_size] -> logits: [batch, num_classes]"""
        outputs, h_n = self.gru(x)  # outputs [B,T,H], h_n [L*dirs,B,H]
        if self.pooling == "mean_max":
            pooled = torch.cat(
                [outputs.mean(dim=1), outputs.amax(dim=1)], dim=1
            )  # [B, 2H]
        else:
            pooled = h_n[-1]  # final layer's hidden state: [B, H]
        logits = self.fc(self.head_dropout(pooled))
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
            "pooling": self.pooling,
        }
