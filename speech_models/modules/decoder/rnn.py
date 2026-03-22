from typing import Literal

import torch
import torch.nn as nn


class RNNDecoder(nn.Module):
    def __init__(
        self,
        rnn_type: Literal["rnn", "lstm"],
        hidden_size: int,
        vocab_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        match rnn_type:
            case "rnn":
                self.rnn = nn.RNN(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                )
            case "lstm":
                self.rnn = nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                )

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """forward path.

        Args:
            x (torch.Tensor): label tokens of shape (batch_size, seq_len).
            xlens (torch.Tensor): label token lengths of shae (batch_size, )

        Returns:
            tuple[torch.Tensor, torch.Tensor]: output
            of rnn of shape (batch_size, seq_len, hidden_size) and its lengths of shape (batch_size, )

        Examples:
            x = token_to_ids([`<blank>`, `i`, `am`, `a`, `cat`])
        """
        x = self.embedding(x)
        output, _ = self.rnn(x)  # h_0 is initialized automatically
        return output, xlens

    def inference_forward(
        self,
        previous_hid: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        previous_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """forward path for inference

        Args:
            previous_hid (torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
                previous hidden state. For RNN, it's a tensor of shape (num_layers, beam_size, hidden_size).
                For LSTM, it's a tuple of two tensors, each of shape (num_layers, beam_size, hidden_size).
            previous_token (torch.Tensor): previous token of shape (beam_size, 1)

        Returns:
            tuple: output of shape (beam_size, 1, hidden_size) and hidden state of this step.
        """
        previous_token_emb = self.embedding(previous_token)
        output, this_step_hid = self.rnn(previous_token_emb, previous_hid)

        return output, this_step_hid
