import torch
import torch.nn as nn


class RNNDecoder(nn.Module):
    def __init__(
        self, hidden_size: int, vocab_size: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.rnn = nn.RNN(
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
        self, previous_hid: torch.Tensor, previous_token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """forward path for inference

        Args:
            previous_hid (torch.Tensor): previous hidden state of shape (num_layers, beam_size, hidden_size).
            previous_token (torch.Tensor): previous token of shape (beam_size, 1)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: output of shape (beam_size, 1, hidden_size) and hidden state of this step of (num_layers, beam_size, hidden_size).
        """
        previous_token_emb = self.embedding(previous_token)
        output, this_step_hid = self.rnn(previous_token_emb, previous_hid)

        return output, this_step_hid
