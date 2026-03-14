import torch
import torch.nn as nn
from einops import rearrange


class Joiner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.lin_enc = nn.Linear(encoder_hidden_size, hidden_size)
        self.lin_dec = nn.Linear(decoder_hidden_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.lin_out = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        decoder_out: torch.Tensor,
        decoder_out_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward path.

        Args:
            encoder_out (torch.Tensor): encoder output of shape (batch_size, seq_len1, encoder_hidden_size).
            encoder_out_lens (torch.Tensor): encoder output lengths of shape (batch_size, ).
            decoder_out (torch.Tensor): decoder output of shape (batch_size, seq_len2, decoder_hidden_size).
            decoder_out_lens (torch.Tensor): decoder output lengths of shape (batch_size, ).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (joiner output, encoder_out_lens, decoder_out_lens).
            Joiner output is of shape (batch_size, T, U, vocab_size),
            where T := max(encoder_out_lens) and U := max(decoder_out_lens).
        """
        encoder_out = self.lin_enc(encoder_out)  # (bs, s1, hid)
        decoder_out = self.lin_dec(decoder_out)  # (bs, s2, hid)

        encoder_out_expanded = rearrange(encoder_out, "bs s1 hid -> bs s1 1 hid")
        decoder_out_expanded = rearrange(decoder_out, "bs s2 hid -> bs 1 s2 hid")

        joiner_out = self.lin_out(
            self.relu(encoder_out_expanded + decoder_out_expanded)
        )

        return joiner_out, encoder_out_lens, decoder_out_lens
