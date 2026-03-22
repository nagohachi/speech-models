import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiheadAttentionWithRelPositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads

        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
        self.hidden_size_each_head = hidden_size // self.num_heads
        self.wo = nn.Linear(self.hidden_size_each_head * self.num_heads, hidden_size)

        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(self.hidden_size_each_head)

        # positional-encoding related parameters
        self.pos_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_u = nn.Parameter(
            torch.Tensor(self.num_heads, self.hidden_size_each_head)
        )
        self.pos_v = nn.Parameter(
            torch.Tensor(self.num_heads, self.hidden_size_each_head)
        )

        nn.init.xavier_uniform_(self.pos_u)
        nn.init.xavier_uniform_(self.pos_v)

    def _split_into_each_head(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "b s (nheads hid_each) -> b nheads s hid_each",
            hid_each=self.hidden_size_each_head,
            nheads=self.num_heads,
        )

    def _shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shifting algorithm for relative positional encoding.

        Args:
            x (torch.Tensor): positional encoding of shape (b, nheads, s, 2s-1)

        Returns:
            torch.Tensor: shifted positional encoding of shape (b, nheads, s, s)
        """

        # Example (s = 3)
        """
        ij-th element means dot product of i-th query and j-th positional embedding vector

        1. given x is:
        00 01 02 03 04
        10 11 12 13 14
        20 21 22 23 24

        note that, for the column number, 
        - 0: distance -2
        - 1: distance -1
        - 2: distance 0
        - 3: distance 1
        - 4: distance 2

        2. zero padding, (s, 2s-1) -> (s, 2s):
        0 00 01 02 03 04
        0 10 11 12 13 14
        0 20 21 22 23 24

        3. reshape, (s, 2s) -> (2s, s):
        0  00 01
        02 03 04
        0  10 11
        12 13 14
        0  20 21
        22 23 24

        4. cut the first line:
        02 03 04
        0  10 11
        12 13 14
        0  20 21
        22 23 24

        5. reshape back: (2s-1, s) -> (s, 2s-1)
        02 03 04 0  10
        11 12 13 14 0
        20 21 22 23 24

        6. slice: (s, 2s-1) -> (s, s)
        02 03 04
        11 12 13
        20 21 22
        """

        b, nheads, s, _ = x.shape
        padded_x = F.pad(x, (1, 0))  # (b, nheads, s, 2s)
        reshaped_x = padded_x.view(b, nheads, 2 * s, s)
        cut_x = reshaped_x[:, :, 1:, :]
        reshaped_back_x = cut_x.view(b, nheads, s, 2 * s - 1)
        sliced_x = reshaped_back_x[:, :, :s, :s]

        return sliced_x

    def _calc_score_with_pos_enc(
        self, query: torch.Tensor, key: torch.Tensor, pos_emb: torch.Tensor
    ) -> torch.Tensor:
        """calculate score (before softmax) with rel positional encoding.

        Args:
            query (torch.Tensor): query of shape (b, nheads, s, hid_each)
            key (torch.Tensor): key of shape (b, nheads, s, hid_each)
            pos_emb (torch.Tensor): positional embedding shared across all the layers and heads of shape (1, 2*s-1, hidden_size)

        Returns:
            torch.Tensor: score of shape (b, nheads, s, s)
        """
        key_transposed = rearrange(key, "b nheads s hid_each -> b nheads hid_each s")

        p = self.pos_linear(pos_emb)  # (1, 2s-1, hidden_size)
        p = rearrange(
            p,
            "1 len_p (nheads hid_each) -> 1 nheads hid_each len_p",
            nheads=self.num_heads,
            hid_each=self.hidden_size_each_head,
        )

        u = rearrange(self.pos_u, "nheads hid_each -> nheads 1 hid_each")
        v = rearrange(self.pos_v, "nheads hid_each -> nheads 1 hid_each")

        matrix_ac = (query + u) @ key_transposed  # (b, nheads, s, s)
        matrix_bd = self._shift((query + v) @ p)  # (b, nheads, s, s)

        return (matrix_ac + matrix_bd) / self.scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        pos_emb: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        """forward path.

        Args:
            query (torch.Tensor): query tensor of shape (batch_size, seq_len, hidden_size).
            key (torch.Tensor): key tensor of shape (batch_size, seq_len, hidden_size).
            value (torch.Tensor): value tensor of shape (batch_size, seq_len, hidden_size).
            key_padding_mask (torch.Tensor): padding mask of shape (batch_size, seq_len).
            pos_emb (torch.Tensor): positional embedding shared across all the layers and heads of shape (batch_size, s, s)
            is_causal (bool): whether this self attention is causal (e.g. streaming mode).

        Returns:
            torch.Tensor: output tensor of shape (batch_size, seq_len, hidden_size).
        """
        if is_causal:
            raise NotImplementedError()

        query = self.wq(query)  # (b s hid)
        key = self.wk(key)  # (b s hid)
        value = self.wv(value)  # (b s hid)

        # split into each head
        query = self._split_into_each_head(query)  # (b nheads s hid_each)
        key = self._split_into_each_head(key)  # (b nheads s hid_each)
        value = self._split_into_each_head(value)  # (b nheads s hid_each)

        scores = self._calc_score_with_pos_enc(query, key, pos_emb)  # (b nheads s s)

        if key_padding_mask is not None:
            # erase "padding" key columns
            mask = rearrange(key_padding_mask, "b s -> b 1 1 s")
            scores = scores.masked_fill(mask, -1.0e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        qk_t_v = attn_weights @ value  # (b nheads s hid_each)

        return self.wo(
            rearrange(qk_t_v, "b nheads s hid_each -> b s (nheads hid_each)")
        )  # (b s hid)
