"""Glow-TTS / Matcha-TTS style text encoder.

Verbatim port of `matcha/models/components/text_encoder.py` from Matcha-TTS,
adapted to dnn-impl's encoder contract:

- forward signature: ``(x: (B, T, C), xlens: (B,), spk_emb=None) ->
  ((B, T, C), xlens)``
- exposes ``self.hidden_size`` (the channel count `CFMbasedModel` reads after
  instantiation)
- exposes class constants ``WANTS_EMBEDDING_SCALE``, ``WANTS_ABSOLUTE_PE``,
  ``EMBEDDING_INIT_STYLE``, ``BUNDLES_SPK_CONCAT`` so the framework can
  conditionally apply the Vaswani-style embedding scale, skip absolute PE,
  re-init the embedding with ``normal_(0, d^-0.5)``, and route speaker
  conditioning to the right point in the encoder pipeline.

The encoder bundles its own optional ``ConvReluNorm`` prenet (Glow-TTS style,
heavy dropout=0.5) so the heavy regularization that prevents the slow
encoder-drift collapse stays self-contained.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Glow-TTS LayerNorm (channel-first, eps=1e-4)
# ----------------------------------------------------------------------------


class GlowTTSLayerNorm(nn.Module):
    """LayerNorm over the channel dim with eps=1e-4.

    Operates on tensors shaped ``(B, C, T)`` (or ``(B, C, ...)``). Mean and
    variance are computed over the channel dim, learnable affine via
    ``gamma``/``beta`` reshaped for broadcasting.
    """

    def __init__(self, channels: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        shape = [1, -1] + [1] * (x.dim() - 2)
        return x * self.gamma.view(*shape) + self.beta.view(*shape)


# ----------------------------------------------------------------------------
# ConvReluNorm prenet (residual; final 1x1 proj zero-initialized)
# ----------------------------------------------------------------------------


class ConvReluNorm(nn.Module):
    """Glow-TTS prenet: stacked Conv1d -> LN -> ReLU -> Dropout, then a 1x1
    projection that is *zero-initialized* so the prenet starts as the
    identity.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(
                in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
            )
        )
        self.norm_layers.append(GlowTTSLayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.norm_layers.append(GlowTTSLayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        # Zero init so the prenet starts as the identity (residual passthrough).
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x_org = x
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            x = conv(x * x_mask)
            x = norm(x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


# ----------------------------------------------------------------------------
# Rotary positional embedding (applied to half the head dim)
# ----------------------------------------------------------------------------


class RotaryPositionalEmbedding(nn.Module):
    """RoPE applied to a fraction of head channels.

    Implementation mirrors ``RotaryPositionalEmbeddings`` from Matcha-TTS,
    which itself follows ``nn.labml.ai/transformers/rope``. The encoder uses
    this for both query and key projections, with ``d`` set to half the head
    channel count (so RoPE rotates half the features and the rest pass
    through unchanged).
    """

    def __init__(self, d: int, base: int = 10_000) -> None:
        super().__init__()
        self.base = base
        self.d = int(d)
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def _build_cache(self, x: torch.Tensor) -> None:
        if (
            self.cos_cached is not None
            and x.shape[0] <= self.cos_cached.shape[0]
            and self.cos_cached.device == x.device
            and self.cos_cached.dtype == x.dtype
        ):
            return

        seq_len = x.shape[0]
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.d, 2, device=x.device).float() / self.d)
        )
        seq_idx = torch.arange(seq_len, device=x.device).float()
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        self.cos_cached = idx_theta2.cos().to(x.dtype)[:, None, None, :]
        self.sin_cached = idx_theta2.sin().to(x.dtype)[:, None, None, :]

    def _neg_half(self, x: torch.Tensor) -> torch.Tensor:
        d_2 = self.d // 2
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, h, t, d_k)  ->  (t, b, h, d_k)
        x = x.permute(2, 0, 1, 3).contiguous()
        self._build_cache(x)

        x_rope, x_pass = x[..., : self.d], x[..., self.d :]
        neg_half_x = self._neg_half(x_rope)
        assert self.cos_cached is not None and self.sin_cached is not None
        x_rope = (x_rope * self.cos_cached[: x.shape[0]]) + (
            neg_half_x * self.sin_cached[: x.shape[0]]
        )
        out = torch.cat([x_rope, x_pass], dim=-1)
        # back to (b, h, t, d_k)
        return out.permute(1, 2, 0, 3).contiguous()


# ----------------------------------------------------------------------------
# Multi-head attention (Conv1d 1x1 QKV/O, RoPE on Q and K)
# ----------------------------------------------------------------------------


class GlowTTSMultiHeadAttention(nn.Module):
    """Multi-head self-attention from Glow-TTS / Matcha.

    Uses 1x1 convolutions for Q/K/V/O projections (functionally identical to
    Linear layers but kept channel-first to match the rest of the encoder).
    Q and K weights are ``xavier_uniform_`` initialized; V is also
    ``xavier_uniform_``. RoPE is applied to half of each head's channels.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert channels % n_heads == 0, (
            f"channels ({channels}) must be divisible by n_heads ({n_heads})"
        )

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)

        # RoPE on half the head channels (matches Matcha exactly).
        self.query_rotary_pe = RotaryPositionalEmbedding(int(self.k_channels * 0.5))
        self.key_rotary_pe = RotaryPositionalEmbedding(int(self.k_channels * 0.5))

        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run self-attention.

        Args:
            x: query input ``(B, C, T)``.
            c: key/value input ``(B, C, T)`` (same as ``x`` for self-attn).
            attn_mask: ``(B, 1, T_q, T_k)`` float mask, 1=valid, 0=ignore.

        Returns:
            ``(B, C, T)`` self-attended output.
        """
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        b, _, t_t = q.size()
        t_s = k.size(2)

        # (B, C, T) -> (B, n_heads, T, k_channels)
        q = q.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        k = k.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        v = v.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        q = self.query_rotary_pe(q)
        k = self.key_rotary_pe(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.k_channels)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)
        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        # (B, n_heads, T, k_channels) -> (B, C, T)
        output = torch.matmul(p_attn, v)
        output = output.transpose(2, 3).contiguous().view(b, self.channels, t_t)
        return self.conv_o(output)


# ----------------------------------------------------------------------------
# Conv FFN (kernel=3, positional FFN)
# ----------------------------------------------------------------------------


class GlowTTSConvFFN(nn.Module):
    """Two ``Conv1d(kernel_size)`` layers with ReLU + dropout in between.

    Gives the encoder block positional context inside the FFN, which is
    critical for MAS-based alignment to remain stable over long training.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.conv_2 = nn.Conv1d(
            filter_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


# ----------------------------------------------------------------------------
# One encoder layer (post-norm: x = LN(x + sublayer))
# ----------------------------------------------------------------------------


class GlowTTSEncoderBlock(nn.Module):
    """One Glow-TTS encoder layer.

    Pattern (per Matcha):
        x  = x * x_mask                          # zero out padded positions
        y  = attn(x, x, attn_mask)               # self-attention with RoPE
        y  = drop(y)
        x  = LN(x + y)                           # post-norm
        y  = ffn(x, x_mask)                      # conv FFN k=3
        y  = drop(y)
        x  = LN(x + y)                           # post-norm
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        kernel_size: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.attn = GlowTTSMultiHeadAttention(
            hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout
        )
        self.norm_1 = GlowTTSLayerNorm(hidden_channels)
        self.ffn = GlowTTSConvFFN(
            hidden_channels,
            hidden_channels,
            filter_channels,
            kernel_size,
            p_dropout=p_dropout,
        )
        self.norm_2 = GlowTTSLayerNorm(hidden_channels)
        self.drop = nn.Dropout(p_dropout)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        x = x * x_mask
        y = self.attn(x, x, attn_mask)
        y = self.drop(y)
        x = self.norm_1(x + y)

        y = self.ffn(x, x_mask)
        y = self.drop(y)
        x = self.norm_2(x + y)
        return x


# ----------------------------------------------------------------------------
# Top-level Glow-TTS encoder
# ----------------------------------------------------------------------------


def _sequence_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """Float mask ``(B, T)`` where ``True``/``1`` = valid, ``False``/``0`` = pad.

    This module deliberately uses a private helper instead of importing
    ``lens_to_mask`` because that returns ``True``=pad, which is the opposite
    polarity from what the conv-based pipeline wants.
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    indices = torch.arange(max_len, device=lengths.device)
    return (indices.unsqueeze(0) < lengths.unsqueeze(1)).to(lengths.device)


class GlowTTSEncoder(nn.Module):
    """Top-level Glow-TTS / Matcha-TTS text encoder block.

    Inputs / outputs match the existing ``TransformerEncoder`` contract:
    ``forward(x: (B, T, C), xlens: (B,), spk_emb=None) -> ((B, T, C), xlens)``.

    Internally the module is channel-first (``(B, C, T)``) for the conv-based
    prenet/attention/FFN; the in/out transposes are localized to ``forward``.

    Class constants tell ``CFMbasedModel`` how to prepare the embedding stream
    feeding this encoder:

    - ``WANTS_EMBEDDING_SCALE = True``: framework should multiply the embedding
      by ``sqrt(d)`` (Vaswani recipe; required because the embedding init is
      ``normal_(0, d^-0.5)``).
    - ``WANTS_ABSOLUTE_PE = False``: framework should NOT add the sinusoidal
      absolute PE to the embedding stream — RoPE inside attention covers it.
    - ``EMBEDDING_INIT_STYLE = "normal_scaled"``: framework should re-init
      ``self.embedding.weight`` with ``normal_(0, d^-0.5)``.
    - ``BUNDLES_SPK_CONCAT = True``: framework should NOT pre-concat the
      speaker embedding to the text stream; pass it via the ``spk_emb`` kwarg
      and let this module concat it after the prenet.
    """

    WANTS_EMBEDDING_SCALE: bool = True
    WANTS_ABSOLUTE_PE: bool = False
    EMBEDDING_INIT_STYLE: str = "normal_scaled"
    BUNDLES_SPK_CONCAT: bool = True

    def __init__(
        self,
        hidden_size: int,
        filter_channels: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        prenet: bool = True,
        prenet_kernel_size: int = 5,
        prenet_n_layers: int = 3,
        prenet_dropout: float = 0.5,
        spk_emb_dim: int = 0,
    ) -> None:
        super().__init__()
        self.text_emb_dim = hidden_size  # the channel count BEFORE speaker concat
        self.spk_emb_dim = spk_emb_dim
        # `hidden_size` (the attribute CFMbasedModel reads) reflects the
        # channel count of the encoder OUTPUT, which equals text_emb_dim +
        # spk_emb_dim when speaker conditioning is bundled.
        self.hidden_size = hidden_size + spk_emb_dim
        self.filter_channels = filter_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        if prenet:
            self.prenet: nn.Module = ConvReluNorm(
                self.text_emb_dim,
                self.text_emb_dim,
                self.text_emb_dim,
                kernel_size=prenet_kernel_size,
                n_layers=prenet_n_layers,
                p_dropout=prenet_dropout,
            )
        else:
            self.prenet = nn.Identity()

        self.layers = nn.ModuleList(
            [
                GlowTTSEncoderBlock(
                    self.hidden_size,
                    filter_channels,
                    num_heads,
                    kernel_size,
                    p_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        xlens: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``(B, T, text_emb_dim)`` text embedding stream (already scaled
                by ``sqrt(d)`` by the framework).
            xlens: ``(B,)`` text token lengths.
            spk_emb: optional ``(B, spk_emb_dim)`` speaker embedding. If
                provided, concatenated to the channel dim AFTER the prenet
                and BEFORE the encoder blocks.

        Returns:
            ``((B, T, hidden_size), xlens)`` encoder output. Note that
            ``hidden_size`` here equals ``text_emb_dim + spk_emb_dim`` when
            speaker conditioning is enabled.
        """
        # (B, T, C) -> (B, C, T) for the conv-based pipeline
        x = x.transpose(1, 2)
        max_len = x.size(2)
        x_mask = (
            _sequence_mask(xlens, max_len).to(dtype=x.dtype, device=x.device).unsqueeze(1)
        )  # (B, 1, T)

        x = self.prenet(x, x_mask) if isinstance(self.prenet, ConvReluNorm) else self.prenet(x)

        if spk_emb is not None:
            if self.spk_emb_dim == 0:
                raise ValueError(
                    "GlowTTSEncoder received spk_emb but was constructed with "
                    "spk_emb_dim=0; pass spk_emb_dim at construction time."
                )
            # (B, spk_emb_dim) -> (B, spk_emb_dim, T)
            spk_t = spk_emb.unsqueeze(-1).expand(-1, -1, x.size(2))
            x = torch.cat([x, spk_t], dim=1)

        # attention mask (B, 1, T_q, T_k) for the bool-fill in scores
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)

        for layer in self.layers:
            x = layer(x, x_mask, attn_mask)

        # zero pad positions one last time so the framework's mu_x masking is
        # consistent with what every intermediate layer saw
        x = x * x_mask

        # back to (B, T, C)
        x = x.transpose(1, 2)
        return x, xlens
