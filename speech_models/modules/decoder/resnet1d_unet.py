import torch
import torch.nn as nn

from speech_models.modules.others.tts.time_step_embedding import TimeStepEmbedding


class Block1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.block(x * mask) * mask


class ResnetBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor
    ) -> torch.Tensor:
        h = self.block1(x, mask)
        h = h + self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        return h + self.res_conv(x * mask)


class Downsample1D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep embedding."""

    def __init__(self, dim: int, time_emb_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_emb_dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): (batch_size, seq_len, dim).
            time_emb (torch.Tensor): (batch_size, time_emb_dim).

        Returns:
            torch.Tensor: (batch_size, seq_len, dim).
        """
        emb = self.linear(self.silu(time_emb)).unsqueeze(1)  # (B, 1, 2*dim)
        scale, shift = emb.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class TransformerBlock1D(nn.Module):
    """Self-attention block for 1D sequences with AdaLN timestep conditioning."""

    def __init__(
        self,
        dim: int,
        time_emb_dim: int,
        num_heads: int = 4,
        ff_mult: int = 1,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.norm1 = AdaLayerNorm(dim, time_emb_dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = AdaLayerNorm(dim, time_emb_dim)
        ff_dim = dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): (batch_size, channels, time) channel-first.
            mask (torch.Tensor): (batch_size, 1, time) where 1=valid, 0=pad.
            time_emb (torch.Tensor): (batch_size, time_emb_dim).

        Returns:
            torch.Tensor: (batch_size, channels, time).
        """
        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        key_padding_mask = mask.squeeze(1) < 0.5  # (B, T), True = ignore

        residual = x
        x = self.norm1(x, time_emb)
        x = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)[0]
        x = x + residual

        residual = x
        x = self.norm2(x, time_emb)
        x = self.ff(x)
        x = x + residual

        # (B, T, C) -> (B, C, T)
        return x.transpose(1, 2)


class ResNet1DUNet(nn.Module):
    """UNet decoder for flow matching TTS.

    Architecture: Down blocks -> Mid blocks -> Up blocks with skip connections.
    Timestep is injected into every ResnetBlock1D.
    Optionally includes TransformerBlock1D after each ResnetBlock1D.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...] = (256, 256),
        num_res_blocks: int = 1,
        num_mid_blocks: int = 2,
        n_transformer_blocks: int = 0,
        num_heads: int = 4,
        ff_mult: int = 1,
        dropout: float = 0.05,
        spk_emb_dim: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        channels = tuple(channels)
        # number of stride-2 downsamples (all except last down block)
        self.num_downsamplings = max(len(channels) - 1, 0)

        # timestep embedding
        self.time_embeddings = TimeStepEmbedding(
            hidden_size=in_channels, output_size=channels[0] * 4
        )
        time_emb_dim = channels[0] * 4

        # speaker conditioning is via channel concat at the input (Matcha-TTS style)
        self.spk_emb_dim = spk_emb_dim

        # down blocks
        self.down_blocks = nn.ModuleList()
        ch_in = in_channels * 2 + spk_emb_dim  # x, mu, (optional) spk concatenated
        for i, ch_out in enumerate(channels):
            is_last = i == len(channels) - 1
            resnets = nn.ModuleList()
            for j in range(num_res_blocks):
                resnets.append(ResnetBlock1D(ch_in if j == 0 else ch_out, ch_out, time_emb_dim))
            transformer = self._make_transformer_blocks(
                ch_out, time_emb_dim, n_transformer_blocks, num_heads, ff_mult, dropout
            )
            downsample = (
                Downsample1D(ch_out)
                if not is_last
                else nn.Conv1d(ch_out, ch_out, 3, padding=1)
            )
            self.down_blocks.append(nn.ModuleList([resnets, transformer, downsample]))
            ch_in = ch_out

        # mid blocks
        self.mid_blocks = nn.ModuleList()
        for _ in range(num_mid_blocks):
            resnet = ResnetBlock1D(channels[-1], channels[-1], time_emb_dim)
            transformer = self._make_transformer_blocks(
                channels[-1], time_emb_dim, n_transformer_blocks, num_heads, ff_mult, dropout
            )
            self.mid_blocks.append(nn.ModuleList([resnet, transformer]))

        # up blocks
        reversed_channels = channels[::-1] + (channels[0],)
        self.up_blocks = nn.ModuleList()
        for i in range(len(reversed_channels) - 1):
            ch_in_up = reversed_channels[i] * 2  # skip connection doubles channels
            ch_out_up = reversed_channels[i + 1]
            is_last = i == len(reversed_channels) - 2
            resnets = nn.ModuleList()
            for j in range(num_res_blocks):
                resnets.append(ResnetBlock1D(ch_in_up if j == 0 else ch_out_up, ch_out_up, time_emb_dim))
            transformer = self._make_transformer_blocks(
                ch_out_up, time_emb_dim, n_transformer_blocks, num_heads, ff_mult, dropout
            )
            upsample = (
                Upsample1D(ch_out_up)
                if not is_last
                else nn.Conv1d(ch_out_up, ch_out_up, 3, padding=1)
            )
            self.up_blocks.append(nn.ModuleList([resnets, transformer, upsample]))

        # final projection
        self.final_block = Block1D(channels[0], channels[0])
        self.final_proj = nn.Conv1d(channels[0], out_channels, 1)

        self._initialize_weights()

    @staticmethod
    def _make_transformer_blocks(
        dim: int, time_emb_dim: int, n_blocks: int, num_heads: int, ff_mult: int, dropout: float
    ) -> nn.ModuleList:
        return nn.ModuleList(
            [TransformerBlock1D(dim, time_emb_dim, num_heads, ff_mult, dropout) for _ in range(n_blocks)]
        )

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Noisy mel of shape (batch_size, in_channels, time).
            mask (torch.Tensor): Padding mask of shape (batch_size, 1, time). 1=valid, 0=pad.
            mu (torch.Tensor): Encoder output of shape (batch_size, in_channels, time).
            t (torch.Tensor): Timestep of shape (batch_size,).
            spk_emb (torch.Tensor | None): Speaker embedding of shape (batch_size, spk_emb_dim).

        Returns:
            torch.Tensor: Predicted velocity of shape (batch_size, out_channels, time).
        """
        t_emb = self.time_embeddings(t)

        if spk_emb is not None:
            spk_t = spk_emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.cat([x, mu, spk_t], dim=1)  # (B, 2*in_channels + spk_emb_dim, T)
        else:
            x = torch.cat([x, mu], dim=1)  # (B, 2*in_channels, T)

        # down
        hiddens = []
        masks = [mask]
        for resnets, transformer_blocks, downsample in self.down_blocks:  # type: ignore[misc]
            mask_down = masks[-1]
            for resnet in resnets:
                x = resnet(x, mask_down, t_emb)
            for transformer_block in transformer_blocks:
                x = transformer_block(x, mask_down, t_emb)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        # mid
        masks.pop()
        mask_mid = masks[-1]
        for resnet, transformer_blocks in self.mid_blocks:  # type: ignore[misc]
            x = resnet(x, mask_mid, t_emb)
            for transformer_block in transformer_blocks:
                x = transformer_block(x, mask_mid, t_emb)

        # up
        for resnets, transformer_blocks, upsample in self.up_blocks:  # type: ignore[misc]
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = x[..., : skip.shape[-1]]  # trim if upsample overshot (odd lengths)
            x = torch.cat([x, skip], dim=1)  # skip connection
            for resnet in resnets:
                x = resnet(x, mask_up, t_emb)
            for transformer_block in transformer_blocks:
                x = transformer_block(x, mask_up, t_emb)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        x = self.final_proj(x * mask_up)

        return x * mask
