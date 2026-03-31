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


class ResNet1DUNet(nn.Module):
    """Minimal UNet decoder for flow matching TTS.

    Architecture: Down blocks -> Mid blocks -> Up blocks with skip connections.
    Timestep is injected into every ResnetBlock1D.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...] = (256, 256),
        num_mid_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # timestep embedding
        self.time_embeddings = TimeStepEmbedding(
            hidden_size=in_channels, output_size=channels[0] * 4
        )
        time_emb_dim = channels[0] * 4

        # down blocks
        self.down_blocks = nn.ModuleList()
        ch_in = in_channels * 2  # x and mu are concatenated
        for i, ch_out in enumerate(channels):
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(ch_in, ch_out, time_emb_dim)
            downsample = (
                Downsample1D(ch_out)
                if not is_last
                else nn.Conv1d(ch_out, ch_out, 3, padding=1)
            )
            self.down_blocks.append(nn.ModuleList([resnet, downsample]))
            ch_in = ch_out

        # mid blocks
        self.mid_blocks = nn.ModuleList()
        for _ in range(num_mid_blocks):
            self.mid_blocks.append(ResnetBlock1D(channels[-1], channels[-1], time_emb_dim))

        # up blocks
        reversed_channels = channels[::-1] + (channels[0],)
        self.up_blocks = nn.ModuleList()
        for i in range(len(reversed_channels) - 1):
            ch_in_up = reversed_channels[i] * 2  # skip connection doubles channels
            ch_out_up = reversed_channels[i + 1]
            is_last = i == len(reversed_channels) - 2
            resnet = ResnetBlock1D(ch_in_up, ch_out_up, time_emb_dim)
            upsample = (
                Upsample1D(ch_out_up)
                if not is_last
                else nn.Conv1d(ch_out_up, ch_out_up, 3, padding=1)
            )
            self.up_blocks.append(nn.ModuleList([resnet, upsample]))

        # final projection
        self.final_block = Block1D(channels[0], channels[0])
        self.final_proj = nn.Conv1d(channels[0], out_channels, 1)

        self._initialize_weights()

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
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Noisy mel of shape (batch_size, in_channels, time).
            mask (torch.Tensor): Padding mask of shape (batch_size, 1, time). 1=valid, 0=pad.
            mu (torch.Tensor): Encoder output of shape (batch_size, in_channels, time).
            t (torch.Tensor): Timestep of shape (batch_size,).

        Returns:
            torch.Tensor: Predicted velocity of shape (batch_size, out_channels, time).
        """
        t_emb = self.time_embeddings(t)

        x = torch.cat([x, mu], dim=1)  # (B, 2*in_channels, T)

        # down
        hiddens = []
        masks = [mask]
        for resnet, downsample in self.down_blocks:  # type: ignore[misc]
            mask_down = masks[-1]
            x = resnet(x, mask_down, t_emb)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        # mid
        masks.pop()
        mask_mid = masks[-1]
        for resnet in self.mid_blocks:
            x = resnet(x, mask_mid, t_emb)

        # up
        for resnet, upsample in self.up_blocks:  # type: ignore[misc]
            mask_up = masks.pop()
            x = torch.cat([x, hiddens.pop()], dim=1)  # skip connection
            x = resnet(x, mask_up, t_emb)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        x = self.final_proj(x * mask_up)

        return x * mask
