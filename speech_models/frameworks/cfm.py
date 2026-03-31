from pathlib import Path

import einops
import torch
import torch.nn as nn
import yaml
from speech_models.modules.decoder.resnet1d_unet import ResNet1DUNet
from speech_models.modules.encoder import ConformerEncoder, TransformerEncoder
from speech_models.modules.others.tts.time_step_embedding import TimeStepEmbedding
from speech_models.modules.utils.mask import lens_to_mask
from speech_models.tokenizers import BPETokenizer, CharTokenizer

encoder_choices = dict(transformer=TransformerEncoder)
decoder_choices = dict(conformer=ConformerEncoder, resnet1d_unet=ResNet1DUNet)


class CFMbasedModel(nn.Module):
    """Conditional Flow Matching-based Text-to-Speech model."""

    def __init__(
        self,
        encoder_config_path: Path | str,
        decoder_config_path: Path | str,
        tokenizer: BPETokenizer | CharTokenizer,
        nmels: int,
    ) -> None:
        super().__init__()
        with open(encoder_config_path, "r") as f:
            c = yaml.safe_load(f)
            encoder_choice = c["encoder"]
            encoder_conf = c["encoder_conf"]
        with open(decoder_config_path, "r") as f:
            c = yaml.safe_load(f)
            decoder_choice = c["decoder"]
            decoder_conf = c["decoder_conf"]

        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        # text encoder
        self.encoder = encoder_choices[encoder_choice](**encoder_conf)
        enc_hidden: int = self.encoder.hidden_size

        # noise decoder
        self.decoder = decoder_choices[decoder_choice](**decoder_conf)
        self.use_unet = isinstance(self.decoder, ResNet1DUNet)

        # text embedding
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=enc_hidden,
            padding_idx=tokenizer.pad_token_id,
        )

        if self.use_unet:
            # UNet handles timestep, concat, and projection internally
            self.post_encoder_proj = nn.Linear(enc_hidden, nmels)
        else:
            assert isinstance(self.decoder, ConformerEncoder)
            dec_hidden_size: int = self.decoder.hidden_size
            self.post_encoder_proj = nn.Linear(enc_hidden, dec_hidden_size)
            self.pre_decoder_proj = nn.Linear(nmels, dec_hidden_size)
            self.post_decoder_proj = nn.Linear(dec_hidden_size, nmels)
            self.time_step_embedding = TimeStepEmbedding(
                hidden_size=dec_hidden_size,
                output_size=dec_hidden_size,
            )

        self.criterion = nn.MSELoss(reduction="none")

    def upsample(
        self,
        text_encoded: torch.Tensor,
        text_token_lens: torch.Tensor,
        gt_mel_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Uniformly upsample text_encoded to match gt_mel_lens.

        Args:
            text_encoded (torch.Tensor): Encoder output of shape (batch_size, text_seq_len, hidden_size).
            text_token_lens (torch.Tensor): Text lengths of shape (batch_size,).
            gt_mel_lens (torch.Tensor): Target mel lengths of shape (batch_size,).

        Returns:
            torch.Tensor: Upsampled tensor of shape (batch_size, max_mel_len, hidden_size).
        """
        batch_size, _, hidden = text_encoded.size()
        max_mel_len = int(gt_mel_lens.max().item())

        upsampled = torch.zeros(
            batch_size,
            max_mel_len,
            hidden,
            device=text_encoded.device,
            dtype=text_encoded.dtype,
        )

        for i in range(batch_size):
            t_len = int(text_token_lens[i].item())
            m_len = int(gt_mel_lens[i].item())

            base_dur = m_len // t_len
            remainder = m_len % t_len
            durations = torch.full((t_len,), base_dur, device=text_encoded.device)
            durations[:remainder] += 1

            upsampled[i, :m_len] = torch.repeat_interleave(
                text_encoded[i, :t_len], durations, dim=0
            )

        return upsampled

    def _make_mask(self, mel_lens: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
        """Create float mask (B, 1, T) where 1=valid, 0=pad."""
        if max_len is not None:
            indices = torch.arange(max_len, device=mel_lens.device)
            return (indices.unsqueeze(0) < mel_lens.unsqueeze(1)).unsqueeze(1).float()
        return (~lens_to_mask(mel_lens)).unsqueeze(1).float()

    def _fix_len(self, length: int) -> int:
        """Round up length to be compatible with UNet downsampling."""
        assert isinstance(self.decoder, ResNet1DUNet)
        factor = 2 ** self.decoder.num_downsamplings
        return int((length + factor - 1) // factor * factor)

    def _pad_to_len(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad (B, T, D) tensor along T to target_len."""
        pad_size = target_len - x.size(1)
        if pad_size > 0:
            return torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        return x

    def _forward_unet(
        self,
        text_tokens: torch.Tensor,
        text_token_lens: torch.Tensor,
        gt_mels: torch.Tensor,
        gt_mel_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = gt_mels.size(0)
        orig_mel_len = gt_mels.size(1)
        fixed_len = self._fix_len(orig_mel_len)

        time_steps = torch.rand(batch_size, device=gt_mels.device)
        gnoise = torch.randn(
            batch_size, fixed_len, gt_mels.size(2), device=gt_mels.device
        )

        text_embeds = self.embedding(text_tokens)
        text_encoded, _ = self.encoder(text_embeds, text_token_lens)

        # mu: (B, T_fixed, mel_dim) -> (B, mel_dim, T_fixed)
        mu = self.post_encoder_proj(
            self._pad_to_len(
                self.upsample(text_encoded, text_token_lens, gt_mel_lens), fixed_len
            )
        ).transpose(1, 2)

        # x_t: (B, T_fixed, mel_dim) -> (B, mel_dim, T_fixed)
        gt_mels_padded = self._pad_to_len(gt_mels, fixed_len)
        t = time_steps[:, None, None]
        x_t = ((1 - t) * gnoise + t * gt_mels_padded).transpose(1, 2)

        mask = self._make_mask(gt_mel_lens, max_len=fixed_len)

        # UNet: (B, mel_dim, T_fixed) -> (B, mel_dim, T_fixed)
        decoder_out = self.decoder(x_t, mask, mu, time_steps)

        # trim back and transpose: (B, mel_dim, T_fixed) -> (B, T_orig, mel_dim)
        decoder_out = decoder_out[:, :, :orig_mel_len].transpose(1, 2)
        gnoise = gnoise[:, :orig_mel_len, :]
        return decoder_out, gt_mel_lens, gnoise

    def _forward_conformer(
        self,
        text_tokens: torch.Tensor,
        text_token_lens: torch.Tensor,
        gt_mels: torch.Tensor,
        gt_mel_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = gt_mels.size(0)

        time_steps = torch.rand(batch_size, device=gt_mels.device)
        gnoise = torch.randn_like(gt_mels)

        text_embeds = self.embedding(text_tokens)
        text_encoded, _ = self.encoder(text_embeds, text_token_lens)

        text_encoded_upsampled = self.post_encoder_proj(
            self.upsample(text_encoded, text_token_lens, gt_mel_lens)
        )

        time_steps_unsq = time_steps[:, None, None]
        mel_t = self.pre_decoder_proj(
            (1 - time_steps_unsq) * gnoise + time_steps_unsq * gt_mels
        )

        embed_t = einops.rearrange(
            self.time_step_embedding(time_steps), "bs dec_hid -> bs 1 dec_hid"
        )

        decoder_out, _ = self.decoder(
            text_encoded_upsampled + mel_t + embed_t, gt_mel_lens
        )
        return self.post_decoder_proj(decoder_out), gt_mel_lens, gnoise

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_token_lens: torch.Tensor,
        gt_mels: torch.Tensor,
        gt_mel_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward path of CFM.

        Args:
            text_tokens (torch.Tensor): Text token ids of shape (batch_size, seq_len).
            text_token_lens (torch.Tensor): Text token lengths of shape (batch_size,).
            gt_mels (torch.Tensor): GT mel spectrograms of shape (batch_size, seq_len, mel_dim).
            gt_mel_lens (torch.Tensor): GT mel lengths of shape (batch_size,).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Predicted velocity field of shape (batch_size, seq_len, mel_dim).
                - mel spectrogram lengths.
                - gaussian noise.
        """
        if self.use_unet:
            return self._forward_unet(
                text_tokens, text_token_lens, gt_mels, gt_mel_lens
            )
        return self._forward_conformer(
            text_tokens, text_token_lens, gt_mels, gt_mel_lens
        )

    @torch.inference_mode()
    def inference_forward(
        self,
        text_tokens: torch.Tensor,
        text_token_lens: torch.Tensor,
        mel_lens: torch.Tensor,
        n_timesteps: int = 10,
    ) -> torch.Tensor:
        """Generate mel spectrograms from text via Euler ODE solver.

        Args:
            text_tokens (torch.Tensor): Text token ids of shape (batch_size, seq_len).
            text_token_lens (torch.Tensor): Text token lengths of shape (batch_size,).
            mel_lens (torch.Tensor): Target mel lengths of shape (batch_size,).
            n_timesteps (int): Number of Euler steps.

        Returns:
            torch.Tensor: Generated mel spectrogram of shape (batch_size, max_mel_len, mel_dim).
        """
        text_embeds = self.embedding(text_tokens)
        text_encoded, _ = self.encoder(text_embeds, text_token_lens)

        max_mel_len = int(mel_lens.max().item())
        batch_size = text_tokens.size(0)

        if self.use_unet:
            assert isinstance(self.decoder, ResNet1DUNet)
            nmels = self.decoder.out_channels
            fixed_len = self._fix_len(max_mel_len)

            mu = self.post_encoder_proj(
                self._pad_to_len(
                    self.upsample(text_encoded, text_token_lens, mel_lens), fixed_len
                )
            ).transpose(1, 2)  # (B, mel_dim, T_fixed)

            mask = self._make_mask(mel_lens, max_len=fixed_len)
            x = torch.randn(batch_size, nmels, fixed_len, device=text_tokens.device)

            t_span = torch.linspace(0, 1, n_timesteps + 1, device=text_tokens.device)
            for step in range(n_timesteps):
                t = t_span[step]
                dt = t_span[step + 1] - t
                time_steps = t.expand(batch_size)
                dphi_dt = self.decoder(x, mask, mu, time_steps)
                x = x + dt * dphi_dt

            return x[:, :, :max_mel_len].transpose(1, 2)  # (B, T, mel_dim)

        else:
            nmels = self.pre_decoder_proj.in_features
            mu = self.post_encoder_proj(
                self.upsample(text_encoded, text_token_lens, mel_lens)
            )

            x = torch.randn(batch_size, max_mel_len, nmels, device=text_tokens.device)

            t_span = torch.linspace(0, 1, n_timesteps + 1, device=text_tokens.device)
            for step in range(n_timesteps):
                t = t_span[step]
                dt = t_span[step + 1] - t
                time_steps = t.expand(batch_size)
                x_proj = self.pre_decoder_proj(x)
                embed_t = einops.rearrange(
                    self.time_step_embedding(time_steps), "bs d -> bs 1 d"
                )
                decoder_out, _ = self.decoder(mu + x_proj + embed_t, mel_lens)
                dphi_dt = self.post_decoder_proj(decoder_out)
                x = x + dt * dphi_dt

            return x

    def get_loss(
        self,
        text_tokens: torch.Tensor,
        text_token_lens: torch.Tensor,
        gt_mels: torch.Tensor,
        gt_mel_lens: torch.Tensor,
    ) -> torch.Tensor:
        decoder_out, gt_mel_lens, gnoise = self.forward(
            text_tokens, text_token_lens, gt_mels, gt_mel_lens
        )
        target = gt_mels - gnoise

        mask = ~lens_to_mask(gt_mel_lens).unsqueeze(-1)  # (batch_size, mel_seq_len, 1)

        loss = self.criterion(decoder_out, target)  # (batch_size, mel_seq_len, dim)
        return (loss * mask).sum() / mask.sum()
