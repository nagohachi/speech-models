import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from speech_models.modules.decoder.resnet1d_unet import ResNet1DUNet
from speech_models.modules.encoder import ConformerEncoder, TransformerEncoder
from speech_models.modules.others.tts.alignment import (
    duration_loss,
    generate_path,
    maximum_path,
)
from speech_models.modules.others.tts.duration_predictor import DurationPredictor
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
        sigma_min: float = 0.0,
        use_prior_loss: bool = False,
        normalize_mel: bool = True,
        mel_stats_path: Path | str | None = None,
        speaker_conditioning: str = "ref_mel",
        num_speakers: int = 0,
        speaker_emb_dim: int = 64,
    ) -> None:
        super().__init__()
        with open(encoder_config_path, "r") as f:
            enc_cfg = yaml.safe_load(f)
            encoder_choice = enc_cfg["encoder"]
            encoder_conf = enc_cfg["encoder_conf"]
        with open(decoder_config_path, "r") as f:
            c = yaml.safe_load(f)
            decoder_choice = c["decoder"]
            decoder_conf = c["decoder_conf"]

        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.sigma_min = sigma_min
        self.use_prior_loss = use_prior_loss
        self.nmels = nmels
        self.speaker_conditioning = speaker_conditioning

        # text encoder
        self.encoder = encoder_choices[encoder_choice](**encoder_conf)
        enc_hidden: int = self.encoder.hidden_size

        # duration predictor (optional)
        self.use_duration_predictor = enc_cfg.get("use_duration_predictor", False)
        if self.use_duration_predictor:
            dp_conf = enc_cfg.get("duration_predictor_conf", {})
            self.duration_predictor = DurationPredictor(
                in_channels=enc_hidden, **dp_conf
            )
            self.proj_mu = nn.Linear(enc_hidden, nmels)

        # speaker embedding (only for speaker_id conditioning)
        if speaker_conditioning == "speaker_id" and num_speakers > 0:
            self.speaker_embedding = nn.Embedding(num_speakers, speaker_emb_dim)
            decoder_conf["spk_emb_dim"] = speaker_emb_dim

        # noise decoder
        self.decoder = decoder_choices[decoder_choice](**decoder_conf)
        self.use_unet = isinstance(self.decoder, ResNet1DUNet)

        # text embedding
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=enc_hidden,
            padding_idx=tokenizer.pad_token_id,
        )

        # projection layers
        if self.use_unet:
            self.post_encoder_proj = nn.Linear(enc_hidden, nmels)
        else:
            raise NotImplementedError()

        # mel normalization
        self.mel_mean: torch.Tensor
        self.mel_std: torch.Tensor
        if normalize_mel and mel_stats_path is not None:
            stats = np.load(mel_stats_path)
            count = stats["count"]
            mean = stats["sum"] / count
            var = stats["sum_square"] / count - mean * mean
            std = np.sqrt(np.maximum(var, 1e-20))
            self.register_buffer("mel_mean", torch.from_numpy(mean.astype(np.float32)))
            self.register_buffer("mel_std", torch.from_numpy(std.astype(np.float32)))
            self._normalize_mel = True
        else:
            self._normalize_mel = False

        self.criterion = nn.MSELoss(reduction="none")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize(self, mel: torch.Tensor) -> torch.Tensor:
        if self._normalize_mel:
            return (mel - self.mel_mean) / self.mel_std
        return mel

    def _denormalize(self, mel: torch.Tensor) -> torch.Tensor:
        if self._normalize_mel:
            return mel * self.mel_std + self.mel_mean
        return mel

    def _validity_mask(
        self, lens: torch.Tensor, max_len: int | None = None
    ) -> torch.Tensor:
        """Float mask (B, T) where 1=valid, 0=pad."""
        if max_len is not None:
            indices = torch.arange(max_len, device=lens.device)
            return (indices.unsqueeze(0) < lens.unsqueeze(1)).float()
        return (~lens_to_mask(lens)).float()

    def _make_mask(
        self, mel_lens: torch.Tensor, max_len: int | None = None
    ) -> torch.Tensor:
        """Float mask (B, 1, T) for UNet conv masking."""
        return self._validity_mask(mel_lens, max_len).unsqueeze(1)

    def _fix_len(self, length: int) -> int:
        assert isinstance(self.decoder, ResNet1DUNet)
        factor = 2**self.decoder.num_downsamplings
        return int((length + factor - 1) // factor * factor)

    def _pad_to_len(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        pad_size = target_len - x.size(1)
        if pad_size > 0:
            return torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        return x

    # ------------------------------------------------------------------
    # Upsampling: uniform or MAS-based
    # ------------------------------------------------------------------

    def upsample(
        self,
        text_encoded: torch.Tensor,
        text_token_lens: torch.Tensor,
        gt_mel_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Uniformly upsample text_encoded to match gt_mel_lens."""
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

    def _align_with_mas(
        self,
        text_encoded: torch.Tensor,
        text_token_lens: torch.Tensor,
        gt_mels: torch.Tensor,
        gt_mel_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align encoder output to mel length using MAS + duration predictor.

        Args:
            text_encoded: (B, T_text, enc_hidden).
            text_token_lens: (B,).
            gt_mels: (B, T_mel, nmels), normalized.
            gt_mel_lens: (B,).

        Returns:
            aligned_enc: (B, T_mel, enc_hidden).
            mu_y_mel: (B, T_mel, nmels) for prior loss.
            dur_loss: scalar.
        """
        # project encoder output to mel space for MAS
        mu_x = self.proj_mu(text_encoded)  # (B, T_text, nmels)

        # masks
        x_mask = self._validity_mask(text_token_lens)  # (B, T_text)
        max_mel_len = int(gt_mel_lens.max().item())
        y_mask = self._validity_mask(gt_mel_lens, max_len=max_mel_len)  # (B, T_mel)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(1)  # (B, T_text, T_mel)

        # MAS: compute log-prior and find optimal alignment
        # mu_x: (B, T_text, nmels), gt_mels: (B, T_mel, nmels)
        # log p(y | mu_x) per (text, mel) pair
        with torch.no_grad():
            mu_x_t = mu_x.transpose(1, 2)  # (B, nmels, T_text)
            y_t = gt_mels.transpose(1, 2)  # (B, nmels, T_mel)
            const = -0.5 * math.log(2 * math.pi) * self.nmels
            factor = -0.5 * torch.ones_like(mu_x_t)
            y_square = torch.matmul(factor.transpose(1, 2), y_t**2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x_t).transpose(1, 2), y_t)
            mu_square = torch.sum(factor * (mu_x_t**2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const  # (B, T_text, T_mel)

            attn = maximum_path(log_prior, attn_mask)  # (B, T_text, T_mel)
            attn = attn.detach()

        # duration predictor loss
        logw_target = torch.log(1e-8 + attn.sum(-1)) * x_mask  # (B, T_text)
        logw = self.duration_predictor(text_encoded.detach(), x_mask)  # (B, T_text)
        d_loss = duration_loss(logw, logw_target, text_token_lens)

        # align encoder output using MAS alignment
        # attn: (B, T_text, T_mel), text_encoded: (B, T_text, enc_hidden)
        aligned_enc = torch.matmul(
            attn.transpose(1, 2), text_encoded
        )  # (B, T_mel, enc_hidden)
        mu_y_mel = torch.matmul(attn.transpose(1, 2), mu_x)  # (B, T_mel, nmels)

        return aligned_enc, mu_y_mel, d_loss

    def _get_aligned_enc(
        self,
        text_encoded: torch.Tensor,
        text_token_lens: torch.Tensor,
        gt_mels: torch.Tensor,
        gt_mel_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Get mel-aligned encoder output. Returns (aligned_enc, mu_y_mel, dur_loss)."""
        if self.use_duration_predictor:
            return self._align_with_mas(
                text_encoded, text_token_lens, gt_mels, gt_mel_lens
            )
        return self.upsample(text_encoded, text_token_lens, gt_mel_lens), None, None

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def _forward_unet(
        self,
        text_tokens: torch.Tensor,
        text_token_lens: torch.Tensor,
        gt_mels: torch.Tensor,
        gt_mel_lens: torch.Tensor,
        ref_mels: torch.Tensor | None = None,
        ref_mel_lens: torch.Tensor | None = None,
        speaker_ids: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        batch_size = gt_mels.size(0)
        s = self.sigma_min

        time_steps = torch.rand(batch_size, device=gt_mels.device)

        text_embeds = self.embedding(text_tokens)
        text_encoded, _ = self.encoder(text_embeds, text_token_lens)

        aligned_enc, mu_y_mel, dur_loss = self._get_aligned_enc(
            text_encoded, text_token_lens, gt_mels, gt_mel_lens
        )

        # Speaker embedding for speaker_id mode
        spk_emb = None
        if self.speaker_conditioning == "speaker_id" and speaker_ids is not None:
            spk_emb = self.speaker_embedding(speaker_ids)  # (B, spk_emb_dim)

        # Determine total length with optional reference mel prefix
        use_ref = self.speaker_conditioning == "ref_mel" and ref_mels is not None and ref_mel_lens is not None
        if use_ref:
            assert ref_mels is not None and ref_mel_lens is not None
            max_ref_len = int(ref_mel_lens.max().item())
            ref_mels_trimmed = ref_mels[:, :max_ref_len, :]
            total_mel_len = max_ref_len + gt_mels.size(1)
            total_mel_lens = ref_mel_lens + gt_mel_lens
        else:
            max_ref_len = 0
            total_mel_len = gt_mels.size(1)
            total_mel_lens = gt_mel_lens

        fixed_len = self._fix_len(total_mel_len)

        # Build mu: [ref_mel | post_encoder_proj(aligned_enc)] -> (B, T_total, nmels)
        mu_target = self.post_encoder_proj(
            self._pad_to_len(aligned_enc, fixed_len - max_ref_len)
        )
        if use_ref:
            mu_ref = self._pad_to_len(ref_mels_trimmed, max_ref_len)
            mu_bt = torch.cat([mu_ref, mu_target], dim=1)[:, :fixed_len, :]
        else:
            mu_bt = mu_target[:, :fixed_len, :]
        mu = mu_bt.transpose(1, 2)  # (B, mel_dim, T_fixed)

        # Build x_t: for ref portion use clean ref_mel, for target portion use noised
        gnoise_target = torch.randn(
            batch_size, fixed_len - max_ref_len, gt_mels.size(2), device=gt_mels.device
        )
        gt_mels_padded = self._pad_to_len(gt_mels, fixed_len - max_ref_len)
        t = time_steps[:, None, None]
        x_t_target = (1 - (1 - s) * t) * gnoise_target + t * gt_mels_padded

        if use_ref:
            ref_padded = self._pad_to_len(ref_mels_trimmed, max_ref_len)
            x_t = torch.cat([ref_padded, x_t_target], dim=1).transpose(1, 2)
        else:
            x_t = x_t_target.transpose(1, 2)

        mask = self._make_mask(total_mel_lens, max_len=fixed_len)
        decoder_out = self.decoder(x_t, mask, mu, time_steps, spk_emb=spk_emb)

        # Extract only target portion for loss computation
        orig_target_len = gt_mels.size(1)
        decoder_out = decoder_out[
            :, :, max_ref_len : max_ref_len + orig_target_len
        ].transpose(1, 2)
        gnoise = gnoise_target[:, :orig_target_len, :]

        # mu for prior loss: use MAS-aligned mel-space mu if available, else post_encoder_proj
        mu_prior = (
            mu_y_mel if mu_y_mel is not None else mu_target[:, :orig_target_len, :]
        )
        return decoder_out, mu_prior, gt_mel_lens, gnoise, dur_loss

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_token_lens: torch.Tensor,
        gt_mels: torch.Tensor,
        gt_mel_lens: torch.Tensor,
        ref_mels: torch.Tensor | None = None,
        ref_mel_lens: torch.Tensor | None = None,
        speaker_ids: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        """Forward path of CFM.

        Returns:
            tuple: (decoder_out, mu_prior, mel_lens, gnoise, dur_loss or None).
        """
        return self._forward_unet(
            text_tokens, text_token_lens, gt_mels, gt_mel_lens, ref_mels, ref_mel_lens,
            speaker_ids=speaker_ids,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict_durations(
        self,
        text_encoded: torch.Tensor,
        text_token_lens: torch.Tensor,
        length_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict mel lengths and aligned encoder output from duration predictor.

        Returns:
            aligned_enc: (B, T_mel, enc_hidden).
            mel_lens: (B,).
        """
        x_mask = self._validity_mask(text_token_lens)  # (B, T_text)
        logw = self.duration_predictor(text_encoded, x_mask)
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        mel_lens = torch.clamp_min(w_ceil.sum(dim=1), 1).long()  # (B,)

        max_mel_len = int(mel_lens.max().item())
        y_mask = self._validity_mask(mel_lens, max_len=max_mel_len)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(1)
        attn = generate_path(w_ceil, attn_mask)
        aligned_enc = torch.matmul(attn.transpose(1, 2), text_encoded)

        return aligned_enc, mel_lens

    @torch.inference_mode()
    def inference_forward(
        self,
        text_tokens: torch.Tensor,
        text_token_lens: torch.Tensor,
        mel_lens: torch.Tensor | None = None,
        n_timesteps: int = 3,
        length_scale: float = 1.0,
        ref_mels: torch.Tensor | None = None,
        ref_mel_lens: torch.Tensor | None = None,
        speaker_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate mel spectrograms from text via Euler ODE solver.

        Args:
            text_tokens: Text token ids (batch_size, seq_len).
            text_token_lens: Text token lengths (batch_size,).
            mel_lens: Target mel lengths (batch_size,). If None, predicted by duration predictor.
            n_timesteps: Number of Euler steps.
            length_scale: Duration scaling factor (inference only).
            ref_mels: Reference mel spectrograms for speaker conditioning (batch_size, ref_mel_len, nmels).
            ref_mel_lens: Reference mel lengths (batch_size,).
            speaker_ids: Speaker ID indices for speaker_id conditioning (batch_size,).

        Returns:
            Tuple of (generated_mels, mel_lens).
            generated_mels: (batch_size, max_mel_len, mel_dim).
            mel_lens: (batch_size,).
        """
        text_embeds = self.embedding(text_tokens)
        text_encoded, _ = self.encoder(text_embeds, text_token_lens)
        batch_size = text_tokens.size(0)

        # get aligned encoder output
        if self.use_duration_predictor and mel_lens is None:
            aligned_enc, mel_lens = self._predict_durations(
                text_encoded, text_token_lens, length_scale
            )
        else:
            assert mel_lens is not None
            aligned_enc = self.upsample(text_encoded, text_token_lens, mel_lens)

        max_mel_len = int(mel_lens.max().item())
        assert isinstance(self.decoder, ResNet1DUNet)
        nmels = self.decoder.out_channels

        # Speaker embedding for speaker_id mode
        spk_emb = None
        if self.speaker_conditioning == "speaker_id" and speaker_ids is not None:
            spk_emb = self.speaker_embedding(speaker_ids)

        use_ref = self.speaker_conditioning == "ref_mel" and ref_mels is not None and ref_mel_lens is not None
        if use_ref:
            assert ref_mels is not None and ref_mel_lens is not None
            ref_mels = self._normalize(ref_mels)
            max_ref_len = int(ref_mel_lens.max().item())
            ref_mels_trimmed = ref_mels[:, :max_ref_len, :]
            total_len = max_ref_len + max_mel_len
            total_lens = ref_mel_lens + mel_lens
        else:
            max_ref_len = 0
            total_len = max_mel_len
            total_lens = mel_lens

        fixed_len = self._fix_len(total_len)

        # Build mu
        mu_target = self.post_encoder_proj(
            self._pad_to_len(aligned_enc, fixed_len - max_ref_len)
        )
        if use_ref:
            mu_ref = self._pad_to_len(ref_mels_trimmed, max_ref_len)
            mu = torch.cat([mu_ref, mu_target], dim=1)[:, :fixed_len, :].transpose(1, 2)
        else:
            mu = mu_target[:, :fixed_len, :].transpose(1, 2)

        mask = self._make_mask(total_lens, max_len=fixed_len)

        # Initialize: ref portion is clean, target portion is noise
        if use_ref:
            ref_padded = self._pad_to_len(ref_mels_trimmed, max_ref_len)
            noise_target = torch.randn(
                batch_size, nmels, fixed_len - max_ref_len, device=text_tokens.device
            )
            x = torch.cat([ref_padded.transpose(1, 2), noise_target], dim=2)
        else:
            x = torch.randn(batch_size, nmels, fixed_len, device=text_tokens.device)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=text_tokens.device)
        for step in range(n_timesteps):
            t = t_span[step]
            dt = t_span[step + 1] - t
            dphi_dt = self.decoder(x, mask, mu, t.expand(batch_size), spk_emb=spk_emb)
            if use_ref:
                # Only update target portion; keep reference portion unchanged
                x[:, :, max_ref_len:] = (
                    x[:, :, max_ref_len:] + dt * dphi_dt[:, :, max_ref_len:]
                )
            else:
                x = x + dt * dphi_dt

        # Extract target portion
        target_out = x[:, :, max_ref_len : max_ref_len + max_mel_len].transpose(1, 2)
        return self._denormalize(target_out), mel_lens

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def get_loss(
        self,
        text_tokens: torch.Tensor,
        text_token_lens: torch.Tensor,
        gt_mels: torch.Tensor,
        gt_mel_lens: torch.Tensor,
        ref_mels: torch.Tensor | None = None,
        ref_mel_lens: torch.Tensor | None = None,
        speaker_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        gt_mels = self._normalize(gt_mels)
        if ref_mels is not None:
            ref_mels = self._normalize(ref_mels)

        decoder_out, mu_prior, gt_mel_lens, gnoise, dur_loss = self.forward(
            text_tokens, text_token_lens, gt_mels, gt_mel_lens, ref_mels, ref_mel_lens,
            speaker_ids=speaker_ids,
        )
        s = self.sigma_min
        target = gt_mels - (1 - s) * gnoise

        mask = ~lens_to_mask(gt_mel_lens).unsqueeze(-1)  # (B, T, 1)

        diff_loss = self.criterion(decoder_out, target)
        n_feats = gt_mels.size(-1)
        diff_loss = (diff_loss * mask).sum() / (mask.sum() * n_feats)

        losses: dict[str, torch.Tensor] = {"diff_loss": diff_loss}

        if dur_loss is not None:
            losses["dur_loss"] = dur_loss

        if self.use_prior_loss:
            prior = 0.5 * ((gt_mels - mu_prior) ** 2 + math.log(2 * math.pi))
            prior_loss = (prior * mask).sum() / (mask.sum() * n_feats)
            losses["prior_loss"] = prior_loss

        return losses
