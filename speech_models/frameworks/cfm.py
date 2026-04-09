import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from speech_models.modules.decoder.resnet1d_unet import ResNet1DUNet
from speech_models.modules.encoder import (
    ConformerEncoder,
    GlowTTSEncoder,
    TransformerEncoder,
)
from speech_models.modules.others.tts.alignment import (
    duration_loss,
    generate_path,
    maximum_path,
)
from speech_models.modules.others.tts.duration_predictor import DurationPredictor
from speech_models.modules.utils.mask import lens_to_mask
from speech_models.modules.utils.positional_encoding import SinusoidalPositionalEncoding
from speech_models.tokenizers import BPETokenizer, CharTokenizer

encoder_choices = dict(transformer=TransformerEncoder, glow_tts=GlowTTSEncoder)
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
            encoder_conf = dict(enc_cfg["encoder_conf"])
        with open(decoder_config_path, "r") as f:
            c = yaml.safe_load(f)
            decoder_choice = c["decoder"]
            decoder_conf = dict(c["decoder_conf"])

        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.sigma_min = sigma_min
        self.use_prior_loss = use_prior_loss
        self.nmels = nmels
        self.speaker_conditioning = speaker_conditioning

        # text embedding dim before any speaker bump (Matcha-TTS style: encoder
        # operates at text_emb_dim + spk_emb_dim, but the text token embedding
        # itself stays at text_emb_dim and speaker info is concat-broadcast)
        text_emb_dim: int = encoder_conf["hidden_size"]
        self.text_emb_dim = text_emb_dim

        # Look up the encoder class so we can read its preferences BEFORE
        # constructing it. The class constants tell us:
        #   - whether to scale the embedding by sqrt(d) at forward
        #   - whether to add absolute sinusoidal PE
        #   - how to initialize the embedding weight
        #   - whether the encoder bundles its own speaker concat (so we should
        #     pass spk_emb via kwarg instead of pre-concatenating)
        encoder_class = encoder_choices[encoder_choice]
        self._scale_embedding: bool = getattr(encoder_class, "WANTS_EMBEDDING_SCALE", False)
        self._add_absolute_pe: bool = getattr(encoder_class, "WANTS_ABSOLUTE_PE", True)
        self._embedding_init_style: str = getattr(
            encoder_class, "EMBEDDING_INIT_STYLE", "default"
        )
        self._encoder_handles_spk_concat: bool = getattr(
            encoder_class, "BUNDLES_SPK_CONCAT", False
        )

        # speaker embedding (only for speaker_id conditioning)
        # NOTE: created BEFORE the encoder so we can bump the encoder hidden_size
        # for the legacy (non-bundling) encoders.
        if speaker_conditioning == "speaker_id" and num_speakers > 0:
            self.speaker_embedding = nn.Embedding(num_speakers, speaker_emb_dim)
            decoder_conf["spk_emb_dim"] = speaker_emb_dim
            if self._encoder_handles_spk_concat:
                # Encoder accepts spk_emb_dim via its constructor and applies
                # the concat at the right point internally; do NOT bump
                # hidden_size on the way in.
                encoder_conf["spk_emb_dim"] = speaker_emb_dim
                effective_hidden = text_emb_dim + speaker_emb_dim
            else:
                # Legacy path: framework pre-concatenates speaker to the text
                # stream before calling the encoder, so the encoder's
                # `hidden_size` parameter must include the speaker dim.
                encoder_conf["hidden_size"] = text_emb_dim + speaker_emb_dim
                effective_hidden = encoder_conf["hidden_size"]
            num_heads = encoder_conf.get("num_heads")
            if num_heads is not None and effective_hidden % num_heads != 0:
                raise ValueError(
                    f"effective encoder hidden_size ({effective_hidden} = "
                    f"{text_emb_dim} + {speaker_emb_dim}) must be divisible by "
                    f"num_heads ({num_heads}) for speaker_id conditioning"
                )

        # text encoder (operates at text_emb_dim + spk_emb_dim if speaker_id mode)
        self.encoder = encoder_class(**encoder_conf)
        enc_hidden: int = self.encoder.hidden_size

        # duration predictor (optional). in_channels is enc_hidden so it is
        # automatically speaker-aware in speaker_id mode.
        self.use_duration_predictor = enc_cfg.get("use_duration_predictor", False)
        if self.use_duration_predictor:
            dp_conf = enc_cfg.get("duration_predictor_conf", {})
            self.duration_predictor = DurationPredictor(
                in_channels=enc_hidden, **dp_conf
            )

        # noise decoder
        self.decoder = decoder_choices[decoder_choice](**decoder_conf)
        self.use_unet = isinstance(self.decoder, ResNet1DUNet)

        # text embedding (kept at text_emb_dim; speaker info is concat-broadcast
        # to its output before being fed to the encoder)
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=text_emb_dim,
            padding_idx=tokenizer.pad_token_id,
        )
        if self._embedding_init_style == "normal_scaled":
            # Vaswani / Glow-TTS recipe: small embeddings paired with the
            # `× sqrt(d)` scale at forward (see _forward_unet / inference_forward).
            nn.init.normal_(self.embedding.weight, 0.0, text_emb_dim**-0.5)
            with torch.no_grad():
                self.embedding.weight[tokenizer.pad_token_id].zero_()

        # Sinusoidal absolute PE — only used when the encoder asks for it via
        # WANTS_ABSOLUTE_PE. Always constructed (cheap; just a buffer) so the
        # forward path stays simple.
        self.text_positional_encoding = SinusoidalPositionalEncoding(
            hidden_size=text_emb_dim, dropout_prob=0.0
        )

        # Single shared encoder->mel projection (Matcha-TTS style).
        # Used by MAS log-prior, prior loss, AND decoder mu conditioning so
        # the same tensor is rectified by every loss path. Init follows the
        # same pattern as the decoder Linear layers (kaiming_normal weight,
        # zero bias) for parity.
        if self.use_unet:
            self.proj_m = nn.Linear(enc_hidden, nmels)
            nn.init.kaiming_normal_(self.proj_m.weight, nonlinearity="relu")
            nn.init.zeros_(self.proj_m.bias)
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
    # Speaker conditioning helpers (Matcha-TTS style: channel concat)
    # ------------------------------------------------------------------

    def _lookup_spk_emb(self, speaker_ids: torch.Tensor | None) -> torch.Tensor | None:
        """Look up dense speaker embedding from ids.

        Returns None when not in speaker_id mode or no speaker_embedding exists.
        """
        if (
            self.speaker_conditioning != "speaker_id"
            or speaker_ids is None
            or not hasattr(self, "speaker_embedding")
        ):
            return None
        return self.speaker_embedding(speaker_ids)  # (B, spk_emb_dim)

    def _concat_spk_to_text_embeds(
        self, text_embeds: torch.Tensor, spk_emb: torch.Tensor | None
    ) -> torch.Tensor:
        """Broadcast speaker embedding along time and concat to text embeddings.

        Args:
            text_embeds: (B, T_text, text_emb_dim).
            spk_emb: (B, spk_emb_dim) or None.

        Returns:
            (B, T_text, text_emb_dim + spk_emb_dim) if spk_emb is not None,
            else text_embeds unchanged.
        """
        if spk_emb is None:
            return text_embeds
        spk_t = spk_emb.unsqueeze(1).expand(-1, text_embeds.size(1), -1)
        return torch.cat([text_embeds, spk_t], dim=-1)

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
        mu_x: torch.Tensor,
        text_encoded: torch.Tensor,
        text_token_lens: torch.Tensor,
        gt_mels: torch.Tensor,
        gt_mel_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align mel-space encoder output to mel length using MAS + duration predictor.

        Args:
            mu_x: (B, T_text, nmels), encoder output projected to mel space via proj_m.
            text_encoded: (B, T_text, enc_hidden), used as duration predictor input only.
            text_token_lens: (B,).
            gt_mels: (B, T_mel, nmels), normalized.
            gt_mel_lens: (B,).

        Returns:
            aligned_mu: (B, T_mel, nmels) — same tensor used for both decoder mu
                conditioning and prior loss (Matcha-TTS style).
            dur_loss: scalar.
        """
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

        # duration predictor loss (DP receives full encoder hidden, detached)
        logw_target = torch.log(1e-8 + attn.sum(-1)) * x_mask  # (B, T_text)
        logw = self.duration_predictor(text_encoded.detach(), x_mask)  # (B, T_text)
        d_loss = duration_loss(logw, logw_target, text_token_lens)

        # align mel-space encoder output using MAS alignment
        aligned_mu = torch.matmul(attn.transpose(1, 2), mu_x)  # (B, T_mel, nmels)

        return aligned_mu, d_loss

    def _get_aligned_mu(
        self,
        mu_x: torch.Tensor,
        text_encoded: torch.Tensor,
        text_token_lens: torch.Tensor,
        gt_mels: torch.Tensor,
        gt_mel_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get mel-aligned mel-space encoder output. Returns (aligned_mu, dur_loss).

        aligned_mu: (B, T_mel, nmels) — used for both decoder mu input and prior loss.
        """
        if self.use_duration_predictor:
            return self._align_with_mas(
                mu_x, text_encoded, text_token_lens, gt_mels, gt_mel_lens
            )
        return self.upsample(mu_x, text_token_lens, gt_mel_lens), None

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

        # Speaker embedding (looked up once, used for both encoder and decoder)
        spk_emb = self._lookup_spk_emb(speaker_ids)  # (B, spk_emb_dim) or None

        text_embeds = self.embedding(text_tokens)
        if self._scale_embedding:
            text_embeds = text_embeds * math.sqrt(self.text_emb_dim)
        if self._add_absolute_pe:
            text_embeds = self.text_positional_encoding(text_embeds)
        if not self._encoder_handles_spk_concat:
            text_embeds = self._concat_spk_to_text_embeds(text_embeds, spk_emb)
        text_encoded, _ = self.encoder(
            text_embeds,
            text_token_lens,
            spk_emb=spk_emb if self._encoder_handles_spk_concat else None,
        )

        # Project encoder output to mel space ONCE. This single tensor is used
        # for MAS, prior loss, and decoder mu conditioning (Matcha-TTS style).
        # Mask padded text positions so they cannot leak into the MAS log-prior.
        mu_x = self.proj_m(text_encoded)  # (B, T_text, nmels)
        mu_x = mu_x * self._validity_mask(text_token_lens).unsqueeze(-1)

        aligned_mu, dur_loss = self._get_aligned_mu(
            mu_x, text_encoded, text_token_lens, gt_mels, gt_mel_lens
        )

        # Determine total length with optional reference mel prefix
        use_ref = (
            self.speaker_conditioning == "ref_mel"
            and ref_mels is not None
            and ref_mel_lens is not None
        )
        ref_mels_trimmed: torch.Tensor | None = None
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

        # Build mu: [ref_mel | aligned_mu] -> (B, T_total, nmels). aligned_mu is
        # already in mel space, so no extra projection is needed.
        mu_target = self._pad_to_len(aligned_mu, fixed_len - max_ref_len)
        if use_ref:
            assert ref_mels_trimmed is not None
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
            assert ref_mels_trimmed is not None
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

        # mu for prior loss is exactly the (unpadded) target portion of the mu
        # the decoder consumed -- single tensor, single source of truth.
        mu_prior = mu_target[:, :orig_target_len, :]
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
            text_tokens,
            text_token_lens,
            gt_mels,
            gt_mel_lens,
            ref_mels,
            ref_mel_lens,
            speaker_ids=speaker_ids,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict_durations(
        self,
        mu_x: torch.Tensor,
        text_encoded: torch.Tensor,
        text_token_lens: torch.Tensor,
        length_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict mel lengths and aligned mel-space encoder output via DP.

        Args:
            mu_x: (B, T_text, nmels), encoder output projected to mel space via proj_m.
            text_encoded: (B, T_text, enc_hidden), used as duration predictor input only.

        Returns:
            aligned_mu: (B, T_mel, nmels).
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
        aligned_mu = torch.matmul(attn.transpose(1, 2), mu_x)

        return aligned_mu, mel_lens

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
        # Speaker embedding (looked up once, used for both encoder and decoder)
        spk_emb = self._lookup_spk_emb(speaker_ids)  # (B, spk_emb_dim) or None

        text_embeds = self.embedding(text_tokens)
        if self._scale_embedding:
            text_embeds = text_embeds * math.sqrt(self.text_emb_dim)
        if self._add_absolute_pe:
            text_embeds = self.text_positional_encoding(text_embeds)
        if not self._encoder_handles_spk_concat:
            text_embeds = self._concat_spk_to_text_embeds(text_embeds, spk_emb)
        text_encoded, _ = self.encoder(
            text_embeds,
            text_token_lens,
            spk_emb=spk_emb if self._encoder_handles_spk_concat else None,
        )
        batch_size = text_tokens.size(0)

        # Project encoder output to mel space ONCE (Matcha-TTS style).
        # Mask padded text positions so they cannot leak into duration prediction
        # or the upsampling matmul.
        mu_x = self.proj_m(text_encoded)  # (B, T_text, nmels)
        mu_x = mu_x * self._validity_mask(text_token_lens).unsqueeze(-1)

        # get aligned mel-space encoder output
        if self.use_duration_predictor and mel_lens is None:
            aligned_mu, mel_lens = self._predict_durations(
                mu_x, text_encoded, text_token_lens, length_scale
            )
        else:
            assert mel_lens is not None
            aligned_mu = self.upsample(mu_x, text_token_lens, mel_lens)

        max_mel_len = int(mel_lens.max().item())
        assert isinstance(self.decoder, ResNet1DUNet)
        nmels = self.decoder.out_channels

        use_ref = (
            self.speaker_conditioning == "ref_mel"
            and ref_mels is not None
            and ref_mel_lens is not None
        )
        ref_mels_trimmed: torch.Tensor | None = None
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

        # Build mu (aligned_mu is already in mel space; no extra projection needed)
        mu_target = self._pad_to_len(aligned_mu, fixed_len - max_ref_len)
        if use_ref:
            assert ref_mels_trimmed is not None
            mu_ref = self._pad_to_len(ref_mels_trimmed, max_ref_len)
            mu = torch.cat([mu_ref, mu_target], dim=1)[:, :fixed_len, :].transpose(1, 2)
        else:
            mu = mu_target[:, :fixed_len, :].transpose(1, 2)

        mask = self._make_mask(total_lens, max_len=fixed_len)

        # Initialize: ref portion is clean, target portion is noise
        if use_ref:
            assert ref_mels_trimmed is not None
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
            text_tokens,
            text_token_lens,
            gt_mels,
            gt_mel_lens,
            ref_mels,
            ref_mel_lens,
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
