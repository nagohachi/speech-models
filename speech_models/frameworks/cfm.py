from pathlib import Path

import einops
import torch
import torch.nn as nn
import yaml
from speech_models.modules.encoder import ConformerEncoder, TransformerEncoder
from speech_models.modules.others.tts.time_step_embedding import TimeStepEmbedding
from speech_models.modules.utils.mask import lens_to_mask
from speech_models.tokenizers import BPETokenizer, CharTokenizer

encoder_choices = dict(transformer=TransformerEncoder)
decoder_choices = dict(conformer=ConformerEncoder)


class CFMbasedModel(nn.Module):
    """Conditional Flow Matching-based Text-to-Speech model."""

    def __init__(
        self,
        frontend_config_path: Path | str,
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
        # noise decoder
        self.decoder = decoder_choices[decoder_choice](**decoder_conf)

        self.post_encoder_proj = nn.Linear(
            self.encoder.hidden_size, self.decoder.hidden_size
        )
        self.pre_decoder_proj = nn.Linear(nmels, self.decoder.hidden_size)
        self.post_decoder_proj = nn.Linear(self.decoder.hidden_size, nmels)

        # text embedding
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.encoder.hidden_size,
            padding_idx=tokenizer.pad_token_id,
        )

        self.time_step_embedding = TimeStepEmbedding(
            hidden_size=self.decoder.hidden_size, output_size=self.decoder.hidden_size
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

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_token_lens: torch.Tensor,
        gt_mels: torch.Tensor,
        gt_mel_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward path of cfm.

        Args:
            text_tokens (torch.Tensor): text token ids of shape (batch_size, ).
            text_token_lens (torch.Tensor): text token lengths of shape (batch_size, ).
            gt_mels (torch.Tensor): ground truth mel spectrograms of shape (batch_size, seq_len, mel_dim).
            gt_mel_lens (torch.Tensor): ground truth mel spectrogram lengths of shape (batch_size, ).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - u_t(x_t, t, mu): it models x_1 - x_0 (mel of gt - gaussian noise).
            - mel spectrogram lengths.
            - gaussian noise, which is used for generation.
        """
        batch_size = gt_mels.size(0)

        # randomly sampled t from [0, 1)
        time_steps = torch.rand(batch_size, device=gt_mels.device)

        # gaussian noise
        gnoise = torch.randn_like(gt_mels)

        text_embeds = self.embedding(text_tokens)
        text_encoded, _ = self.encoder(
            text_embeds, text_token_lens
        )  # (batch_size, seq_len, enc_hid)

        # encoded text, works as \mu
        text_encoded_upsampled = self.post_encoder_proj(
            self.upsample(text_encoded, text_token_lens, gt_mel_lens)
        )  # (batch_size, mel_seq_len, decoder_hid)

        # internally divided mel, works as x_t
        time_steps_unsq = time_steps[:, None, None]
        mel_t = self.pre_decoder_proj(
            (1 - time_steps_unsq) * gnoise + time_steps_unsq * gt_mels
        )  # (batch_size, seq_len, decoder_hid)

        # time embedding
        embed_t = einops.rearrange(
            self.time_step_embedding(time_steps), "bs dec_hid -> bs 1 dec_hid"
        )  # (batch_size, 1, decoder_hid)

        # simply added together and input to decoder
        decoder_out, _ = self.decoder(
            text_encoded_upsampled + mel_t + embed_t, gt_mel_lens
        )
        return self.post_decoder_proj(decoder_out), gt_mel_lens, gnoise

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

        mu = self.post_encoder_proj(
            self.upsample(text_encoded, text_token_lens, mel_lens)
        )  # (batch_size, max_mel_len, decoder_hid)

        max_mel_len = int(mel_lens.max().item())
        nmels = self.pre_decoder_proj.in_features
        x = torch.randn(
            text_tokens.size(0), max_mel_len, nmels, device=text_tokens.device
        )

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=text_tokens.device)

        for step in range(n_timesteps):
            t = t_span[step]
            dt = t_span[step + 1] - t

            time_steps = t.expand(text_tokens.size(0))
            x_proj = self.pre_decoder_proj(x)
            embed_t = einops.rearrange(
                self.time_step_embedding(time_steps), "bs dec_hid -> bs 1 dec_hid"
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
