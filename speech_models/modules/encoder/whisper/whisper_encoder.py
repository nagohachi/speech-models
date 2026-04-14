import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
import whisper.audio as wa


class WhisperEncoder(nn.Module):
    """OpenAI Whisper audio encoder, plugged into the Conformer-style encoder
    contract.

    Input is ``(B, T, n_mels)`` variable-length log-mel produced by
    ``WhisperFrontend``. The 3000-frame pad/trim that the underlying Whisper
    encoder requires (Whisper uses fixed absolute positional embeddings) is
    applied here, not in the frontend, so the frontend stays consumer-agnostic.

    The encoder is frozen by default.
    """

    N_FRAMES: int = wa.N_FRAMES  # 3000

    def __init__(self, name: str = "medium", freeze: bool = True) -> None:
        super().__init__()
        whisper_model = whisper.load_model(name)
        self.encoder = whisper_model.encoder
        self.hidden_size: int = whisper_model.dims.n_audio_state
        self._frozen = freeze
        if freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> "WhisperEncoder":
        super().train(mode)
        if self._frozen:
            self.encoder.eval()
        return self

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, n_mels) log-mel spectrogram from WhisperFrontend.
            xlens: (B,) valid mel frame counts.

        Returns:
            feats: (B, T_enc, hidden_size) encoder hidden states.
            feat_lens: (B,) valid encoder frame counts.
        """
        # (B, T, n_mels) -> (B, n_mels, T) for whisper.encoder
        mels = x.transpose(1, 2)

        # Pad or trim to exactly N_FRAMES (Whisper has fixed positional embeds)
        T = mels.size(-1)
        if T < self.N_FRAMES:
            mels = F.pad(mels, (0, self.N_FRAMES - T))
        else:
            mels = mels[..., : self.N_FRAMES]

        if self._frozen:
            with torch.no_grad():
                feats = self.encoder(mels)  # (B, 1500, hidden_size)
        else:
            feats = self.encoder(mels)

        # Whisper's conv stem downsamples mel frames by 2
        feat_lens = ((xlens + 1) // 2).long().clamp(max=feats.size(1))
        return feats, feat_lens
