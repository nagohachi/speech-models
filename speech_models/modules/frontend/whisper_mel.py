import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import whisper.audio as wa


class WhisperFrontend(nn.Module):
    """Compute Whisper-compatible log-mel spectrograms in a batched manner.

    Uses the same STFT parameters and normalization as ``openai-whisper`` but
    operates on padded mini-batches without a Python loop.  The output is zero-
    padded to ``N_FRAMES`` (3000) so that it can be fed directly to the Whisper
    encoder.

    Optional SpecAugment (frequency and time masking) is applied during training.
    """

    N_FFT: int = wa.N_FFT  # 400
    HOP_LENGTH: int = wa.HOP_LENGTH  # 160
    N_FRAMES: int = wa.N_FRAMES  # 3000

    def __init__(
        self,
        n_mels: int = 80,
        apply_specaug: bool = True,
        specaug_freq_mask_param: int = 27,
        specaug_time_mask_param: int = 10000,
        specaug_time_prob: float = 0.05,
    ) -> None:
        super().__init__()
        assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
        self.n_mels = n_mels
        self.apply_specaug = apply_specaug
        # mel filter bank from whisper assets (non-trainable)
        self.register_buffer("mel_filters", wa.mel_filters("cpu", n_mels))

        if apply_specaug:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=specaug_freq_mask_param, iid_masks=True
            )
            self.time_mask = torchaudio.transforms.TimeMasking(
                time_mask_param=specaug_time_mask_param,
                iid_masks=True,
                p=specaug_time_prob,
            )

    def forward(
        self, wavs: torch.Tensor, wav_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            wavs: (B, samples) audio at 16 kHz.
            wav_lens: (B,) number of valid samples per utterance.

        Returns:
            mels: (B, n_mels, N_FRAMES) padded log-mel spectrograms.
            mel_lens: (B,) number of valid mel frames before padding.
        """
        window = torch.hann_window(self.N_FFT, device=wavs.device)
        stft = torch.stft(
            wavs, self.N_FFT, self.HOP_LENGTH, window=window, return_complex=True
        )
        magnitudes = stft[..., :-1].abs() ** 2  # (B, n_fft//2+1, T)

        mel_spec = self.mel_filters @ magnitudes  # (B, n_mels, T)

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        # per-sample max normalization (whisper uses global max for single sample)
        max_val = log_spec.amax(dim=(-2, -1), keepdim=True)
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        # SpecAugment (training only)
        if self.apply_specaug and self.training:
            log_spec = self.freq_mask(log_spec)
            log_spec = self.freq_mask(log_spec)
            for _ in range(10):
                log_spec = self.time_mask(log_spec)

        # mel frame count per utterance
        mel_lens = (wav_lens / self.HOP_LENGTH).long() + 1
        mel_lens = mel_lens.clamp(max=log_spec.size(-1))

        # pad or trim to N_FRAMES
        T = log_spec.size(-1)
        if T < self.N_FRAMES:
            log_spec = F.pad(log_spec, (0, self.N_FRAMES - T))
        else:
            log_spec = log_spec[..., : self.N_FRAMES]
        mel_lens = mel_lens.clamp(max=self.N_FRAMES)

        return log_spec, mel_lens
