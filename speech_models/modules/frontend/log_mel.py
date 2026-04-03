import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class BatchedFbank(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        frame_length: float = 25.0,
        frame_shift: float = 10.0,
        f_min: float = 20.0,
        f_max: float | None = None,
        apply_specaug: bool = True,
        specaug_freq_mask_param: int = 27,
        specaug_time_mask_param: int = 10000,
        specaug_time_prob: float = 0.05,
        power: float = 2.0,
        center: bool = True,
        mel_scale: str = "htk",
        mel_norm: str | None = None,
        log_eps: float = 1e-6,
        log_clamp: bool = False,
        stft_pad: int | None = None,
    ) -> None:
        super().__init__()

        n_fft = int(sample_rate * frame_length / 1000)
        hop_length = int(sample_rate * frame_shift / 1000)

        if stft_pad is not None:
            self._stft_pad = stft_pad
            actual_center = False
        elif not center:
            self._stft_pad = (n_fft - hop_length) // 2
            actual_center = False
        else:
            self._stft_pad = 0
            actual_center = True

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            mel_scale=mel_scale,
            norm=mel_norm,
            power=power,
            center=actual_center,
        )
        self._log_eps = log_eps
        self._log_clamp = log_clamp
        self._center = actual_center
        self._n_fft = n_fft
        self._hop_length = hop_length

        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=specaug_freq_mask_param, iid_masks=True
        )

        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=specaug_time_mask_param, iid_masks=True, p=specaug_time_prob
        )

        self.apply_specaug = apply_specaug

    def forward(
        self, wavs: torch.Tensor, wav_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self._stft_pad > 0:
            wavs = F.pad(
                wavs.unsqueeze(1), (self._stft_pad, self._stft_pad), mode="reflect"
            ).squeeze(1)

        melspec = self.mel_spectrogram(wavs)

        if self._log_clamp:
            log_melspec = torch.log(torch.clamp(melspec, min=self._log_eps))
        else:
            log_melspec = torch.log(melspec + self._log_eps)

        if self.apply_specaug and self.training:
            log_melspec = self.freq_mask(log_melspec)
            log_melspec = self.freq_mask(log_melspec)

            for _ in range(10):
                log_melspec = self.time_mask(log_melspec)

        log_melspec = log_melspec.transpose(1, 2)

        if self._center:
            xlens = wav_lens // self._hop_length + 1
        else:
            xlens = (wav_lens + 2 * self._stft_pad - self._n_fft) // self._hop_length + 1

        return log_melspec, xlens
