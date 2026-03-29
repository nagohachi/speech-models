import torch
import torch.nn as nn
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
    ) -> None:
        super().__init__()

        n_fft = int(sample_rate * frame_length / 1000)
        hop_length = int(sample_rate * frame_shift / 1000)

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            mel_scale="htk",
        )
        self.eps = 1e-6

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

        melspec = self.mel_spectrogram(wavs)
        log_melspec = torch.log(melspec + self.eps)

        if self.apply_specaug and self.training:
            log_melspec = self.freq_mask(log_melspec)
            log_melspec = self.freq_mask(log_melspec)

            for _ in range(10):
                log_melspec = self.time_mask(log_melspec)

        log_melspec = log_melspec.transpose(1, 2)
        xlens = wav_lens // self.mel_spectrogram.hop_length + 1

        return log_melspec, xlens
