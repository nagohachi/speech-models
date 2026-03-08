import torch
import torch.nn as nn
import torchaudio


class BatchedFbank(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        frame_length: float = 25.0,  # ms
        frame_shift: float = 10.0,  # ms
    ) -> None:
        super().__init__()

        n_fft = int(sample_rate * frame_length / 1000)
        hop_length = int(sample_rate * frame_shift / 1000)

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            f_min=20.0,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            mel_scale="htk",
        )
        self.eps = 1e-6

    def forward(
        self, wavs: torch.Tensor, wav_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            wavs (torch.Tensor): 音声波形 (batch_size, max_num_samples)
            wav_lens (torch.Tensor): 各音声の実際のサンプル数 (batch_size,)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - mel spectrogram (batch_size, num_frames, num_mel_bins)
                - frame length (batch_size,)
        """
        melspec = self.mel_spectrogram(wavs)

        log_melspec = torch.log(melspec + self.eps)

        log_melspec = log_melspec.transpose(1, 2)

        xlens = wav_lens // self.mel_spectrogram.hop_length + 1

        return log_melspec, xlens
