import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor


class HuggingFaceFrontend(nn.Module):
    """Wrap a HuggingFace feature extractor (e.g. Wav2Vec2FeatureExtractor) so
    it behaves like a frontend in the Conformer-style convention.

    For Wav2Vec2-style models (WavLM, HuBERT, Wav2Vec2) the extractor only
    performs per-sample normalization and padding — no STFT — so the
    ``feature`` returned here is just the normalized waveform with feature
    dim = 1. The matching encoder (e.g. ``WavLMEncoder``) squeezes the
    trailing dim before forwarding through the underlying HF model.
    """

    def __init__(self, name: str, sample_rate: int = 16000) -> None:
        super().__init__()
        self.extractor = AutoFeatureExtractor.from_pretrained(name)
        self.sample_rate = sample_rate

    def forward(
        self, wavs: torch.Tensor, wav_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            wavs: (B, T_samples) raw waveforms at ``sample_rate``.
            wav_lens: (B,) number of valid samples per utterance.

        Returns:
            feats: (B, T_max, 1) normalized waveform-as-feature.
            xlens: (B,) sample counts (unchanged).
        """
        wav_list = [
            w[: int(l)].detach().cpu().float().numpy()
            for w, l in zip(wavs, wav_lens.tolist())
        ]
        out = self.extractor(
            wav_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        feats = out.input_values.to(wavs.device)  # (B, T_max)
        return feats.unsqueeze(-1), wav_lens.clone()
