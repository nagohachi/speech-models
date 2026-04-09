import contextlib

import torch
import torch.nn as nn
from transformers import WavLMModel


class WavLMEncoder(nn.Module):
    """Wrap HuggingFace ``WavLMModel`` in the Conformer-style encoder contract.

    Input is ``(B, T_samples, 1)`` from ``HuggingFaceFrontend``
    (waveform-as-feature, normalized by ``Wav2Vec2FeatureExtractor``). The
    trailing feature dim is squeezed before forwarding through WavLM.

    Output is ``(B, T_enc, hidden_size)`` with ``T_enc`` derived from WavLM's
    own conv-stride feature extractor stack.

    The model is frozen by default. Set ``layer`` to an integer for
    SUPERB-style hidden-state probing; ``None`` returns the top of the stack.
    """

    def __init__(
        self,
        name: str = "microsoft/wavlm-large",
        freeze: bool = True,
        layer: int | None = None,
    ) -> None:
        super().__init__()
        self.model = WavLMModel.from_pretrained(name)
        self.hidden_size: int = self.model.config.hidden_size
        self.layer = layer
        self._frozen = freeze
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> "WavLMEncoder":
        super().train(mode)
        if self._frozen:
            self.model.eval()
        return self

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T_samples, 1) normalized waveform-as-feature.
            xlens: (B,) sample counts.

        Returns:
            feats: (B, T_enc, hidden_size) encoder hidden states.
            feat_lens: (B,) valid encoder frame counts.
        """
        wavs = x.squeeze(-1)  # (B, T_samples)
        attn_mask = (
            torch.arange(wavs.size(1), device=wavs.device) < xlens.unsqueeze(1)
        ).long()

        ctx = torch.no_grad() if self._frozen else contextlib.nullcontext()
        with ctx:
            out = self.model(
                input_values=wavs,
                attention_mask=attn_mask,
                output_hidden_states=self.layer is not None,
            )

        if self.layer is not None:
            feats = out.hidden_states[self.layer]
        else:
            feats = out.last_hidden_state

        feat_lens = (
            self.model._get_feat_extract_output_lengths(xlens)
            .long()
            .clamp(max=feats.size(1))
        )
        return feats, feat_lens
