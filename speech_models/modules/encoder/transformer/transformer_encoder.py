import torch
import torch.nn as nn
from speech_models.modules.utils.mask import lens_to_mask


class TransformerEncoder(nn.Module):
    """Generic transformer encoder.

    NOTE: This encoder does NOT add positional encoding. The caller is
    responsible for adding PE to the input embedding (or for using a
    PE-equivariant downstream loss). Keeping PE out of the encoder lets
    different consumers pick their own scheme (sinusoidal, learned, RoPE, etc.)
    or none at all.
    """

    # Class-level hints for `CFMbasedModel` (and other consumers) about how to
    # prepare the input embedding stream. Defaults match the historical
    # behavior so this encoder is unchanged for ASR / speech-LLM consumers.
    WANTS_EMBEDDING_SCALE: bool = False
    WANTS_ABSOLUTE_PE: bool = True
    EMBEDDING_INIT_STYLE: str = "default"
    BUNDLES_SPK_CONCAT: bool = False

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        if dim_feedforward is None:
            dim_feedforward = 4 * hidden_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        x: torch.Tensor,
        xlens: torch.Tensor,
        spk_emb: torch.Tensor | None = None,  # noqa: ARG002 - kept for API parity
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = lens_to_mask(xlens)
        return self.transformer_encoder(x, src_key_padding_mask=mask), xlens
