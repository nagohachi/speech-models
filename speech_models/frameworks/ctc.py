from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import yaml
from speech_models.modules.encoder.conformer.conformer_encoder import ConformerEncoder
from speech_models.modules.frontend.log_mel import BatchedFbank
from speech_models.tokenizers.bpe_tokenizer import BPETokenizer

frontend_choices = dict(batched_fbank=BatchedFbank)
encoder_choices = dict(conformer=ConformerEncoder)


class CTCBasedASR(nn.Module):
    def __init__(
        self,
        frontend_config_path: Path | str,
        encoder_config_path: Path | str,
        tokenizer: BPETokenizer,
    ) -> None:
        super().__init__()

        with open(frontend_config_path, "r") as f:
            c = yaml.safe_load(f)
            frontend_choice = c["frontend"]
            frontend_conf = c["frontend_conf"]
        with open(encoder_config_path, "r") as f:
            c = yaml.safe_load(f)
            encoder_choice = c["encoder"]
            encoder_conf = c["encoder_conf"]

        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        self.frontend = frontend_choices[frontend_choice](**frontend_conf)
        self.encoder = encoder_choices[encoder_choice](**encoder_conf)
        self.ctc_linear = nn.Linear(encoder_conf["hidden_size"], self.vocab_size)

        self.criterion = nn.CTCLoss(
            blank=self.tokenizer.blank_token_id, zero_infinity=True
        )

    def forward(
        self, wavs: torch.Tensor, wav_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            wavs: (Batch, Time)
            wav_lens: (Batch,)
        Returns:
            log_probs: (Time, Batch, Vocab)
            xlens: (Batch,)
        """
        x, xlens = self.frontend(wavs, wav_lens)
        x, xlens = self.encoder(x, xlens)
        logits = self.ctc_linear(x)
        log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)

        return log_probs, xlens

    def _get_loss(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.criterion(log_probs, targets, input_lengths, target_lengths)

    def get_loss(
        self,
        wavs: torch.Tensor,
        wav_lens: torch.Tensor,
        label_tokens: torch.Tensor,
        label_token_lens: torch.Tensor,
    ) -> torch.Tensor:
        log_probs, xlens = self.forward(wavs, wav_lens)
        return self._get_loss(log_probs, label_tokens, xlens, label_token_lens)

    def inference_forward(
        self,
        wavs: torch.Tensor,
        wav_lens: torch.Tensor,
        inference_algorithm: Literal["greedy_search", "beam_search"] = "greedy_search",
    ) -> list[str]:
        # if inference_algorithm != "greedy_search":
        #     raise NotImplementedError()

        log_probs, xlens = self.forward(wavs, wav_lens)
        preds = log_probs.argmax(dim=-1).transpose(0, 1)
        hyp_tokens = [self.tokenizer.ctc_collapse(pred) for pred in preds]

        return [self.tokenizer.decode(line) for line in hyp_tokens]
