from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from speech_models.modules.decoder.rnn import RNNDecoder
from speech_models.modules.encoder.conformer.conformer_encoder import ConformerEncoder
from speech_models.modules.frontend.log_mel import BatchedFbank
from speech_models.modules.others.rnnt.joiner import Joiner
from speech_models.tokenizers.bpe_tokenizer import BPETokenizer
from torchaudio.transforms import RNNTLoss

frontend_choices = dict(batched_fbank=BatchedFbank)
encoder_choices = dict(conformer=ConformerEncoder)
decoder_choices = dict(rnn=RNNDecoder)


class RNNTbasedASR(nn.Module):
    def __init__(
        self,
        frontend_config_path: Path | str,
        encoder_config_path: Path | str,
        decoder_config_path: Path | str,
        joiner_config_path: Path | str,
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
        with open(decoder_config_path, "r") as f:
            c = yaml.safe_load(f)
            decoder_choice = c["decoder"]
            decoder_conf = c["decoder_conf"]
        with open(joiner_config_path, "r") as f:
            c = yaml.safe_load(f)
            joiner_conf = c["joiner_conf"]

        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        self.frontend = frontend_choices[frontend_choice](**frontend_conf)
        self.encoder = encoder_choices[encoder_choice](**encoder_conf)
        self.decoder = decoder_choices[decoder_choice](
            **decoder_conf, vocab_size=self.tokenizer.vocab_size
        )
        self.joiner = Joiner(**joiner_conf, vocab_size=self.tokenizer.vocab_size)

        self.criterion = RNNTLoss(blank=self.tokenizer.blank_token_id)

    def _add_blank(self, label_tokens: torch.Tensor) -> torch.Tensor:
        """add blank at the beginning of label tokens.

        Args:
            label_tokens (torch.Tensor): label tokens of shape (batch_size, seq_len2).

        Returns:
            torch.Tensor: label tokens tensor with blank token at the beginning.
        """
        return F.pad(label_tokens, (1, 0), value=self.tokenizer.blank_token_id)

    def forward(
        self,
        wavs: torch.Tensor,
        wav_lens: torch.Tensor,
        label_tokens: torch.Tensor,
        label_token_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward path.

        Args:
            wavs (torch.Tensor): waveform of shape (batch_size, seq_len1).
            wav_lens (torch.Tensor): lengths of wavs of shape (batch_size, )
            label_tokens (torch.Tensor): label tokens of shape (batch_size, seq_len2).
            label_token_lens (torch.Tensor): label token lengths of shape (batch_size, )

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - logit of shape (batch_size, T, U + 1, vocab_size)
            - encoder out lengths of shape (batch_size, )
            - decoder out lengths of shape (batch_size, )

            where T := max(encoder out lengths) and U := max(decoder out lengths)
        """
        x, xlens = self.frontend(wavs, wav_lens)

        # encoder_out: (bs, T, hid)
        encoder_out, encoder_out_lens = self.encoder(x, xlens)

        label_tokens = self._add_blank(label_tokens)
        label_token_lens = label_token_lens + 1

        # decoder_out: (bs, U + 1, hid)
        decoder_out, decoder_out_lens = self.decoder(label_tokens, label_token_lens)

        # joiner_out: (bs, T, U + 1, vocab_size)
        joiner_out = self.joiner(
            encoder_out, encoder_out_lens, decoder_out, decoder_out_lens
        )

        return joiner_out, encoder_out_lens, decoder_out_lens

    def _get_loss(
        self,
        joiner_out: torch.Tensor,
        label_tokens: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        label_lens: torch.Tensor,
    ) -> torch.Tensor:
        """

        Args:
            joiner_out (torch.Tensor): joiner output of shape (batch_size, T, U + 1, vocab_size).
            label_tokens (torch.Tensor): label tokens of shape (batch_size, U).
            encoder_out_lens (torch.Tensor): encoder output lengths of shape (batch_size, ), where T := max(encoder_out_lens)
            label_lens (torch.Tensor): label lengths of shape (batch_size, ), where U := max(decoder_out_lens).

        Returns:
            torch.Tensor: calculated rnnt loss.
        """
        # RNNTLoss requires float32 dtype for float tensors and int32 dtye for int tensors
        joiner_out = joiner_out.float()
        label_tokens = label_tokens.int()
        encoder_out_lens = encoder_out_lens.int()
        label_lens = label_lens.int()

        return self.criterion(joiner_out, label_tokens, encoder_out_lens, label_lens)

    def get_loss(
        self,
        wavs: torch.Tensor,
        wav_lens: torch.Tensor,
        label_tokens: torch.Tensor,
        label_token_lens: torch.Tensor,
    ):
        joint_out, encoder_out_lens, decoder_out_lens = self.forward(
            wavs, wav_lens, label_tokens, label_token_lens
        )
        return self._get_loss(
            joint_out, label_tokens, encoder_out_lens, label_token_lens
        )

    def inference_forward(
        self,
        wavs: torch.Tensor,
        wav_lens: torch.Tensor,
        inference_algorithm: Literal["greedy_search"] = "greedy_search",
    ) -> list[str]:
        """inference

        Args:
            wavs (torch.Tensor): audio tensor of shape (batch_size, seq_len)
            wav_lens (torch.Tensor): audio tensor lengts of sape (batch_size, )
            inference_algorithm (Literal["greedy_search"], optional):
                inference algorithm. Defaults to "greedy_search".

        Returns:
            list[torch.Tensor]: list of hypothesis.
        """
        x, xlens = self.frontend(wavs, wav_lens)
        x, xlens = self.encoder(x, xlens)

        hypothesis = []
        for encoder_out, encoder_out_len in zip(x, xlens):
            encoder_out = encoder_out[: encoder_out_len.item(), :]

            if inference_algorithm == "greedy_search":
                hyp_tokens = self.greedy_search(encoder_out).tolist()
                hypothesis.append(self.tokenizer.decode(hyp_tokens))
            else:
                raise NotImplementedError(
                    "inference algorithms other than greedy search is not implemented."
                )

        return hypothesis

    def greedy_search(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """greedy search for transducer.

        Args:
            encoder_out (torch.Tensor): encoder out for each sample of shape (seq_len, encoder_hidden_size).

        Returns:
            torch.Tensor: greedy search result of decoded tokens.
        """
        hypothesis = []
        decoder_hidden = torch.zeros(
            (self.decoder.num_layers, 1, self.decoder.hidden_size),
            device=encoder_out.device,
        )
        decoded_token = torch.tensor(
            [[self.tokenizer.blank_token_id]],
            device=encoder_out.device,
            dtype=torch.long,
        )

        # compute decoder_out_u from the initialized stage
        decoder_out_u, decoder_hidden = self.decoder.inference_forward(
            decoder_hidden, decoded_token
        )

        for encoder_out_t in encoder_out:  # encoder_out_t: (encoder_hidden_size, )
            encoder_out_t = rearrange(encoder_out_t, "hidden_size -> 1 1 hidden_size")

            while True:
                joiner_out_t_u = self.joiner(
                    encoder_out_t, decoder_out_u
                )  # (1, 1, 1) -> scholar

                decoded_token_item = joiner_out_t_u.argmax(dim=-1).item()

                # predicted token is <blank>
                if decoded_token_item == self.tokenizer.blank_token_id:
                    break
                # predicted token is not <blank>
                hypothesis.append(decoded_token_item)

                decoded_token = torch.Tensor([[decoded_token_item]]).long()
                decoder_out_u, decoder_hidden = self.decoder.inference_forward(
                    decoder_hidden, decoded_token
                )

        return torch.Tensor(hypothesis).long()
