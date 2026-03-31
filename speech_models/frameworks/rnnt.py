from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from speech_models.modules.decoder.rnn import RNNDecoder
from speech_models.modules.encoder.conformer.conformer_encoder import ConformerEncoder
from speech_models.modules.frontend.global_mvn import GlobalMVN
from speech_models.modules.frontend.log_mel import BatchedFbank
from speech_models.modules.others.rnnt.joiner import Joiner
from speech_models.tokenizers.bpe_tokenizer import BPETokenizer
from torchaudio.transforms import RNNTLoss

frontend_choices = dict(batched_fbank=BatchedFbank)
encoder_choices = dict(conformer=ConformerEncoder)
decoder_choices = dict(rnn=RNNDecoder)
normalize_choices = dict(global_mvn=GlobalMVN)


class RNNTbasedASR(nn.Module):
    def __init__(
        self,
        frontend_config_path: Path | str,
        encoder_config_path: Path | str,
        decoder_config_path: Path | str,
        joiner_config_path: Path | str,
        tokenizer: BPETokenizer,
        loss_type: Literal["standard", "pruned"] = "standard",
        prune_range: int = 5,
        simple_loss_scaling: float = 0.5,
        warmup_steps: int = 5000,
        use_torch_compile: bool = False,
        feats_stats_path: Path | str | None = None,
    ) -> None:
        super().__init__()

        with open(frontend_config_path, "r") as f:
            c = yaml.safe_load(f)
            frontend_choice = c["frontend"]
            frontend_conf = c["frontend_conf"]
            normalize_choice = c.get("normalize")
            normalize_conf = c.get("normalize_conf", {})
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
        self.loss_type = loss_type
        self.prune_range = prune_range
        self.simple_loss_scaling = simple_loss_scaling
        self.warmup_steps = warmup_steps
        self.steps_num = 0

        self.frontend = frontend_choices[frontend_choice](**frontend_conf)
        if normalize_choice is not None:
            if feats_stats_path is not None:
                normalize_conf["stats_file"] = str(feats_stats_path)
            self.normalize = normalize_choices[normalize_choice](**normalize_conf)
        else:
            self.normalize = None
        self.encoder = encoder_choices[encoder_choice](**encoder_conf)
        self.decoder = decoder_choices[decoder_choice](
            **decoder_conf,
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.joiner = Joiner(**joiner_conf, vocab_size=self.tokenizer.vocab_size)

        if loss_type == "standard":
            self.criterion = RNNTLoss(blank=self.tokenizer.blank_token_id)
        else:
            self.am_proj = nn.Linear(encoder_conf["hidden_size"], self.vocab_size)
            self.lm_proj = nn.Linear(decoder_conf["hidden_size"], self.vocab_size)

        if use_torch_compile:
            self.forward = torch.compile(self.forward, dynamic=True)
            if loss_type == "pruned":
                self._pruned_forward = torch.compile(self._pruned_forward, dynamic=True)

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
        if self.normalize is not None:
            x, xlens = self.normalize(x, xlens)

        # encoder_out: (bs, T, hid)
        encoder_out, encoder_out_lens = self.encoder(x, xlens)

        label_tokens = self._add_blank(label_tokens)
        label_token_lens = label_token_lens + 1

        # decoder_out: (bs, U + 1, hid)
        decoder_out, decoder_out_lens = self.decoder(label_tokens, label_token_lens)

        # joiner_out: (bs, T, U + 1, vocab_size)
        joiner_out = self.joiner(encoder_out, decoder_out)

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

    def _pruned_forward(
        self,
        wavs: torch.Tensor,
        wav_lens: torch.Tensor,
        label_tokens: torch.Tensor,
        label_token_lens: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compilable forward portion of the pruned loss path.

        Returns:
            tuple: (encoder_out, decoder_out, encoder_out_lens, label_token_lens, am, lm)
        """
        x, xlens = self.frontend(wavs, wav_lens)
        if self.normalize is not None:
            x, xlens = self.normalize(x, xlens)
        encoder_out, encoder_out_lens = self.encoder(x, xlens)

        label_tokens_with_blank = self._add_blank(label_tokens)
        decoder_out_lens = label_token_lens + 1

        decoder_out, _ = self.decoder(label_tokens_with_blank, decoder_out_lens)

        am = self.am_proj(encoder_out)  # (B, T, V)
        lm = self.lm_proj(decoder_out)  # (B, U+1, V)

        return encoder_out, decoder_out, encoder_out_lens, label_token_lens, am, lm

    def _get_pruned_loss(
        self,
        wavs: torch.Tensor,
        wav_lens: torch.Tensor,
        label_tokens: torch.Tensor,
        label_token_lens: torch.Tensor,
    ) -> torch.Tensor:
        import k2

        self.steps_num += 1

        if self.steps_num < self.warmup_steps:
            ratio = self.steps_num / self.warmup_steps
            pruned_loss_scaling = 0.1 + 0.9 * ratio
            cur_simple_loss_scaling = 1.0 - ratio * (1.0 - self.simple_loss_scaling)
        else:
            pruned_loss_scaling = 1.0
            cur_simple_loss_scaling = self.simple_loss_scaling

        encoder_out, decoder_out, encoder_out_lens, label_token_lens, am, lm = (
            self._pruned_forward(wavs, wav_lens, label_tokens, label_token_lens)
        )

        # boundary: (B, 4)
        # each row is [begin_symbol, begin_frame, end_symbol, end_frame]
        batch_size = encoder_out.size(0)
        boundary = torch.zeros(
            (batch_size, 4), dtype=torch.int64, device=encoder_out.device
        )
        boundary[:, 2] = label_token_lens
        boundary[:, 3] = encoder_out_lens

        with torch.amp.autocast("cuda", enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=label_tokens.long(),
                termination_symbol=self.tokenizer.blank_token_id,
                lm_only_scale=0.0,
                am_only_scale=0.0,
                boundary=boundary,
                reduction="mean",
                return_grad=True,
            )

        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=self.prune_range,
        )

        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.lin_enc(encoder_out),
            lm=self.joiner.lin_dec(decoder_out),
            ranges=ranges,
        )

        logits = self.joiner.forward_pruned(
            am_pruned, lm_pruned
        )  # (B, T, prune_range, V)

        with torch.amp.autocast("cuda", enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=label_tokens.long(),
                ranges=ranges,
                termination_symbol=self.tokenizer.blank_token_id,
                boundary=boundary,
                reduction="mean",
            )

        return cur_simple_loss_scaling * simple_loss + pruned_loss_scaling * pruned_loss

    def get_loss(
        self,
        wavs: torch.Tensor,
        wav_lens: torch.Tensor,
        label_tokens: torch.Tensor,
        label_token_lens: torch.Tensor,
    ):
        if self.loss_type == "pruned":
            return self._get_pruned_loss(wavs, wav_lens, label_tokens, label_token_lens)
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
        inference_algorithm: Literal["greedy_search", "beam_search"] = "greedy_search",
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
        if self.normalize is not None:
            x, xlens = self.normalize(x, xlens)
        x, xlens = self.encoder(x, xlens)

        hypothesis = []
        for encoder_out, encoder_out_len in zip(x, xlens):
            encoder_out = encoder_out[: encoder_out_len.item(), :]

            if inference_algorithm == "greedy_search":
                hyp_tokens = self.greedy_search(encoder_out).tolist()
                hypothesis.append(self.tokenizer.decode(hyp_tokens))
            else:
                hyp_tokens = self.beam_search(encoder_out).tolist()
                hypothesis.append(self.tokenizer.decode(hyp_tokens))

        return hypothesis

    def greedy_search(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """greedy search for transducer.

        Args:
            encoder_out (torch.Tensor): encoder out for each sample of shape (seq_len, encoder_hidden_size).

        Returns:
            torch.Tensor: greedy search result of decoded tokens.
        """
        hypothesis = []
        h_0 = torch.zeros(
            (self.decoder.num_layers, 1, self.decoder.hidden_size),
            device=encoder_out.device,
        )

        match self.decoder.rnn_type:
            case "lstm":
                c_0 = torch.zeros_like(h_0)
                decoder_hidden = (h_0, c_0)
            case "rnn":
                decoder_hidden = h_0

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

            # TODO: set in command line
            max_emit_per_frame = 10
            emit_count = 0

            while True:
                joiner_out_t_u = self.joiner(
                    encoder_out_t, decoder_out_u
                )  # (1, 1, 1) -> scholar

                decoded_token_item = joiner_out_t_u.argmax(dim=-1).item()

                # predicted token is <blank>
                if (
                    decoded_token_item == self.tokenizer.blank_token_id
                    or emit_count >= max_emit_per_frame
                ):
                    break
                # predicted token is not <blank>
                hypothesis.append(decoded_token_item)

                decoded_token = torch.tensor(
                    [[decoded_token_item]], device=encoder_out.device, dtype=torch.long
                )
                decoder_out_u, decoder_hidden = self.decoder.inference_forward(
                    decoder_hidden, decoded_token
                )

                emit_count += 1

        return torch.Tensor(hypothesis).long()

    def beam_search(
        self, encoder_out: torch.Tensor, beam_size: int = 6
    ) -> torch.Tensor:
        """batched beam search for transducer to improve inference speed."""
        import torch.nn.functional as F

        device = encoder_out.device
        blank_id = self.tokenizer.blank_token_id
        vocab_size = self.vocab_size

        # --- 初期化 ---
        h_0 = torch.zeros(
            (self.decoder.num_layers, 1, self.decoder.hidden_size),
            device=device,
        )

        if self.decoder.rnn_type == "lstm":
            c_0 = torch.zeros_like(h_0)
            hidden = (h_0, c_0)
        else:
            hidden = h_0

        dec_token = torch.tensor([[blank_id]], device=device, dtype=torch.long)
        dec_out, hidden = self.decoder.inference_forward(hidden, dec_token)

        beam = [{"tokens": [], "logp": 0.0, "hidden": hidden, "dec_out": dec_out}]

        for encoder_out_t in encoder_out:  # (hidden_size, )
            enc_out_t = rearrange(encoder_out_t, "hidden_size -> 1 1 hidden_size")

            A = beam
            B = []
            max_emit = 10

            for _ in range(max_emit):
                if not A:
                    break

                batch_A = len(A)

                batched_dec_out = torch.cat([hyp["dec_out"] for hyp in A], dim=0)
                batched_enc_out = enc_out_t.expand(batch_A, -1, -1)

                joiner_out = self.joiner(batched_enc_out, batched_dec_out)
                joiner_out = joiner_out.view(batch_A, vocab_size)

                log_probs = F.log_softmax(joiner_out, dim=-1)

                prev_logp = torch.tensor(
                    [hyp["logp"] for hyp in A], device=device
                ).unsqueeze(1)
                total_logp = prev_logp + log_probs  # (batch_A, vocab_size)

                num_candidates = min(beam_size, total_logp.numel())
                topk_logp, topk_indices = total_logp.view(-1).topk(num_candidates)

                next_A_cands = []
                for i in range(num_candidates):
                    logp = topk_logp[i].item()
                    flat_idx = topk_indices[i].item()

                    hyp_idx = flat_idx // vocab_size
                    token_id = flat_idx % vocab_size

                    base_hyp = A[hyp_idx]

                    if token_id == blank_id:
                        B.append(
                            {
                                "tokens": base_hyp["tokens"],
                                "logp": logp,
                                "hidden": base_hyp["hidden"],
                                "dec_out": base_hyp["dec_out"],
                            }
                        )
                    else:
                        next_A_cands.append(
                            {
                                "hyp_idx": hyp_idx,
                                "token_id": token_id,
                                "logp": logp,
                                "base_hyp": base_hyp,
                            }
                        )

                B = sorted(B, key=lambda x: x["logp"], reverse=True)[:beam_size]

                next_A_cands = sorted(
                    next_A_cands, key=lambda x: x["logp"], reverse=True
                )[:beam_size]

                if not next_A_cands:
                    break

                batch_next_A = len(next_A_cands)
                batched_tokens = torch.tensor(
                    [[cand["token_id"]] for cand in next_A_cands],
                    device=device,
                    dtype=torch.long,
                )  # (batch_next_A, 1)

                if self.decoder.rnn_type == "lstm":
                    h_t = torch.cat(
                        [cand["base_hyp"]["hidden"][0] for cand in next_A_cands], dim=1
                    )
                    c_t = torch.cat(
                        [cand["base_hyp"]["hidden"][1] for cand in next_A_cands], dim=1
                    )
                    batched_hidden = (h_t, c_t)
                else:
                    batched_hidden = torch.cat(
                        [cand["base_hyp"]["hidden"] for cand in next_A_cands], dim=1
                    )

                new_dec_out, new_hidden = self.decoder.inference_forward(
                    batched_hidden, batched_tokens
                )

                A = []
                for i, cand in enumerate(next_A_cands):
                    if self.decoder.rnn_type == "lstm":
                        h_i = new_hidden[0][:, i : i + 1, :].contiguous()
                        c_i = new_hidden[1][:, i : i + 1, :].contiguous()
                        hid_i = (h_i, c_i)
                    else:
                        hid_i = new_hidden[:, i : i + 1, :].contiguous()

                    A.append(
                        {
                            "tokens": cand["base_hyp"]["tokens"] + [cand["token_id"]],
                            "logp": cand["logp"],
                            "hidden": hid_i,
                            "dec_out": new_dec_out[i : i + 1],  # (1, 1, hidden_size)
                        }
                    )

            beam = sorted(B + A, key=lambda x: x["logp"], reverse=True)[:beam_size]

        return torch.tensor(beam[0]["tokens"], dtype=torch.long, device=device)
