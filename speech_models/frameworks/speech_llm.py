from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from speech_models.modules.encoder.wavlm.wavlm_encoder import WavLMEncoder
from speech_models.modules.encoder.whisper.whisper_encoder import WhisperEncoder
from speech_models.modules.frontend.huggingface_frontend import HuggingFaceFrontend
from speech_models.modules.frontend.whisper_frontend import WhisperFrontend
from speech_models.modules.others.speech_llm.projector import (
    ConvProjector,
    LinearProjector,
    MLPProjector,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

frontend_choices = dict(
    whisper_mel=WhisperFrontend,
    huggingface=HuggingFaceFrontend,
)
encoder_choices = dict(
    whisper=WhisperEncoder,
    wavlm=WavLMEncoder,
)
projector_choices = dict(linear=LinearProjector, mlp=MLPProjector, conv=ConvProjector)

SPEECH_PLACEHOLDER = "{speech}"


class SpeechLLM(nn.Module):
    """Speech LLM with a pluggable speech encoder -> Projector -> LLM.

    The speech encoder (frozen by default) extracts features, the projector
    maps them to the LLM embedding space, and a chat-template prompt wraps the
    features::

        [prefix_tokens, speech_embeds, postfix_tokens, target_tokens]

    The frontend / encoder pair is chosen via separate yaml files following
    the same convention as ``CTCBasedASR`` (``frontend:``+``frontend_conf:``
    and ``encoder:``+``encoder_conf:``). The LLM, projector, prompt template,
    LoRA config and loss-related hyperparameters live in a third yaml file
    (``speech_llm_config_path``).

    Example frontend yaml::

        frontend: whisper_mel
        frontend_conf:
          n_mels: 128

    Example encoder yaml::

        encoder: whisper
        encoder_conf:
          name: large-v3
          freeze: true

    Example speech_llm yaml::

        llm_name_or_path: meta-llama/Llama-3.2-3B-Instruct
        freeze_llm: true
        prompt: "Transcribe the following audio.\\n{speech}"
        projector: mlp
        projector_conf:
          output_dim: 3072
          hidden_dim: 3072
          downsample_k: 3

    The ``projector_conf.input_dim`` is auto-filled from
    ``encoder.hidden_size`` if omitted; if both are set and disagree a
    ``ValueError`` is raised.
    """

    def __init__(
        self,
        frontend_config_path: Path | str,
        encoder_config_path: Path | str,
        speech_llm_config_path: Path | str,
    ) -> None:
        super().__init__()

        # --- Frontend ---
        with open(frontend_config_path, "r") as f:
            c = yaml.safe_load(f)
            frontend_choice = c["frontend"]
            frontend_conf = c.get("frontend_conf", {})
        self.frontend = frontend_choices[frontend_choice](**frontend_conf)

        # --- Speech encoder ---
        with open(encoder_config_path, "r") as f:
            c = yaml.safe_load(f)
            encoder_choice = c["encoder"]
            encoder_conf = c.get("encoder_conf", {})
        self.encoder = encoder_choices[encoder_choice](**encoder_conf)

        # --- LLM + projector + prompt ---
        with open(speech_llm_config_path, "r") as f:
            c = yaml.safe_load(f)
            llm_name_or_path: str = c["llm_name_or_path"]
            freeze_llm: bool = c.get("freeze_llm", True)
            prompt: str | None = c.get("prompt")
            lora_config: dict[str, Any] | None = c.get("lora")
            projector_choice: str = c["projector"]
            projector_conf: dict[str, Any] = dict(c.get("projector_conf", {}))
            self.max_new_tokens: int = c.get("max_new_tokens", 256)
            self.label_smoothing: float = c.get("label_smoothing", 0.0)
            self.length_normalized_loss: bool = c.get("length_normalized_loss", False)

        self.llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

        # --- LoRA (optional) ---
        if lora_config is not None:
            from peft import LoraConfig, get_peft_model

            peft_cfg = LoraConfig(task_type="CAUSAL_LM", **lora_config)
            self.llm = get_peft_model(self.llm, peft_cfg)

        self.llm.gradient_checkpointing_enable()

        # --- Projector (trainable), with input_dim auto-fill ---
        if "input_dim" not in projector_conf:
            projector_conf["input_dim"] = self.encoder.hidden_size
        elif projector_conf["input_dim"] != self.encoder.hidden_size:
            raise ValueError(
                f"projector_conf.input_dim={projector_conf['input_dim']} does not "
                f"match encoder.hidden_size={self.encoder.hidden_size}"
            )
        self.projector = projector_choices[projector_choice](**projector_conf)
        self._has_conv_projector = isinstance(self.projector, ConvProjector)

        # --- End-of-turn token (for Llama-3 Instruct style models) ---
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id != self.tokenizer.unk_token_id:
            self.eot_id = eot_id
        else:
            self.eot_id = self.tokenizer.eos_token_id

        # --- Prompt template ---
        self._setup_prompt(prompt)

    # ------------------------------------------------------------------
    # Prompt setup
    # ------------------------------------------------------------------

    def _setup_prompt(self, prompt: str | None) -> None:
        """Build prefix/postfix token IDs from a chat-template prompt.

        ``prompt`` is a user-role message containing ``{speech}``.  The method
        applies the LLM's chat template, then splits on the placeholder to
        derive the prefix (everything before speech) and postfix (everything
        after speech, including the assistant header).
        """
        if prompt is None:
            self.prefix_token_ids = None
            self.postfix_token_ids = None
            return

        assert SPEECH_PLACEHOLDER in prompt, (
            f"prompt must contain '{SPEECH_PLACEHOLDER}'"
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (ValueError, AttributeError):
            # Fallback for tokenizers without chat template (e.g. GPT-2)
            formatted = prompt

        prefix_text, postfix_text = formatted.split(SPEECH_PLACEHOLDER)

        # apply_chat_template already includes BOS in the text;
        # use add_special_tokens=False to avoid a duplicate BOS.
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        postfix_ids = self.tokenizer.encode(postfix_text, add_special_tokens=False)

        self.register_buffer(
            "prefix_token_ids", torch.tensor(prefix_ids, dtype=torch.long)
        )
        self.register_buffer(
            "postfix_token_ids", torch.tensor(postfix_ids, dtype=torch.long)
        )

    def _get_prompt_embeds(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor | None, int, torch.Tensor | None, int]:
        embed_fn = self.llm.get_input_embeddings()
        prefix_embeds, prefix_len = None, 0
        postfix_embeds, postfix_len = None, 0

        if self.prefix_token_ids is not None:
            prefix_embeds = (
                embed_fn(self.prefix_token_ids).unsqueeze(0).expand(batch_size, -1, -1)
            )
            prefix_len = self.prefix_token_ids.size(0)
        if self.postfix_token_ids is not None:
            postfix_embeds = (
                embed_fn(self.postfix_token_ids).unsqueeze(0).expand(batch_size, -1, -1)
            )
            postfix_len = self.postfix_token_ids.size(0)

        return prefix_embeds, prefix_len, postfix_embeds, postfix_len

    # ------------------------------------------------------------------
    # Speech encoding
    # ------------------------------------------------------------------

    def encode_speech(
        self, wavs: torch.Tensor, wav_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode audio waveforms into LLM-space embeddings.

        Args:
            wavs: (B, samples) audio at the encoder's expected sample rate.
            wav_lens: (B,) sample counts.

        Returns:
            speech_embeds: (B, T_enc, llm_dim) projected speech features.
            speech_lens: (B,) valid frame counts after encoding + projection.
        """
        feats, xlens = self.frontend(wavs, wav_lens)         # (B, T, D_feat)
        feats, xlens = self.encoder(feats, xlens)            # (B, T_enc, hidden_size)
        speech_embeds = self.projector(feats)                # (B, T', llm_dim)

        if self._has_conv_projector:
            stride = self.projector.stride
            xlens = (
                ((xlens + stride - 1) // stride)
                .long()
                .clamp(max=speech_embeds.size(1))
            )
        elif hasattr(self.projector, "ds_k") and self.projector.ds_k > 1:
            xlens = (xlens // self.projector.ds_k).clamp(max=speech_embeds.size(1))

        return speech_embeds, xlens

    # ------------------------------------------------------------------
    # Build full input sequence
    # ------------------------------------------------------------------

    def _build_inputs(
        self,
        speech_embeds: torch.Tensor,
        speech_lens: torch.Tensor,
        label_tokens: torch.Tensor | None = None,
        label_token_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Assemble [prefix, speech, postfix, (text)] embeddings and mask.

        Returns:
            inputs_embeds, attention_mask, context_len (= P + S + Q)
        """
        B = speech_embeds.size(0)
        device = speech_embeds.device

        S = int(speech_lens.max().item())
        speech_embeds = speech_embeds[:, :S]

        prefix_embeds, P, postfix_embeds, Q = self._get_prompt_embeds(B, device)

        parts_embeds: list[torch.Tensor] = []
        parts_mask: list[torch.Tensor] = []

        if prefix_embeds is not None:
            parts_embeds.append(prefix_embeds)
            parts_mask.append(torch.ones(B, P, device=device))

        parts_embeds.append(speech_embeds)
        parts_mask.append(
            (torch.arange(S, device=device) < speech_lens.unsqueeze(1)).float()
        )

        if postfix_embeds is not None:
            parts_embeds.append(postfix_embeds)
            parts_mask.append(torch.ones(B, Q, device=device))

        ctx_len = P + S + Q

        if label_tokens is not None:
            assert label_token_lens is not None
            text_embeds = self.llm.get_input_embeddings()(label_tokens)
            T = text_embeds.size(1)
            parts_embeds.append(text_embeds)
            parts_mask.append(
                (torch.arange(T, device=device) < label_token_lens.unsqueeze(1)).float()
            )

        inputs_embeds = torch.cat(parts_embeds, dim=1)
        attention_mask = torch.cat(parts_mask, dim=1).long()

        return inputs_embeds, attention_mask, ctx_len

    # ------------------------------------------------------------------
    # Forward / Loss
    # ------------------------------------------------------------------

    def _build_labels(
        self, ctx_len: int, label_tokens: torch.Tensor, label_token_lens: torch.Tensor
    ) -> torch.Tensor:
        """Build labels with -100 for context (prefix+speech+postfix) positions."""
        B, T = label_tokens.shape
        device = label_tokens.device
        ignore = torch.full((B, ctx_len), -100, device=device, dtype=label_tokens.dtype)
        labels = torch.cat([ignore, label_tokens], dim=1)
        text_mask = torch.arange(T, device=device) < label_token_lens.unsqueeze(1)
        labels[:, ctx_len:][~text_mask] = -100
        return labels

    def get_loss(
        self,
        wavs: torch.Tensor,
        wav_lens: torch.Tensor,
        label_tokens: torch.Tensor,
        label_token_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss on the text (assistant) portion only.

        Supports label smoothing and length-normalized loss (per-token average
        instead of per-batch average).
        """
        # --- Append EOS to label tokens (cf. ESPnet add_sos_eos) ---
        B, T = label_tokens.shape
        eos_col = torch.full(
            (B, 1), self.eot_id, device=label_tokens.device, dtype=label_tokens.dtype
        )
        label_tokens = torch.cat([label_tokens, eos_col], dim=1)
        label_token_lens = label_token_lens + 1

        speech_embeds, speech_lens = self.encode_speech(wavs, wav_lens)
        inputs_embeds, attention_mask, ctx_len = self._build_inputs(
            speech_embeds, speech_lens, label_tokens, label_token_lens
        )
        labels = self._build_labels(ctx_len, label_tokens, label_token_lens)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # Shifted cross-entropy with label smoothing
        shift_logits = logits[:, :-1].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
            reduction="sum" if self.length_normalized_loss else "mean",
        )

        if self.length_normalized_loss:
            num_tokens = (shift_labels != -100).sum()
            loss = loss / num_tokens

        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def inference_forward(
        self,
        wavs: torch.Tensor,
        wav_lens: torch.Tensor,
        max_new_tokens: int | None = None,
        num_beams: int = 1,
    ) -> list[str]:
        """Generate text from speech via autoregressive decoding."""
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        speech_embeds, speech_lens = self.encode_speech(wavs, wav_lens)
        inputs_embeds, attention_mask, _ = self._build_inputs(
            speech_embeds, speech_lens
        )

        eos_token_ids = list(
            {self.tokenizer.eos_token_id, self.eot_id}
        )

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            eos_token_id=eos_token_ids,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
