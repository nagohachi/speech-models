from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
import yaml
from speech_models.modules.frontend.whisper_mel import WhisperFrontend
from speech_models.modules.others.speech_llm.projector import (
    ConvProjector,
    LinearProjector,
    MLPProjector,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

projector_choices = dict(linear=LinearProjector, mlp=MLPProjector, conv=ConvProjector)

SPEECH_PLACEHOLDER = "{speech}"


class SpeechLLM(nn.Module):
    """Speech LLM with Whisper encoder (frozen) -> Projector -> LLM.

    The Whisper encoder extracts speech features, the projector maps them to the
    LLM embedding space, and a chat-template prompt wraps the features::

        [prefix_tokens, speech_embeds, postfix_tokens, target_tokens]

    The ``prompt`` parameter is a user-role message containing ``{speech}`` as a
    placeholder.  It is formatted through ``tokenizer.apply_chat_template`` so
    that the model receives proper role / modality markers.

    Example config::

        prompt: "Transcribe the following audio.\\n{speech}"

    For Llama-3.2-Instruct this produces::

        <|begin_of_text|><|start_header_id|>user<|end_header_id|>

        Transcribe the following audio.
        [SPEECH]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    LoRA can be applied to the LLM for parameter-efficient fine-tuning.
    """

    def __init__(
        self,
        projector_config_path: Path | str,
        whisper_name: str = "base",
        llm_name_or_path: str = "meta-llama/Llama-3.2-1B",
        freeze_llm: bool = True,
        prompt: str | None = None,
        lora_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        with open(projector_config_path, "r") as f:
            c = yaml.safe_load(f)
            projector_choice = c["projector"]
            projector_conf = c.get("projector_conf", {})
            self.max_new_tokens: int = c.get("max_new_tokens", 256)

        # --- Whisper encoder (frozen) ---
        whisper_model = whisper.load_model(whisper_name)
        self.frontend = WhisperFrontend(n_mels=whisper_model.dims.n_mels)
        self.whisper_encoder = whisper_model.encoder
        self.whisper_encoder.eval()
        for p in self.whisper_encoder.parameters():
            p.requires_grad = False

        # --- LLM ---
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

        # --- Projector (trainable) ---
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
            wavs: (B, samples) audio at 16 kHz.
            wav_lens: (B,) sample counts.

        Returns:
            speech_embeds: (B, T_enc, llm_dim) projected speech features.
            speech_lens: (B,) valid frame counts after encoding + projection.
        """
        mels, mel_lens = self.frontend(wavs, wav_lens)

        with torch.no_grad():
            speech_features = self.whisper_encoder(mels)  # (B, 1500, whisper_dim)

        speech_lens = ((mel_lens + 1) / 2).long().clamp(max=speech_features.size(1))
        speech_embeds = self.projector(speech_features)

        if self._has_conv_projector:
            stride = self.projector.stride
            speech_lens = (
                ((speech_lens + stride - 1) / stride)
                .long()
                .clamp(max=speech_embeds.size(1))
            )
        elif hasattr(self.projector, "ds_k") and self.projector.ds_k > 1:
            speech_lens = (speech_lens // self.projector.ds_k).clamp(
                max=speech_embeds.size(1)
            )

        return speech_embeds, speech_lens

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

        An EOS token (``<|eot_id|>`` for Llama-3 Instruct) is appended to each
        label so that the model learns when to stop generating.  Labels are
        passed directly to the LLM so that HuggingFace can use its internal
        chunked cross-entropy, avoiding materialisation of the full
        ``(B, L, vocab_size)`` logits tensor.
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
            labels=labels,
        )
        return outputs.loss

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
