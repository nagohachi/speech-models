from pathlib import Path

import sentencepiece as spm


class BPETokenizer:
    def __init__(self, model: spm.SentencePieceProcessor) -> None:
        self.sp = model
        self.pad_token = "<PAD>"
        self.blank_token = "<BLANK>"

    @property
    def unk_token(self) -> str:
        unk_id = self.sp.unk_id()
        return self.sp.id_to_piece(unk_id)  # type: ignore

    @property
    def pad_token_id(self) -> int:
        return self.sp.piece_to_id("<PAD>")  # type: ignore

    @property
    def blank_token_id(self) -> int:
        return self.sp.piece_to_id("<BLANK>")  # type: ignore

    @classmethod
    def load(cls, model_path: str | Path):
        model = spm.SentencePieceProcessor(model_file=str(model_path))  # type: ignore
        return cls(model)

    @classmethod
    def train(
        cls,
        text_path: str | Path,
        vocab_size: int,
        output_dir: str | Path,
        model_prefix: str = "bpe_model",
    ) -> "BPETokenizer":
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / model_prefix

        spm.SentencePieceTrainer.train(  # type: ignore
            input=str(text_path),
            model_prefix=str(save_path),
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type="bpe",
            user_defined_symbols=["<PAD>", "<BLANK>"],
        )

        return cls.load(f"{save_path}.model")

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)  # type: ignore

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)  # type: ignore

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()  # type: ignore
