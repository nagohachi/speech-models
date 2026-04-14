from itertools import groupby
from pathlib import Path

import torch


class CharTokenizer:
    def __init__(self, chars: list[str]) -> None:
        self.pad_token = "<PAD>"
        self.blank_token = "<BLANK>"
        specials = [self.pad_token, self.blank_token]
        vocab = specials + [c for c in chars if c not in specials]
        self._token_to_id: dict[str, int] = {c: i for i, c in enumerate(vocab)}
        self._id_to_token: dict[int, str] = {i: c for c, i in self._token_to_id.items()}

    @property
    def pad_token_id(self) -> int:
        return self._token_to_id[self.pad_token]

    @property
    def blank_token_id(self) -> int:
        return self._token_to_id[self.blank_token]

    @classmethod
    def load(cls, vocab_path: str | Path) -> "CharTokenizer":
        chars = Path(vocab_path).read_text().strip().splitlines()
        return cls(chars)

    @classmethod
    def train(
        cls,
        text_path: str | Path | list[str | Path],
        output_dir: str | Path,
        model_prefix: str = "char_vocab",
    ) -> "CharTokenizer":
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{model_prefix}.txt"

        if not isinstance(text_path, list):
            text_path = [text_path]

        chars: set[str] = set()
        for p in text_path:
            chars.update(Path(p).read_text())

        chars.discard("\n")
        vocab = sorted(chars)
        instance = cls(vocab)
        save_path.write_text(
            "\n".join(
                instance._id_to_token[i] for i in range(len(instance._id_to_token))
            )
        )
        return instance

    def encode(self, text: str) -> list[int]:
        return [self._token_to_id[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self._id_to_token[i]
            for i in ids
            if i != self.pad_token_id and i != self.blank_token_id
        )

    def ctc_collapse(self, ids: list[int] | torch.Tensor) -> list[int]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        collapsed_ids = [k for k, _ in groupby(ids)]
        return [i for i in collapsed_ids if i != self.blank_token_id]

    def ctc_greedy_decode(self, ids: list[int] | torch.Tensor) -> str:
        return self.decode(self.ctc_collapse(ids))

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)
