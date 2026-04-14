import re

from phonemizer.backend import EspeakBackend
from unidecode import unidecode

_whitespace_re = re.compile(r"\s+")
_brackets_re = re.compile(r"[\[\]\(\)\{\}]")
_abbreviations = [
    (re.compile(rf"\b{abbr}\.", re.IGNORECASE), expansion)
    for abbr, expansion in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


class PhonemizerG2P:
    def __init__(
        self,
        preserve_punctuation: bool = False,
        with_stress: bool = False,
        language_switch: str = "keep-flags",
        text_cleaners: list[str] | None = None,
    ) -> None:
        self.preserve_punctuation = preserve_punctuation
        self.with_stress = with_stress
        self.language_switch = language_switch
        self.text_cleaners = text_cleaners or []

    def _clean_text(self, text: str, lang: str) -> str:
        for cleaner in self.text_cleaners:
            if cleaner == "convert_to_ascii":
                text = unidecode(text)
            elif cleaner == "lowercase":
                text = text.lower()
            elif cleaner == "expand_abbreviations":
                if lang.startswith("en"):
                    for regex, replacement in _abbreviations:
                        text = re.sub(regex, replacement, text)
            elif cleaner == "collapse_whitespace":
                text = re.sub(_whitespace_re, " ", text)
        return text

    @staticmethod
    def _postprocess(text: str) -> str:
        text = re.sub(_brackets_re, "", text)
        text = re.sub(_whitespace_re, " ", text)
        return text.strip()

    def to_phoneme_batch(
        self, lang: str, grapheme_list: list[str], chunk_size: int = 128, njobs: int = 8
    ) -> list[str]:
        backend = EspeakBackend(
            language=lang,
            preserve_punctuation=self.preserve_punctuation,
            with_stress=self.with_stress,
            language_switch=self.language_switch,
        )
        cleaned = [self._clean_text(t, lang) for t in grapheme_list]
        phonemes: list[str] = []
        for i in range(0, len(cleaned), chunk_size):
            chunk = cleaned[i : i + chunk_size]
            phonemes.extend(backend.phonemize(chunk, njobs=njobs))
        return [self._postprocess(p) for p in phonemes]
