from phonemizer.backend import EspeakBackend


class PhonemizerG2P:
    def to_phoneme_batch(
        self, lang: str, grapheme_list: list[str], chunk_size: int = 128, njobs: int = 8
    ) -> list[str]:
        backend = EspeakBackend(language=lang)
        phonemes = []
        for i in range(0, len(grapheme_list), chunk_size):
            grapheme_chunk = grapheme_list[i : i + chunk_size]
            phonemes.extend(backend.phonemize(grapheme_chunk, njobs=njobs))
        return phonemes
